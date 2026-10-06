"""앙상블 판정: 위치는 YOLO, 합격·재검사·불합격 판정은 YOLO + 조각 분류 CNN 점수를 합쳐서.

근거: judge.py 결과에서 사진 단위 구분력은 CNN(AUROC 0.9998)이, 위치 정확도는 YOLO(IoU50 F1 0.950)가 낫다.

  1. 뒤집기 TTA: 사진을 그대로·좌우·상하·상하좌우 4가지로 뒤집어 두 모델에 넣고, 사진 최고 점수를 모은다
     (학습 때도 뒤집기 증강을 써서 물리적으로 자연스러운 변형). 모델마다 4개 점수의 평균을 쓴다.
  2. 보정: 모델마다 온도 스케일링(val NLL 최소, judge.fit_calibration)으로 확률을 맞춘 뒤
     로짓 평균 → 앙상블 확률.
  3. 불확실성: 8개(2모델 × 4뒤집기) 보정 확률의 표준편차. 판정이 '합격'이어도 흔들림이 크면 재검사로 돌린다.
     흔들림 기준선 = val 가짜 정상 중 합격 판정 사진의 흔들림 95% 분위 (정상 재검사 추가를 약 5%로 묶음).
  4. 합격선·불합격선은 judge.py 와 같은 규칙(불량 99% / 정상 95% 분위). 모두 val 에서 정하고 test 는 마지막에 한 번.
Wang et al. (2019, Neurocomputing) 의 시험 시 증강 불확실성을 판정 단계에 옮긴 것이다.

실행: .venv\\Scripts\\python.exe src\\ensemble.py --yolo ratio3_e100 --cnn cnn_aug
결과: results/ensemble/ (scores_*.csv, summary.json, compare.csv, 그래프)
"""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cnn as C
from judge import NORMAL_Q, RECALL_TARGET, ece, fit_calibration, logit, tiers
from train_yolo import weights_path

ROOT = Path(__file__).resolve().parents[1]   # 저장소 최상위 폴더
# 뒤집기 4가지. 배열은 (높이, 너비) 라서 축 1 을 뒤집으면 좌우, 축 0 을 뒤집으면 상하다
FLIPS = {"원본": lambda a: a, "좌우": lambda a: a[:, ::-1], "상하": lambda a: a[::-1, :], "상하좌우": lambda a: a[::-1, ::-1]}
UNC_Q = 0.95   # 흔들림 기준선: val 에서 합격 판정된 가짜 정상의 흔들림 분포에서 이 분위


def tta_scores(df, yolo, cnn_name):
    """사진마다 4가지 뒤집기를 두 모델에 넣어 사진 최고 점수 8개를 구한다.

    df: 판정 세트 표(열 path, machine). yolo · cnn_name: 모델 이름.
    반환: df 에 열 Y_<뒤집기>, C_<뒤집기> (각 0~1, 검출이 없으면 0.0) 를 붙인 것. Y 는 YOLO, C 는 CNN 이다.
    """
    from ultralytics import YOLO
    ym = YOLO(str(weights_path(yolo)))
    cm, boxes, dev = C.load(cnn_name)
    out = {f"{m}_{k}": [] for m in ["Y", "C"] for k in FLIPS}
    for r in tqdm(list(df.itertuples()), desc="TTA"):
        g = np.asarray(Image.open(r.path).convert("L"))
        # 뒤집은 배열은 원본을 거꾸로 읽는 뷰라서, 메모리에 연속으로 놓인 사본으로 만든다
        arrs = {k: np.ascontiguousarray(f(g)) for k, f in FLIPS.items()}
        # 배열을 직접 넘길 때는 3채널(BGR)로 바꾼다. 4장을 한 묶음으로 추론한다
        res = ym.predict([cv2.cvtColor(a, cv2.COLOR_GRAY2BGR) for a in arrs.values()], imgsz=640, conf=0.001,
                         max_det=100, verbose=False)
        for k, rr in zip(arrs, res):
            c = rr.boxes.conf.cpu().numpy()
            out[f"Y_{k}"].append(float(c.max()) if len(c) else 0.0)
            # 최고 점수만 쓰므로 뒤집힌 좌표를 되돌릴 필요가 없다
            d = C.detect(cm, arrs[k], boxes[int(r.machine)], dev)
            out[f"C_{k}"].append(float(d["score"].max()) if len(d) else 0.0)
    for k, v in out.items():
        df[k] = v
    return df


def thresholds(s, y):
    """judge.py 와 같은 규칙의 기준선. s: (N,) 점수, y: (N,) 정답(불량 1 · 정상 0). 반환: (합격선, 불합격선).

    합격선 = 불량 점수의 하위 1% 분위(보간 없이 아래쪽 값), 불합격선 = 정상 점수의 95% 분위.
    불합격선이 합격선보다 낮으면 합격선에 맞춘다.
    """
    t_low = float(np.quantile(s[y == 1], 1 - RECALL_TARGET, method="lower"))
    t_high = float(np.quantile(s[y == 0], NORMAL_Q))
    return t_low, max(t_low, t_high)


def rates(test, col):
    """영상 종류(kind)별 판정 비율. col: 판정 열 이름. 반환: {종류: {판정: 비율 0~1}}."""
    tab = test.groupby("kind")[col].value_counts(normalize=True).unstack(fill_value=0)
    return {k: {c: round(float(v), 4) for c, v in row.items()} for k, row in tab.iterrows()}


def main():
    """TTA 점수 계산 → 보정 · 앙상블 확률 → 판정 방식 6가지를 val 기준선으로 test 에서 비교.

    results/ensemble/ 에 scores_<분할>.csv, weight_grid_val.csv, compare.csv, judged_test.csv, summary.json, scatter.png 를 남긴다.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo", default="ratio3_e100")
    ap.add_argument("--cnn", default="cnn_aug")
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args()
    data, out = ROOT / "data", ROOT / "results" / "ensemble"
    out.mkdir(parents=True, exist_ok=True)

    # 1) TTA 점수. --reuse 이면 저장된 점수를 다시 쓴다
    sets = {}
    for s in ["val", "test"]:
        f = out / f"scores_{s}.csv"
        if args.reuse and f.exists():
            sets[s] = pd.read_csv(f)
            continue
        df = pd.read_csv(data / f"judge_{s}.csv")
        df["y"] = (df["kind"] != "normal").astype(int)   # 불량(실제 · 합성) 1, 가짜 정상 0
        sets[s] = tta_scores(df, args.yolo, args.cnn)
        sets[s].to_csv(f, index=False, encoding="utf-8-sig")
    val, test = sets["val"], sets["test"]
    ks = list(FLIPS)

    # 2) 보정과 앙상블. 열 이름: Y = YOLO, C = CNN, E = 두 모델 앙상블, W = 가중 앙상블
    #    <모델>_tta = 4개 뒤집기 점수의 평균, _p = 보정 확률, _z = 보정 확률의 로짓, _unc = 흔들림
    # 모델별 보정 (TTA 평균 점수로 맞춘 온도를 각 뒤집기 점수에도 똑같이 적용)
    cal, prm = {}, {}
    for m in ["Y", "C"]:
        for df in (val, test):
            df[f"{m}_tta"] = df[[f"{m}_{k}" for k in ks]].mean(1)
        cal[m], prm[m] = fit_calibration(val[f"{m}_tta"].to_numpy(), val["y"].to_numpy(), "temp")
    for df in (val, test):
        for m in ["Y", "C"]:
            df[f"{m}_p"] = cal[m](df[f"{m}_tta"].to_numpy())
            for k in ks:
                df[f"{m}_{k}_p"] = cal[m](df[f"{m}_{k}"].to_numpy())
        # 앙상블 확률 = 두 모델 보정 확률의 로짓을 평균해 다시 시그모이드
        df["E_p"] = 1 / (1 + np.exp(-(logit(df["Y_p"].to_numpy()) + logit(df["C_p"].to_numpy())) / 2))
        # 흔들림 = 보정 확률 8개(2모델 × 4뒤집기)의 표준편차 (pandas 기본값인 표본 표준편차)
        df["E_unc"] = df[[f"{m}_{k}_p" for m in ["Y", "C"] for k in ks]].std(1)

    # 가중 앙상블: YOLO 비중 w 와 TTA 사용 여부를 val 정상 합격률(불량 99% 잡는 합격선 기준)로만 고른다
    # _z 는 TTA 평균 점수의 로짓, _z0 는 뒤집지 않은 원본 점수만 보정한 로짓이다
    for df in (val, test):
        for m in ["Y", "C"]:
            df[f"{m}_z"] = logit(df[f"{m}_p"].to_numpy())
            df[f"{m}_z0"] = logit(cal[m](df[f"{m}_원본"].to_numpy()))
    grid = []
    for suf in ["z", "z0"]:
        for w in [0, 0.25, 0.5, 0.75, 1]:   # w = 0 은 CNN 만, w = 1 은 YOLO 만
            sv = w * val[f"Y_{suf}"] + (1 - w) * val[f"C_{suf}"]
            t_low, _ = thresholds(sv.to_numpy(), vy := val["y"].to_numpy())
            # 정상 합격률 = 가짜 정상 중 합격선 아래인 비율. 합격선은 조합마다 val 불량 99% 규칙으로 다시 정한다
            grid.append(dict(tta=suf == "z", w=w, val_정상합격=round(float((sv[vy == 0] < t_low).mean()), 4)))
    grid = pd.DataFrame(grid)
    # 정상 합격률이 가장 높은 조합. 같으면 TTA 를 안 쓰는 쪽, 그다음은 w 가 작은 쪽(안정 정렬)을 고른다
    best = grid.sort_values(["val_정상합격", "tta"], ascending=[False, True], kind="stable").iloc[0]
    suf = "z" if best["tta"] else "z0"
    for df in (val, test):
        df["W_p"] = 1 / (1 + np.exp(-(best["w"] * df[f"Y_{suf}"] + (1 - best["w"]) * df[f"C_{suf}"])))
    grid.to_csv(out / "weight_grid_val.csv", index=False, encoding="utf-8-sig")

    # 3) 방식별 판정 비교: 기준선은 val 에서 정하고 test 에서 잰다
    # 비교할 판정 방식: (이름, 점수 열, 불확실성 사용 여부)
    # "YOLO 단독" 만 보정하지 않은 원본 점수를 쓴다
    ways = [("YOLO 단독", "Y_원본", False), ("YOLO + TTA", "Y_p", False), ("CNN + TTA", "C_p", False),
            ("앙상블", "E_p", False), ("앙상블 + 불확실성", "E_p", True),
            (f"가중 앙상블 (val 선택: YOLO {best['w']:g}, TTA {'사용' if best['tta'] else '안 씀'})", "W_p", False)]
    summary, comp = {"보정": prm}, []
    vy, ty = val["y"].to_numpy(), test["y"].to_numpy()
    for name, col, use_u in ways:
        vs, ts = val[col].to_numpy(), test[col].to_numpy()
        t_low, t_high = thresholds(vs, vy)
        vt, tt = tiers(vs, t_low, t_high), tiers(ts, t_low, t_high)
        u_thr = None
        if use_u:
            # 흔들림 기준선: val 에서 합격 판정된 가짜 정상의 흔들림 95% 분위. 합격이어도 이 이상 흔들리면 재검사로 돌린다
            u_thr = float(np.quantile(val.loc[(vy == 0) & (vt == "합격"), "E_unc"], UNC_Q))
            tt = np.where((tt == "합격") & (test["E_unc"].to_numpy() >= u_thr), "재검사", tt)
        test[f"판정_{name}"] = tt
        r = rates(test, f"판정_{name}")
        sn = test[test["kind"] == "synth_ng"]
        miss = sn[sn[f"판정_{name}"] == "합격"]   # 합격으로 나간 합성 불량 = 놓침
        # AUROC_실제 는 (실제 불량 대 가짜 정상), AUROC_합성 은 (합성 불량 대 가짜 정상) 으로 잰다
        row = {"방식": name, "합격선": round(t_low, 4), "불합격선": round(t_high, 4),
               "흔들림선": round(u_thr, 4) if u_thr is not None else None,
               "정상_합격": r["normal"].get("합격", 0), "정상_재검사": r["normal"].get("재검사", 0),
               "정상_불합격": r["normal"].get("불합격", 0), "실제불량_불합격": r["real_ng"].get("불합격", 0),
               "합성불량_놓침수": int(len(miss)), "합성불량_놓침률": round(len(miss) / len(sn), 4),
               "AUROC_실제": round(float(roc_auc_score(test.loc[test.kind != "synth_ng", "y"],
                                                     test.loc[test.kind != "synth_ng", col])), 4),
               "AUROC_합성": round(float(roc_auc_score(test.loc[test.kind != "real_ng", "y"],
                                                     test.loc[test.kind != "real_ng", col])), 4)}
        if col != "Y_원본":   # 보정한 확률에만 ECE 를 적는다
            row["ECE"] = round(ece(ts, ty), 4)
        if len(miss):
            row["놓침_대비중앙"] = round(float(miss["c_meas"].median()), 3)
            row["놓침_지름중앙"] = round(float(miss["d"].median()), 2)
        comp.append(row)
        print(row)
    comp = pd.DataFrame(comp)
    comp.to_csv(out / "compare.csv", index=False, encoding="utf-8-sig")
    summary["비교"] = comp.to_dict("records")
    # 불확실성으로 재검사로 돌린 사진 중 실제 불량 비율 (돌린 효과)
    moved = test[(test["판정_앙상블"] == "합격") & (test["판정_앙상블 + 불확실성"] == "재검사")]
    summary["흔들림으로_재검사"] = {"수": int(len(moved)), "그중_불량": int(moved["y"].sum()),
                              "종류": moved["kind"].value_counts().to_dict()}
    test.to_csv(out / "judged_test.csv", index=False, encoding="utf-8-sig")
    json.dump(summary, open(out / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=float)
    print(summary["흔들림으로_재검사"])

    # 그래프: 앙상블 확률 × 흔들림, 종류별 색 (test)
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=150)
    for kind, c, lab in [("normal", "#9aa5b1", "가짜 정상"), ("synth_ng", "#0a8f86", "합성 불량"), ("real_ng", "#c96f24", "실제 불량")]:
        q = test[test["kind"] == kind]
        ax.scatter(q["E_p"], q["E_unc"], s=9, alpha=.7, color=c, label=lab, edgecolors="none")
    row = comp[comp["방식"] == "앙상블 + 불확실성"].iloc[0]
    # 세로 실선 = 합격선, 세로 점선 = 불합격선, 가로선 = 흔들림 기준선
    ax.axvline(row["합격선"], color="#1d2733", lw=1)
    ax.axvline(row["불합격선"], color="#1d2733", lw=1, ls="--")
    ax.axhline(row["흔들림선"], color="#8c5bb5", lw=1)
    # 가로축은 로짓 눈금이다 (0 과 1 근처가 넓게 펴진다)
    ax.set(xscale="logit", xlabel="앙상블 불량 확률", ylabel="흔들림 (8개 확률의 표준편차)")
    ax.grid(alpha=.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "scatter.png")


if __name__ == "__main__":
    main()
