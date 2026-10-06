"""이미지 단위 합격/재검사/불합격 판정: 확률보정(Guo 2017 온도 스케일링) + 재현율 우선 3단 기준.

평가셋: normal_set.py 가 만든 가짜 정상(normal) · 실제 불량(real_ng) · 합성 불량(synth_ng).
영상 점수 = 영상 안 예측 중 최고 점수 (YOLO는 신뢰도, 베이스라인은 어두운 깊이).
val에서 기준을 정하고 test는 마지막에 한 번만 잰다.

  합격선  t_low : val 불량(실제+합성) 99%가 이 점수 이상 → 이 아래면 합격. (놓침 1% 이내 목표)
  불합격선 t_high: val 가짜 정상 점수의 NORMAL_Q 분위 → 이 이상이면 불합격. (정상을 버리는 비율 약 5% 이내)
    최고값을 쓰지 않는 이유: 가짜 정상에도 사람이 표시하지 않은 옅은 점(라벨 누락 의심)이 남아 있어
    최고값이 그 점에 끌려 올라간다. 불합격선을 넘은 가짜 정상은 suspects.csv 로 따로 내보내 눈으로 확인한다.
  그 사이 = 재검사. t_low 가 t_high 보다 높으면 재검사 구간 없이 한 기준선(t_low)만 쓴다.

사양 기준(권장, results/testpiece_val_<모델>/spec.csv 가 있을 때 함께 계산):
  '모든 불량 99%'는 이 장비·AI로 보장할 수 없는 아주 옅은 합성 이물까지 잡으려다 합격선이 크게 내려가
  정상 제품 약 20%가 재검사로 밀린다. 그래서 안전 목표를 '검출 사양(val 테스트피스, 90% 보장 진하기) 이상 이물은
  합격시키지 않는다'로 정한다.
  합격선  = val 의 실제 불량 + 사양 이상 합성 불량 중 최저 점수
  불합격선 = max(합격선, val 가짜 정상 점수 99% 분위)   (정상이 불합격까지 가는 일은 약 1%)
  사양보다 옅은 이물은 보장하지 않으며, 그 놓침 수를 따로 보고한다.

확률보정: YOLO 신뢰도 p 를 logit(p)/T 로 다시 시그모이드. T 는 val NLL 최소화.
베이스라인 점수는 확률이 아니라서 Platt(a·s+b) 로 맞춘다. 전후 ECE(10구간)를 test에서 비교.

실행: .venv\\Scripts\\python.exe src\\judge.py --yolo y26s_640 y26s_640_aug
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import baseline as B
import cnn as C
from train_yolo import weights_path

ROOT = Path(__file__).resolve().parents[1]   # 저장소 최상위 폴더
RECALL_TARGET = 0.99   # 합격선: val 불량 중 이 비율이 합격선 이상이 되게 한다 (놓침 1% 이내 목표)
NORMAL_Q = 0.95        # 불합격선: val 가짜 정상 점수의 이 분위 (정상을 버리는 비율 약 5% 이내)
EPS = 1e-6             # 로그 · 나눗셈에서 0 이 나오지 않게 더하는 작은 값


def yolo_scores(name, paths, machines=None):
    """영상 점수(영상 안 예측 중 최고 점수)를 구한다. 이름이 CNN 이면 CNN 으로, 아니면 YOLO 로 예측한다.

    name: 모델 이름. paths: 영상 경로 목록. machines: 영상별 호기(CNN 의 박스 크기에만 쓴다).
    반환: (영상 수,) 배열, 0~1. 예측이 하나도 없는 영상은 0.0 이다.
    """
    if C.is_cnn(name):   # 조각 분류 CNN: 영상 안 봉우리 최고 확률
        # id 대신 목록 안 순번(0, 1, 2, ...)을 넘겨, 결과를 입력 순서대로 다시 늘어놓는다
        d = C.predict_paths(name, paths, list(range(len(paths))), machines)
        return d.groupby("id")["score"].max().reindex(range(len(paths)), fill_value=0.0).to_numpy()
    from ultralytics import YOLO
    model = YOLO(str(weights_path(name)))
    out = []
    # 32장씩 묶어 추론한다. conf=0.001: 낮은 점수 박스까지 받아 최고 점수가 빠지지 않게 한다
    for k in tqdm(range(0, len(paths), 32), desc=name):
        for r in model.predict([str(p) for p in paths[k:k + 32]], imgsz=640, conf=0.001, max_det=100, verbose=False):
            c = r.boxes.conf.cpu().numpy()
            out.append(float(c.max()) if len(c) else 0.0)
    return np.array(out)


def baseline_scores(prm, paths, machines):
    """베이스라인(top-hat)의 영상 점수 = 후보 중 최고 점수.

    prm: results/baseline_clean/metrics.json 의 params (se, sigma, score, box_by_machine).
    반환: (영상 수,) 배열. 확률이 아니라 베이스라인 점수 그대로이고, 후보가 없으면 0.0 이다.
    """
    boxes = {int(k): v for k, v in prm["box_by_machine"].items()}   # json 에서는 키가 문자열이라 정수로 되돌린다
    out = []
    for p, m in tqdm(list(zip(paths, machines)), desc="베이스라인"):
        d = B.detect(np.asarray(Image.open(p)), prm["se"], prm["sigma"], prm["score"], boxes[int(m)])
        out.append(float(d["score"].max()) if len(d) else 0.0)
    return np.array(out)


def logit(p):
    """확률 → 로짓 log(p / (1-p)). 0 과 1 에서 무한대가 되지 않게 [EPS, 1-EPS] 로 자른 뒤 계산한다."""
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_calibration(s, y, kind):
    """kind='temp': sigmoid(logit(s)/T), kind='platt': sigmoid(a·s+b). 반환: 보정 함수, 파라미터.

    s: (N,) 영상 점수, y: (N,) 정답(불량 1 · 정상 0). 음의 로그 가능도(NLL)가 최소인 값을 찾는다.
    보정 함수는 점수 배열을 받아 같은 모양의 불량 확률(0~1)을 돌려준다.
    """
    if kind == "temp":
        # 온도 스케일링: 로짓을 T 하나로 나눈다. 점수의 순서는 바뀌지 않는다
        z = logit(s)
        nll = lambda t: -np.mean(y * np.log(1 / (1 + np.exp(-z / t[0])) + EPS) +
                                 (1 - y) * np.log(1 - 1 / (1 + np.exp(-z / t[0])) + EPS))
        t = minimize(nll, [1.0], bounds=[(0.05, 20)]).x[0]   # T = 1(보정 없음)에서 시작해 0.05~20 안에서 찾는다
        return (lambda q: 1 / (1 + np.exp(-logit(q) / t))), {"T": round(float(t), 4)}
    # Platt: 점수를 val 평균 · 표준편차로 표준화한 값에 a, b 를 맞춘다. 반환하는 a, b 도 표준화한 점수 기준이다
    mu, sd = s.mean(), s.std() + EPS
    zs = (s - mu) / sd
    nll = lambda ab: -np.mean(y * np.log(1 / (1 + np.exp(-(ab[0] * zs + ab[1]))) + EPS) +
                              (1 - y) * np.log(1 - 1 / (1 + np.exp(-(ab[0] * zs + ab[1]))) + EPS))
    a, b = minimize(nll, [1.0, 0.0]).x
    return (lambda q: 1 / (1 + np.exp(-(a * (q - mu) / sd + b)))), {"a": round(float(a), 4), "b": round(float(b), 4)}


def ece(p, y, bins=10):
    """기대 보정 오차(ECE). p: (N,) 예측 확률 0~1, y: (N,) 정답 0/1. 반환: 0~1, 작을수록 확률이 실제 비율과 맞다.

    확률을 같은 폭의 bins 구간으로 나눠, 구간마다 |평균 확률 - 실제 불량 비율| 을 표본 비율로 가중해 더한다.
    """
    idx = np.minimum((p * bins).astype(int), bins - 1)   # p = 1.0 은 마지막 구간에 넣는다
    e = 0.0
    for k in range(bins):
        m = idx == k
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def in_spec(df, spec):
    """합성 불량이 검출 사양(그 호기·그 지름 이하 사양 지름의 90% 보장 진하기) 이상인가. 실제 불량은 모두 사양 안으로 본다.

    df: 판정 세트 표(열 kind, machine, d = 지름, c0 = 넣은 진하기).
    spec: 사양 표(열 machine, d, min_c0 = 그 지름에서 보장되는 최소 진하기).
    반환: (행 수,) bool. 가짜 정상은 False 다.
    """
    ds = np.array(sorted(spec["d"].unique()))
    out = []
    for r in df.itertuples():
        if r.kind == "real_ng":
            out.append(True)
        elif r.kind != "synth_ng" or not (ds <= r.d).any():   # 정상이거나, 사양표의 가장 작은 지름보다 작은 이물
            out.append(False)
        else:
            # 이물 지름 이하인 사양 지름 중 가장 큰 것의 기준을 쓴다. 기준이 비어 있으면(NaN) 사양 밖으로 본다
            c = spec[(spec["machine"] == r.machine) & (spec["d"] == ds[ds <= r.d].max())]["min_c0"].iloc[0]
            out.append(bool(pd.notna(c) and r.c0 >= c))
    return np.array(out)


def tiers(s, t_low, t_high):
    """영상 점수를 3단 판정으로 바꾼다. 불합격선 이상 = 불합격, 합격선 이상 = 재검사, 그 아래 = 합격.

    s: (N,) 점수. 반환: (N,) 문자열 배열("합격" / "재검사" / "불합격"). 두 기준선 모두 '이상'이 걸리는 쪽이다.
    """
    return np.where(s >= t_high, "불합격", np.where(s >= t_low, "재검사", "합격"))


def main():
    """판정 세트(val · test) 점수 계산 → val 에서 기준선 · 확률보정 → test 판정 · 요약 저장.

    results/judge/ 에 scores_<분할>.csv, judged_test.csv, suspects.csv, summary.json, reliability.png 를 남긴다.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo", nargs="+", default=["y26s_640", "y26s_640_aug"], help="YOLO 또는 CNN(cnn.py) 모델 이름")
    ap.add_argument("--reuse", action="store_true", help="저장된 점수(scores_*.csv)를 다시 씀")
    args = ap.parse_args()
    data = ROOT / "data"
    out = ROOT / "results" / "judge"
    out.mkdir(parents=True, exist_ok=True)
    # 판정 세트 목록(normal_set.py 가 만든 것). 한 행이 영상 한 장이고 kind 는 normal / real_ng / synth_ng 이다
    sets = {s: pd.read_csv(data / f"judge_{s}.csv") for s in ["val", "test"]}
    bl = json.load(open(ROOT / "results/baseline_clean/metrics.json", encoding="utf-8"))

    # 1) 영상 점수 계산. 모델 이름이 그대로 점수 열 이름이 된다
    models = ["베이스라인"] + args.yolo
    for s, df in list(sets.items()):
        if args.reuse and (out / f"scores_{s}.csv").exists():
            sets[s] = pd.read_csv(out / f"scores_{s}.csv")
            continue
        df["y"] = (df["kind"] != "normal").astype(int)   # 불량(실제 · 합성) 1, 가짜 정상 0
        df["베이스라인"] = baseline_scores(bl["params"], df["path"].tolist(), df["machine"].tolist())
        for m in args.yolo:
            df[m] = yolo_scores(m, df["path"].tolist(), df["machine"].tolist())
        df.to_csv(out / f"scores_{s}.csv", index=False, encoding="utf-8-sig")

    # 2) 모델별 기준선 · 확률보정 · 판정 ('모든 불량 99%' 기준)
    val, test = sets["val"], sets["test"]
    summary, rel = {}, {}
    for m in models:
        vs, vy = val[m].to_numpy(), val["y"].to_numpy()
        # 기준선 (val)
        # 합격선 = 불량 점수의 하위 1% 분위. method="lower" 는 보간하지 않고 실제 점수 중 아래쪽 값을 고른다
        t_low = float(np.quantile(vs[vy == 1], 1 - RECALL_TARGET, method="lower"))
        t_high = float(np.quantile(vs[vy == 0], NORMAL_Q))
        if t_low >= t_high:           # 불량과 정상이 완전히 갈리면 한 기준선
            t_high = t_low
        # 보정 (val 에서 맞추고 test 에서 잼)
        cal, prm = fit_calibration(vs, vy, "platt" if m == "베이스라인" else "temp")
        ts, ty = test[m].to_numpy(), test["y"].to_numpy()
        # 보정 전 ECE 용 값. 베이스라인 점수는 0~1 이 아니라서 test 최고 점수로 나눠 0~1 로 맞춘다
        raw_p = ts / ts.max() if m == "베이스라인" else ts
        res = {"기준선": {"합격선": round(t_low, 4), "불합격선": round(t_high, 4)}, "보정": prm,
               "ECE_보정전": round(ece(np.clip(raw_p, 0, 1), ty), 4), "ECE_보정후": round(ece(cal(ts), ty), 4)}
        # AUROC
        # 불량 종류별로 따로 잰다: (실제 불량 대 가짜 정상), (합성 불량 대 가짜 정상)
        for kind in ["real_ng", "synth_ng"]:
            sel = test["kind"].isin([kind, "normal"])
            res[f"AUROC_{kind}"] = round(float(roc_auc_score(test.loc[sel, "y"], test.loc[sel, m])), 4)
        # 3단 판정 비율 (test)
        test[f"{m}_판정"] = tiers(ts, t_low, t_high)
        # 행 = 영상 종류, 열 = 판정, 값 = 그 종류 안에서의 비율
        tab = test.groupby("kind")[f"{m}_판정"].value_counts(normalize=True).unstack(fill_value=0)
        res["판정비율"] = {k: {c: round(float(v), 4) for c, v in row.items()} for k, row in tab.iterrows()}
        # 합성 불량 중 합격(=놓침)이 몰린 조건
        sn = test[test["kind"] == "synth_ng"]
        miss = sn[sn[f"{m}_판정"] == "합격"]
        res["합성불량_놓침"] = {"수": int(len(miss)), "대비중앙": round(float(miss["c_meas"].median()), 3) if len(miss) else None,
                           "지름중앙": round(float(miss["d"].median()), 2) if len(miss) else None,
                           "파편비율": round(float(miss["shard"].mean()), 3) if len(miss) else None}
        summary[m] = res
        rel[m] = (cal(ts), ty, raw_p)   # 신뢰도 그래프용: (보정 후 확률, 정답, 보정 전 값)
        print(m, json.dumps(res, ensure_ascii=False))

    # 사양 기준 판정
    # 3) 검출 사양 파일이 있는 모델만 계산한다 (testpiece.py --split val 을 먼저 돌려야 생긴다)
    for m in models:
        sf = ROOT / "results" / f"testpiece_val_{m}" / "spec.csv"
        if not sf.exists():
            continue
        spec = pd.read_csv(sf)
        spec = spec[spec["model"] == "YOLO"]   # 사양 파일의 model 열이 "YOLO" 인 행만 쓴다
        for df in (val, test):
            df[f"{m}_사양내"] = in_spec(df, spec)
        vs, vy = val[m].to_numpy(), val["y"].to_numpy()
        # 합격선 = val 의 사양 안 불량(실제 + 사양 이상 합성) 중 최저 점수
        t_low = float(val.loc[(vy == 1) & val[f"{m}_사양내"], m].min())
        # 불합격선 = val 가짜 정상 점수의 99% 분위. 합격선보다 낮으면 합격선에 맞춘다
        t_high = max(t_low, float(np.quantile(vs[vy == 0], 0.99)))
        ts = test[m].to_numpy()
        test[f"{m}_판정_사양"] = tiers(ts, t_low, t_high)
        tab = test.groupby("kind")[f"{m}_판정_사양"].value_counts(normalize=True).unstack(fill_value=0)
        sn = test[test["kind"] == "synth_ng"]
        miss = sn[sn[f"{m}_판정_사양"] == "합격"]
        # 놓침 항목은 "놓친 수/대상 수" 문자열로 적는다
        summary[m]["사양기준"] = {
            "기준선": {"합격선": round(t_low, 4), "불합격선": round(t_high, 4)}, "사양파일": str(sf.relative_to(ROOT)),
            "판정비율": {k: {c: round(float(v), 4) for c, v in row.items()} for k, row in tab.iterrows()},
            "합성불량_사양내_놓침": f"{int(miss[f'{m}_사양내'].sum())}/{int(sn[f'{m}_사양내'].sum())}",
            "합성불량_사양밖_놓침": f"{int((~miss[f'{m}_사양내']).sum())}/{int((~sn[f'{m}_사양내']).sum())}",
            "실제불량_놓침": int((test.loc[test['kind'] == 'real_ng', f"{m}_판정_사양"] == "합격").sum()),
            "실제불량_최저점수": round(float(test.loc[test["kind"] == "real_ng", m].min()), 4)}
        print(m, "사양기준", json.dumps(summary[m]["사양기준"], ensure_ascii=False))

    # 4) 저장. judged_test.csv 에는 점수 · 판정 · 사양내 열이 모두 들어간다
    test.to_csv(out / "judged_test.csv", index=False, encoding="utf-8-sig")
    # 라벨 누락 의심: '모든 불량 99%' 기준의 불합격선을 넘은 가짜 정상 (val · test 모두, 베이스라인 제외)
    sus = []
    for s, df in sets.items():
        for m in args.yolo:
            th = summary[m]["기준선"]["불합격선"]
            q = df[(df["kind"] == "normal") & (df[m] >= th)]
            sus += [dict(split=s, model=m, img=r.img, score=round(r[m], 3), path=r.path) for _, r in q.iterrows()]
    pd.DataFrame(sus).to_csv(out / "suspects.csv", index=False, encoding="utf-8-sig")
    json.dump(summary, open(out / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # 신뢰도 그래프 (test): 대각선에 가까울수록 확률을 믿을 수 있다
    plt.rcParams["font.family"] = "Malgun Gothic"
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 3.8), dpi=140, sharey=True)
    for ax, m in zip(np.atleast_1d(axes), models):   # 모델이 하나뿐이어도 돌 수 있게 배열로 만든다
        p, y, raw = rel[m]
        for q, lab, c in [(np.clip(raw, 0, 1), "보정 전", "#9aa5b1"), (p, "보정 후", "#0a8f86")]:
            # ece 와 같은 10구간. 구간마다 (평균 확률, 실제 불량 비율) 한 점이고 빈 구간은 건너뛴다
            idx = np.minimum((q * 10).astype(int), 9)
            xs = [q[idx == k].mean() for k in range(10) if (idx == k).any()]
            ys = [y[idx == k].mean() for k in range(10) if (idx == k).any()]
            ax.plot(xs, ys, "o-", color=c, ms=4, lw=1.6, label=lab)
        ax.plot([0, 1], [0, 1], ":", color="#6a7480", lw=1)   # 완전히 보정된 경우의 대각선
        ax.set(title=m, xlabel="모델이 말한 불량 확률", xlim=(0, 1), ylim=(0, 1))
        ax.grid(alpha=.3)
    np.atleast_1d(axes)[0].set_ylabel("실제 불량 비율")
    np.atleast_1d(axes)[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "reliability.png")


if __name__ == "__main__":
    main()
