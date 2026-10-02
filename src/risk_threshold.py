"""보장 기준선: 판정 기준선을 '검증 사진 최약 불량의 점수' 한 점 대신, 통계적 보장과 점수 흔들림 여유를 둔 값으로 정한다.

배경: 박스 기준선(0.6437)이 검증 최약 실제 이물 점수와 같아, 같은 모델을 다시 돌리면 GPU 계산 차이로 재현율이 흔들렸다.

1. 점수 흔들림 ε: 검증 판정 사진(가짜 정상·실제 불량·합성 불량)을 묶음 크기 1 · 8 · 32 로 다시 추론해
   사진 최고 점수가 실행 조건에 따라 얼마나 달라지는지 잰다 (Pham et al. 2020: 구현 수준 요인만으로도 결과가 달라진다).
2. 합격선 (Learn then Test, Angelopoulos et al. 2021 의 고정 순서 검정):
   위험 R(t) = 사양 이상 불량이 점수 t 미만이라 합격되는 비율. t 를 낮은 쪽(안전)부터 올리며
   H0: R(t) > α 를 이항 꼬리 p값 P(Binom(n, α) ≤ k(t)) 로 검정하고, p ≤ δ 인 동안의 가장 큰 t 를 고른다.
   → "놓침률 α 이하"를 확률 1-δ 로 보장. 그 뒤 흔들림 ε 만큼 더 낮춘다 (t_final = t_LTT - ε).
3. 불합격선: 위험 = 가짜 정상이 점수 t 이상이라 불합격되는 비율. t 를 높은 쪽(안전)부터 내리며 같은 방식으로
   "정상 폐기율 β 이하"를 보장하는 가장 낮은 t 를 고르고, 흔들림 ε 만큼 더 올린다.
4. 표본 수로 보장 가능한 최소 α: 불량 n 개를 하나도 놓치지 않아도 (1-α)^n ≤ δ 여야 하므로 α ≥ 1 - δ^(1/n).
모두 val 로 정하고 test 는 마지막에 한 번 잰다.
채택(--alpha, --beta, 기본 1% · 5%): 놓침 1% 보장은 이 데이터(사양 안 불량 328개)로 보장할 수 있는 거의 최선이고(하한 0.91%),
불합격선은 실제 불량을 사람이 확인한 뒤 폐기하는 재검사 중심 운영(정상 자동 폐기 최소)을 택했다. predict.py 가 이 값을 기본으로 쓴다.

실행: .venv\\Scripts\\python.exe src\\risk_threshold.py --yolo ratio3_e100
결과: results/risk_threshold/ (summary.json, jitter_val.csv)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom
from tqdm import tqdm

from judge import in_spec, tiers
from train_yolo import weights_path

ROOT = Path(__file__).resolve().parents[1]
DELTA = 0.05
ALPHAS = [0.01, 0.02, 0.05]
BETAS = [0.05, 0.10]


def scores(model, paths, batch):
    out = []
    for k in tqdm(range(0, len(paths), batch), desc=f"묶음 {batch}", leave=False):
        for r in model.predict([str(p) for p in paths[k:k + batch]], imgsz=640, conf=0.001, max_det=100, verbose=False):
            c = r.boxes.conf.cpu().numpy()
            out.append(float(c.max()) if len(c) else 0.0)
    return np.array(out)


def ltt_low(s_def, alpha, delta=DELTA):
    """합격선: 불량 점수들 s_def. 낮은 t 부터 올리며 놓침률 > alpha 를 기각할 수 있는 가장 큰 t (= 다음 불량 점수 직전)."""
    s = np.sort(s_def)
    n = len(s)
    best = None
    for k in range(n):                       # t 를 k번째 불량 점수로 두면 그보다 낮은 k 개가 놓침
        if binom.cdf(k, n, alpha) <= delta:
            best = s[k]                      # t = s[k] 이면 s[k] 미만인 k 개만 합격 (s[k] 자신은 불합격)
        else:
            break
    return best


def ltt_high(s_norm, beta, delta=DELTA):
    """불합격선: 정상 점수들. 높은 t 부터 내리며 정상 폐기율 > beta 를 기각할 수 있는 가장 낮은 t."""
    s = np.sort(s_norm)[::-1]
    n = len(s)
    best = None
    for k in range(n):                       # t 를 k번째로 높은 정상 점수 바로 위로 두면 그보다 높은 k 개가 폐기
        if binom.cdf(k, n, beta) <= delta:
            best = s[k] + 1e-6
        else:
            break
    return best


def evaluate(df, col, t_low, t_high, spec_col):
    tt = tiers(df[col].to_numpy(), t_low, t_high)
    n, r, sn = df.kind == "normal", df.kind == "real_ng", df.kind == "synth_ng"
    ins = df[spec_col].to_numpy()
    return {"합격선": round(float(t_low), 4), "불합격선": round(float(t_high), 4),
            "정상_합격": round(float((tt[n] == "합격").mean()), 4), "정상_재검사": round(float((tt[n] == "재검사").mean()), 4),
            "정상_불합격": round(float((tt[n] == "불합격").mean()), 4),
            "실제불량_합격(놓침)": int((tt[r] == "합격").sum()), "실제불량_불합격": round(float((tt[r] == "불합격").mean()), 4),
            "사양안_합성_놓침": f"{int(((tt == '합격') & sn & ins).sum())}/{int((sn & ins).sum())}",
            "사양밖_합성_놓침": f"{int(((tt == '합격') & sn & ~ins).sum())}/{int((sn & ~ins).sum())}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo", default="ratio3_e100")
    ap.add_argument("--alpha", type=float, default=0.01, help="채택할 놓침률 보장 수준")
    ap.add_argument("--beta", type=float, default=0.05, help="채택할 정상 폐기율 보장 수준")
    args = ap.parse_args()
    from ultralytics import YOLO
    out = ROOT / "results" / "risk_threshold"
    out.mkdir(parents=True, exist_ok=True)
    m = args.yolo
    val = pd.read_csv(ROOT / "results/judge/scores_val.csv")
    test = pd.read_csv(ROOT / "results/judge/scores_test.csv")
    spec = pd.read_csv(ROOT / f"results/testpiece_val_{m}/spec.csv")
    spec = spec[spec["model"] == "YOLO"]
    for df in (val, test):
        df["사양내"] = in_spec(df, spec)

    # 1) 흔들림
    model = YOLO(str(weights_path(m)))
    paths = val["path"].tolist()
    jit = pd.DataFrame({"img": val["img"], "kind": val["kind"], "저장값(묶음32)": val[m]})
    for b in [1, 8, 32]:
        jit[f"묶음{b}"] = scores(model, paths, b)
    cols = [c for c in jit.columns if c.startswith("묶음") or c.startswith("저장값")]
    jit["흔들림"] = jit[cols].max(1) - jit[cols].min(1)
    jit.to_csv(out / "jitter_val.csv", index=False, encoding="utf-8-sig")
    eps = float(jit["흔들림"].max())
    near = jit[(jit[cols].min(1) > 0.4) & (jit[cols].max(1) < 0.9)]["흔들림"]      # 기준선이 놓일 만한 점수대
    summary = {"점수_흔들림": {"최대": round(eps, 4), "중앙": round(float(jit["흔들림"].median()), 5),
                           "점수0.4~0.9_최대": round(float(near.max()), 4) if len(near) else None,
                           "흔들림0초과_사진수": int((jit["흔들림"] > 0).sum()), "사진수": len(jit)}}
    print("흔들림", summary["점수_흔들림"])

    # 2) 3) 보장 기준선 (val)
    vd = val[(val.y == 1) & val["사양내"]][m].to_numpy()
    vn = val[val.kind == "normal"][m].to_numpy()
    summary["표본"] = {"사양안_불량_n": len(vd), "정상_n": len(vn),
                     "불량을_하나도_안놓쳐도_보장가능한_최소_놓침률": round(1 - DELTA ** (1 / len(vd)), 4),
                     "정상을_하나도_안버려도_보장가능한_최소_폐기율": round(1 - DELTA ** (1 / len(vn)), 4)}
    cur = json.load(open(ROOT / "results/judge/summary.json", encoding="utf-8"))[m]["사양기준"]["기준선"]
    rows = {"현재(사양 기준: 최약 불량 · 정상 99% 지점)": (cur["합격선"], cur["불합격선"])}
    for a in ALPHAS:
        tl = ltt_low(vd, a)
        for b in BETAS:
            th = ltt_high(vn, b)
            if tl is None or th is None:
                continue
            t_low, t_high = tl - eps, max(th + eps, tl - eps)
            rows[f"보장: 놓침 ≤{a:.0%}, 정상 폐기 ≤{b:.0%} (흔들림 여유 포함)"] = (t_low, t_high)
    tl, th = ltt_low(vd, args.alpha), ltt_high(vn, args.beta)
    summary["채택"] = {"모델": m, "합격선": round(float(tl - eps), 4), "불합격선": round(float(max(th + eps, tl - eps)), 4),
                     "보장": f"사양 이상 불량 놓침률 ≤ {args.alpha:.0%}, 정상 폐기율 ≤ {args.beta:.0%} (각각 확률 {1 - DELTA:.0%}, val)",
                     "흔들림_여유": round(eps, 4), "보장전_합격선": round(float(tl), 4), "보장전_불합격선": round(float(th), 4)}
    print("채택", summary["채택"])
    summary["후보"] = {}
    for name, (tl, th) in rows.items():
        summary["후보"][name] = {"val": evaluate(val, m, tl, th, "사양내"), "test": evaluate(test, m, tl, th, "사양내")}
        print(name, json.dumps(summary["후보"][name]["test"], ensure_ascii=False))
    json.dump(summary, open(out / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=float)


if __name__ == "__main__":
    main()
