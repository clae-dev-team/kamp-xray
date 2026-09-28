"""이미지 단위 합격/재검사/불합격 판정: 확률보정(Guo 2017 온도 스케일링) + 재현율 우선 3단 기준.

평가셋: normal_set.py 가 만든 가짜 정상(normal) · 실제 불량(real_ng) · 합성 불량(synth_ng).
영상 점수 = 영상 안 예측 중 최고 점수 (YOLO는 신뢰도, 베이스라인은 어두운 깊이).
val에서 기준을 정하고 test는 마지막에 한 번만 잰다.

  합격선  t_low : val 불량(실제+합성) 99%가 이 점수 이상 → 이 아래면 합격. (놓침 1% 이내 목표)
  불합격선 t_high: val 가짜 정상 점수의 NORMAL_Q 분위 → 이 이상이면 불합격. (정상을 버리는 비율 약 5% 이내)
    최고값을 쓰지 않는 이유: 가짜 정상에도 사람이 표시하지 않은 옅은 점(라벨 누락 의심)이 남아 있어
    최고값이 그 점에 끌려 올라간다. 불합격선을 넘은 가짜 정상은 suspects.csv 로 따로 내보내 눈으로 확인한다.
  그 사이 = 재검사. t_low 가 t_high 보다 높으면 재검사 구간 없이 한 기준선(t_low)만 쓴다.

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

ROOT = Path(__file__).resolve().parents[1]
RECALL_TARGET = 0.99
NORMAL_Q = 0.95
EPS = 1e-6


def yolo_scores(name, paths):
    from ultralytics import YOLO
    model = YOLO(str(ROOT / "runs" / name / "weights" / "best.pt"))
    out = []
    for k in tqdm(range(0, len(paths), 32), desc=name):
        for r in model.predict([str(p) for p in paths[k:k + 32]], imgsz=640, conf=0.001, max_det=100, verbose=False):
            c = r.boxes.conf.cpu().numpy()
            out.append(float(c.max()) if len(c) else 0.0)
    return np.array(out)


def baseline_scores(prm, paths, machines):
    boxes = {int(k): v for k, v in prm["box_by_machine"].items()}
    out = []
    for p, m in tqdm(list(zip(paths, machines)), desc="베이스라인"):
        d = B.detect(np.asarray(Image.open(p)), prm["se"], prm["sigma"], prm["score"], boxes[int(m)])
        out.append(float(d["score"].max()) if len(d) else 0.0)
    return np.array(out)


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_calibration(s, y, kind):
    """kind='temp': sigmoid(logit(s)/T), kind='platt': sigmoid(a·s+b). 반환: 보정 함수, 파라미터."""
    if kind == "temp":
        z = logit(s)
        nll = lambda t: -np.mean(y * np.log(1 / (1 + np.exp(-z / t[0])) + EPS) +
                                 (1 - y) * np.log(1 - 1 / (1 + np.exp(-z / t[0])) + EPS))
        t = minimize(nll, [1.0], bounds=[(0.05, 20)]).x[0]
        return (lambda q: 1 / (1 + np.exp(-logit(q) / t))), {"T": round(float(t), 4)}
    mu, sd = s.mean(), s.std() + EPS
    zs = (s - mu) / sd
    nll = lambda ab: -np.mean(y * np.log(1 / (1 + np.exp(-(ab[0] * zs + ab[1]))) + EPS) +
                              (1 - y) * np.log(1 - 1 / (1 + np.exp(-(ab[0] * zs + ab[1]))) + EPS))
    a, b = minimize(nll, [1.0, 0.0]).x
    return (lambda q: 1 / (1 + np.exp(-(a * (q - mu) / sd + b)))), {"a": round(float(a), 4), "b": round(float(b), 4)}


def ece(p, y, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    e = 0.0
    for k in range(bins):
        m = idx == k
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def tiers(s, t_low, t_high):
    return np.where(s >= t_high, "불합격", np.where(s >= t_low, "재검사", "합격"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo", nargs="+", default=["y26s_640", "y26s_640_aug"])
    ap.add_argument("--reuse", action="store_true", help="저장된 점수(scores_*.csv)를 다시 씀")
    args = ap.parse_args()
    data = ROOT / "data"
    out = ROOT / "results" / "judge"
    out.mkdir(parents=True, exist_ok=True)
    sets = {s: pd.read_csv(data / f"judge_{s}.csv") for s in ["val", "test"]}
    bl = json.load(open(ROOT / "results/baseline_clean/metrics.json", encoding="utf-8"))

    models = ["베이스라인"] + args.yolo
    for s, df in list(sets.items()):
        if args.reuse and (out / f"scores_{s}.csv").exists():
            sets[s] = pd.read_csv(out / f"scores_{s}.csv")
            continue
        df["y"] = (df["kind"] != "normal").astype(int)
        df["베이스라인"] = baseline_scores(bl["params"], df["path"].tolist(), df["machine"].tolist())
        for m in args.yolo:
            df[m] = yolo_scores(m, df["path"].tolist())
        df.to_csv(out / f"scores_{s}.csv", index=False, encoding="utf-8-sig")

    val, test = sets["val"], sets["test"]
    summary, rel = {}, {}
    for m in models:
        vs, vy = val[m].to_numpy(), val["y"].to_numpy()
        # 기준선 (val)
        t_low = float(np.quantile(vs[vy == 1], 1 - RECALL_TARGET, method="lower"))
        t_high = float(np.quantile(vs[vy == 0], NORMAL_Q))
        if t_low >= t_high:           # 불량과 정상이 완전히 갈리면 한 기준선
            t_high = t_low
        # 보정 (val 에서 맞추고 test 에서 잼)
        cal, prm = fit_calibration(vs, vy, "platt" if m == "베이스라인" else "temp")
        ts, ty = test[m].to_numpy(), test["y"].to_numpy()
        raw_p = ts / ts.max() if m == "베이스라인" else ts
        res = {"기준선": {"합격선": round(t_low, 4), "불합격선": round(t_high, 4)}, "보정": prm,
               "ECE_보정전": round(ece(np.clip(raw_p, 0, 1), ty), 4), "ECE_보정후": round(ece(cal(ts), ty), 4)}
        # AUROC
        for kind in ["real_ng", "synth_ng"]:
            sel = test["kind"].isin([kind, "normal"])
            res[f"AUROC_{kind}"] = round(float(roc_auc_score(test.loc[sel, "y"], test.loc[sel, m])), 4)
        # 3단 판정 비율 (test)
        test[f"{m}_판정"] = tiers(ts, t_low, t_high)
        tab = test.groupby("kind")[f"{m}_판정"].value_counts(normalize=True).unstack(fill_value=0)
        res["판정비율"] = {k: {c: round(float(v), 4) for c, v in row.items()} for k, row in tab.iterrows()}
        # 합성 불량 중 합격(=놓침)이 몰린 조건
        sn = test[test["kind"] == "synth_ng"]
        miss = sn[sn[f"{m}_판정"] == "합격"]
        res["합성불량_놓침"] = {"수": int(len(miss)), "대비중앙": round(float(miss["c_meas"].median()), 3) if len(miss) else None,
                           "지름중앙": round(float(miss["d"].median()), 2) if len(miss) else None,
                           "파편비율": round(float(miss["shard"].mean()), 3) if len(miss) else None}
        summary[m] = res
        rel[m] = (cal(ts), ty, raw_p)
        print(m, json.dumps(res, ensure_ascii=False))

    test.to_csv(out / "judged_test.csv", index=False, encoding="utf-8-sig")
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
    for ax, m in zip(np.atleast_1d(axes), models):
        p, y, raw = rel[m]
        for q, lab, c in [(np.clip(raw, 0, 1), "보정 전", "#9aa5b1"), (p, "보정 후", "#0a8f86")]:
            idx = np.minimum((q * 10).astype(int), 9)
            xs = [q[idx == k].mean() for k in range(10) if (idx == k).any()]
            ys = [y[idx == k].mean() for k in range(10) if (idx == k).any()]
            ax.plot(xs, ys, "o-", color=c, ms=4, lw=1.6, label=lab)
        ax.plot([0, 1], [0, 1], ":", color="#6a7480", lw=1)
        ax.set(title=m, xlabel="모델이 말한 불량 확률", xlim=(0, 1), ylim=(0, 1))
        ax.grid(alpha=.3)
    np.atleast_1d(axes)[0].set_ylabel("실제 불량 비율")
    np.atleast_1d(axes)[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "reliability.png")


if __name__ == "__main__":
    main()
