"""합성 저대비 이물 평가: 베이스라인과 YOLO가 이물 대비·크기에 따라 어디서부터 놓치는가.

합성 이물 하나마다 '판정 임계값 이상 예측의 중심이 이물 박스(±2px) 안에 있으면 검출'로 본다.
임계값은 각 모델이 val에서 정한 F1 최대 값을 그대로 쓴다 (합성셋으로 다시 맞추지 않는다).
오검출 = 합성·실제 이물 어느 쪽에도 닿지 않은 판정 임계값 이상 예측.

실행: .venv\\Scripts\\python.exe src\\synth_eval.py --yolo y26s_640
결과: results/synth_eval/ (이물별 점수표, 조건별 검출률표, 히트맵, 대비 곡선)
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import baseline as B

ROOT = Path(__file__).resolve().parents[1]
MARGIN = 2


def yolo_preds(name, paths, ids, imgsz=640, batch=32):
    from ultralytics import YOLO
    model = YOLO(str(ROOT / "runs" / name / "weights" / "best.pt"))
    rows = []
    for k in tqdm(range(0, len(paths), batch), desc=f"YOLO {name}"):
        res = model.predict([str(p) for p in paths[k:k + batch]], imgsz=imgsz, conf=0.001,
                            max_det=100, verbose=False)
        for i, r in zip(ids[k:k + batch], res):
            for (x0, y0, x1, y1), s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                rows.append((i, (x0 + x1) / 2, (y0 + y1) / 2, float(s)))
    return pd.DataFrame(rows, columns=["img", "px", "py", "score"])


def baseline_preds(prm, paths, ids, machines):
    boxes = {int(k): v for k, v in prm["box_by_machine"].items()}
    rows = []
    for p, i, m in tqdm(list(zip(paths, ids, machines)), desc="베이스라인"):
        d = B.detect(np.asarray(Image.open(p)), prm["se"], prm["sigma"], prm["score"], boxes[int(m)])
        for r in d.itertuples():
            rows.append((i, (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2, float(r.score)))
    return pd.DataFrame(rows, columns=["img", "px", "py", "score"])


def score_defects(pred, defects, real_boxes, thr, half):
    """이물별 최고 점수·검출 여부, 영상별 오검출 수."""
    best = np.zeros(len(defects))
    fp = {}
    by_img = dict(tuple(pred.groupby("img")))
    for img, grp in defects.groupby("img"):
        q = by_img.get(img)
        if q is None:
            fp[img] = 0
            continue
        used = np.zeros(len(q), bool)
        for idx, r in grp.iterrows():
            near = (np.abs(q["px"] - r.cx) <= half + MARGIN) & (np.abs(q["py"] - r.cy) <= half + MARGIN)
            if near.any():
                best[defects.index.get_loc(idx)] = q.loc[near, "score"].max()
                used |= near.to_numpy()
        for x0, y0, x1, y1 in real_boxes[grp["src"].iloc[0]]:
            used |= ((q["px"] >= x0 - MARGIN) & (q["px"] <= x1 + MARGIN) &
                     (q["py"] >= y0 - MARGIN) & (q["py"] <= y1 + MARGIN)).to_numpy()
        fp[img] = int(((q["score"] >= thr).to_numpy() & ~used).sum())
    return best, fp


def heatmap(ax, tab, title):
    im = ax.imshow(tab.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto", origin="lower")
    ax.set_xticks(range(tab.shape[1]), [f"{c:g}" for c in tab.columns])
    ax.set_yticks(range(tab.shape[0]), [f"{c:g}" for c in tab.index])
    for (i, j), v in np.ndenumerate(tab.to_numpy()):
        ax.text(j, i, f"{v * 100:.0f}", ha="center", va="center", fontsize=7.5,
                color="white" if v > 0.6 else "#1d2733")
    ax.set(xlabel="이물 지름 (px)", ylabel="명목 대비 c0", title=title)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "data.yaml"))
    ap.add_argument("--yolo", default="y26s_640")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    data = ROOT / cfg["out_dir"]
    syn = data / "synth"
    out = ROOT / "results" / "synth_eval"
    out.mkdir(parents=True, exist_ok=True)
    scfg = json.load(open(syn / "config.json", encoding="utf-8"))
    half = scfg["box"] / 2

    imgs = pd.read_csv(syn / "images.csv")
    defects = pd.read_csv(syn / "defects.csv")
    man = pd.read_csv(data / "manifest.csv").set_index("id")
    real_boxes = {}
    for s in imgs["src"].unique():
        b = np.loadtxt(data / "clean/labels" / f"{s}.txt", ndmin=2)
        w, h = man.loc[s, "w"], man.loc[s, "h"]
        real_boxes[s] = np.stack([(b[:, 1] - b[:, 3] / 2) * w, (b[:, 2] - b[:, 4] / 2) * h,
                                  (b[:, 1] + b[:, 3] / 2) * w, (b[:, 2] + b[:, 4] / 2) * h], 1)
    paths = [syn / "images" / f"{i}.png" for i in imgs["img"]]

    bl = json.load(open(ROOT / "results/baseline_clean/metrics.json", encoding="utf-8"))
    yo = json.load(open(ROOT / f"results/yolo_{args.yolo}/metrics.json", encoding="utf-8"))
    models = {
        "베이스라인": (baseline_preds(bl["params"], paths, imgs["img"].tolist(), imgs["machine"].tolist()),
                    bl["thresholds"]["F1최대"]),
        "YOLO": (yolo_preds(args.yolo, paths, imgs["img"].tolist()), yo["thresholds"]["F1최대"]),
    }

    summary = {}
    for name, (pred, thr) in models.items():
        best, fp = score_defects(pred, defects, real_boxes, thr, half)
        defects[f"{name}_score"] = best
        defects[f"{name}_hit"] = best >= thr
        summary[name] = {"thr": thr, "검출률_전체": round(float((best >= thr).mean()), 4),
                         "오검출_영상당": round(float(np.mean(list(fp.values()))), 4),
                         "오검출_합계": int(sum(fp.values()))}
    defects.to_csv(out / "defects_scored.csv", index=False, encoding="utf-8-sig")

    tabs = {n: defects.pivot_table(index="c0", columns="d", values=f"{n}_hit", aggfunc="mean") for n in models}
    for n, t in tabs.items():
        t.round(3).to_csv(out / f"rate_{n}.csv", encoding="utf-8-sig")

    # 측정 대비 구간별 검출률 (실제 이물과 같은 방식으로 잰 대비)
    bins = [0, 0.04, 0.06, 0.08, 0.10, 0.13, 0.16, 0.20, 0.25, 0.30, 0.40, 0.70]
    defects["c_bin"] = pd.cut(defects["c_meas"], bins)
    curve = defects.groupby("c_bin", observed=True)[[f"{n}_hit" for n in models]].mean()
    curve["n"] = defects.groupby("c_bin", observed=True).size()
    curve.round(3).to_csv(out / "rate_by_measured_contrast.csv", encoding="utf-8-sig")
    for n in models:
        summary[n]["구간별_검출률"] = {str(k): round(float(v), 3) for k, v in curve[f"{n}_hit"].items()}
    real = pd.read_csv(ROOT / "results/defect_stats/real_defects.csv")
    summary["실제이물_측정대비"] = {k: round(float(real["contrast"].quantile(q)), 3)
                            for k, q in [("p05", .05), ("p50", .5), ("p95", .95)]}
    json.dump(summary, open(out / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    plt.rcParams["font.family"] = "Malgun Gothic"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), dpi=150)
    for ax, (n, t) in zip(axes, tabs.items()):
        im = heatmap(ax, t, f"{n} 검출률(%)")
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.savefig(out / "heatmap.png", bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    mid = [(b.left + b.right) / 2 for b in curve.index]
    ax.axvspan(real["contrast"].quantile(.05), real["contrast"].quantile(.95), color="#f2c14e", alpha=.25,
               label="실제 이물 대비 (5~95%)")
    for n, c in [("베이스라인", "#9aa5b1"), ("YOLO", "#1f5fa8")]:
        ax.plot(mid, curve[f"{n}_hit"], "o-", color=c, lw=2, ms=4, label=n)
    ax.set(xlabel="측정 대비 (주변 대비 어두운 비율)", ylabel="검출률", ylim=(0, 1.03), xlim=(0, 0.6))
    ax.grid(alpha=.3)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "rate_by_contrast.png")

    print(json.dumps({n: {k: v for k, v in s.items() if k != "구간별_검출률"} for n, s in summary.items()},
                     ensure_ascii=False))
    print(curve.round(3).to_string())
    for n, t in tabs.items():
        print(n)
        print((t * 100).round(0).to_string())


if __name__ == "__main__":
    main()
