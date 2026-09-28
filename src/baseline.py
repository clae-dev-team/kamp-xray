"""고전 영상처리 베이스라인: 학습 없이 '주변보다 어두운 작은 점'을 찾는다.

방법
  1. 가우시안으로 잡음을 살짝 누른다 (sigma)
  2. black top-hat = closing(영상) - 영상. 구조요소(se)보다 작은 어두운 점만 밝게 남는다.
     이물(약 2~4px 점)은 남고, 그보다 넓은 띠·제품 윤곽은 사라진다.
  3. 검출 점수 (둘 중 train에서 나은 쪽)
     depth : top-hat 값 그대로 = 주변보다 몇 회색 단계 어두운가
     z     : 영상마다 top-hat 분포로 표준화 (호기별 잡음 차이 보정)
  4. 제품 영역 안의 국소 최댓값을 후보로, 학습셋 정답의 호기별 중앙 크기 박스를 씌운다

하이퍼파라미터(se, sigma, score)는 train의 AP로, 판정 임계값은 val로 고르고 test는 마지막에 한 번만 잰다.
실행: .venv\\Scripts\\python.exe src\\baseline.py
"""
import argparse
import itertools
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import metrics as M

ROOT = Path(__file__).resolve().parents[1]
GRID = {"se": [3, 5, 7, 9], "sigma": [0.0, 0.7, 1.2], "score": ["depth", "z"]}
CAND_MIN = {"depth": 3.0, "z": 2.0}   # 후보로 남길 최소 점수 (PR 곡선을 그릴 만큼 넉넉히)
RECALL_TARGET = 0.95  # 재현율 우선 판정용 목표


def product_mask(gray):
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    _, m = cv2.threshold(blur, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return cv2.erode(m, np.ones((3, 3), np.uint8))


def detect(gray, se, sigma, score, box_wh):
    f = gray.astype(np.float32)
    if sigma > 0:
        f = cv2.GaussianBlur(f, (0, 0), sigma)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se, se))
    th = cv2.morphologyEx(f, cv2.MORPH_BLACKHAT, k)
    pm = product_mask(gray) > 0
    # 표준화: 값이 정수라 MAD가 0으로 무너지므로, 상위 0.5%(이물 후보)를 뺀 표준편차를 쓴다
    v = th[pm]
    v = v[v <= np.quantile(v, 0.995)]
    z = (th - v.mean()) / (v.std() + 1e-3) if score == "z" else th
    peak = (z == cv2.dilate(z, np.ones((7, 7), np.uint8))) & pm & (z >= CAND_MIN[score])
    ys, xs = np.nonzero(peak)
    bw, bh = box_wh
    return pd.DataFrame({"x0": xs - bw / 2 + 0.5, "y0": ys - bh / 2 + 0.5,
                         "x1": xs + bw / 2 + 0.5, "y1": ys + bh / 2 + 0.5, "score": z[ys, xs]})


def run(images, ids, prm, box_by_machine, machine):
    out = []
    for i in ids:
        d = detect(images[i], **prm, box_wh=box_by_machine[machine[i]])
        d.insert(0, "id", i)
        out.append(d)
    return pd.concat(out, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "data.yaml"))
    ap.add_argument("--variant", default="clean", help="clean(정제본) / raw(표시 남은 원본, 흑백 변환)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    data = ROOT / cfg["out_dir"]
    out = ROOT / "results" / f"baseline_{args.variant}"
    out.mkdir(parents=True, exist_ok=True)

    man = pd.read_csv(data / "manifest.csv")
    man = man[man["labeled"]].set_index("id")
    sizes = {i: (r.w, r.h) for i, r in man.iterrows()}
    machine = {i: int(m) for i, m in man["machine"].items()}
    split = {s: man.index[man["split"] == s].tolist() for s in ["train", "val", "test"]}
    label_dir = data / args.variant / "labels"
    gt = {s: M.load_gt(ids, label_dir, sizes) for s, ids in split.items()}
    images = {i: np.asarray(Image.open(data / args.variant / "images" / f"{i}.png").convert("L"))
              for i in tqdm(man.index, desc="영상 읽기")}

    # 박스 크기 = train 정답의 호기별 중앙값
    box_by_machine = {}
    for m in sorted(man["machine"].unique()):
        b = np.concatenate([gt["train"][i] for i in split["train"] if machine[i] == m])
        box_by_machine[int(m)] = (float(np.median(b[:, 2] - b[:, 0])), float(np.median(b[:, 3] - b[:, 1])))

    # 1) train에서 구조요소·평활·점수 방식 선택
    rows = []
    n_train = sum(len(g) for g in gt["train"].values())
    for vals in tqdm(list(itertools.product(*GRID.values())), desc="격자 탐색"):
        prm = dict(zip(GRID, vals))
        p = run(images, split["train"], prm, box_by_machine, machine)
        pm, _ = M.match(p, gt["train"])
        _, _, rec, apv = M.pr_curve(pm, n_train)
        rows.append(dict(**prm, AP=round(apv, 4), max_recall=round(float(rec[-1]), 4),
                         cand_per_img=round(len(p) / len(split["train"]), 1)))
    grid = pd.DataFrame(rows).sort_values("AP", ascending=False, kind="stable")
    grid.to_csv(out / "grid_train.csv", index=False, encoding="utf-8-sig")
    best = grid.iloc[0]
    prm = dict(se=int(best["se"]), sigma=float(best["sigma"]), score=str(best["score"]))

    # 2) val에서 임계값 두 가지 (F1 최대 / 재현율 우선)
    preds = {s: run(images, ids, prm, box_by_machine, machine) for s, ids in split.items()}
    n_val = sum(len(g) for g in gt["val"].values())
    pm_val, _ = M.match(preds["val"], gt["val"])
    thr = {"F1최대": M.best_f1_threshold(pm_val, n_val),
           f"재현율{int(RECALL_TARGET * 100)}": M.recall_threshold(pm_val, n_val, RECALL_TARGET)}
    thr[f"재현율{int(RECALL_TARGET * 100)}"] = min(thr.values())  # 재현율 우선은 F1 기준보다 느슨하게

    # 3) 모든 분할 채점 (test는 여기서 처음 본다)
    res = {"params": {**prm, "box_by_machine": box_by_machine, "cand_min": CAND_MIN[prm["score"]]},
           "thresholds": thr, "metrics": {}}
    for s in ["train", "val", "test"]:
        for name, t in thr.items():
            for rule in ["center", "iou50"]:
                res["metrics"][f"{s}/{name}/{rule}"] = M.evaluate(preds[s], gt[s], t, rule)
    for s, p in preds.items():
        pm, _ = M.match(p, gt[s])
        pm.to_csv(out / f"pred_{s}.csv", index=False, encoding="utf-8-sig")
    json.dump(res, open(out / "metrics.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # 4) test PR 곡선
    plt.rcParams["font.family"] = "Malgun Gothic"
    fig, ax = plt.subplots(figsize=(5, 4.2), dpi=150)
    n_test = sum(len(g) for g in gt["test"].values())
    for rule, c in [("center", "#1f5fa8"), ("iou50", "#9aa5b1")]:
        pm, _ = M.match(preds["test"], gt["test"], rule)
        _, prec, rec, apv = M.pr_curve(pm, n_test)
        ax.plot(rec, prec, color=c, lw=1.8, label=f"{rule}  AP {apv:.3f}")
    ax.set(xlabel="재현율", ylabel="정밀도", xlim=(0, 1), ylim=(0, 1.02),
           title=f"베이스라인 (top-hat se={prm['se']}, σ={prm['sigma']}, {prm['score']}) · test")
    ax.grid(alpha=.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "pr_test.png")

    print(grid.head(5).to_string(index=False))
    for k, v in res["metrics"].items():
        if k.startswith(("val", "test")):
            print(k, {x: v[x] for x in ["AP", "thr", "TP", "FP", "FN", "precision", "recall", "F1"]})


if __name__ == "__main__":
    main()
