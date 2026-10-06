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

ROOT = Path(__file__).resolve().parents[1]   # 저장소 최상위 폴더
# train AP 로 고를 후보값. se = 구조요소 지름(px), sigma = 가우시안 평활 세기(px, 0 은 평활 없음), score = 점수 방식
GRID = {"se": [3, 5, 7, 9], "sigma": [0.0, 0.7, 1.2], "score": ["depth", "z"]}
CAND_MIN = {"depth": 3.0, "z": 2.0}   # 후보로 남길 최소 점수 (PR 곡선을 그릴 만큼 넉넉히)
RECALL_TARGET = 0.95  # 재현율 우선 판정용 목표


def product_mask(gray):
    """배경보다 어두운 제품 영역을 Otsu 이진화로 구한다.

    gray: (H, W) uint8 흑백 영상. 반환: 같은 크기의 uint8 배열(제품 1, 배경 0).
    경계 한 겹(3×3 침식)은 깎아 낸다.
    """
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    # THRESH_BINARY_INV: Otsu 기준보다 어두운 쪽(제품)을 1로 둔다
    _, m = cv2.threshold(blur, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return cv2.erode(m, np.ones((3, 3), np.uint8))


def detect(gray, se, sigma, score, box_wh):
    """영상 한 장에서 이물 후보를 찾는다.

    gray: (H, W) uint8 흑백 영상. se: 구조요소 지름(px). sigma: 가우시안 평활 세기(px, 0 이면 생략).
    score: "depth"(top-hat 값 그대로, 회색 단계) 또는 "z"(영상 안에서 표준화한 값).
    box_wh: 후보에 씌울 박스의 (너비, 높이) px.
    반환: 열 x0, y0, x1, y1(픽셀), score 인 표. 후보 하나가 한 행이고 점수 하한은 CAND_MIN 이다.
    """
    f = gray.astype(np.float32)
    if sigma > 0:
        f = cv2.GaussianBlur(f, (0, 0), sigma)   # 창 크기 (0, 0) 은 sigma 에 맞춰 자동으로 정해진다
    # black top-hat = 닫기(영상) - 영상. 원형 구조요소보다 작은 어두운 점만 양수로 남는다
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se, se))
    th = cv2.morphologyEx(f, cv2.MORPH_BLACKHAT, k)
    pm = product_mask(gray) > 0
    # 표준화: 값이 정수라 MAD가 0으로 무너지므로, 상위 0.5%(이물 후보)를 뺀 표준편차를 쓴다
    v = th[pm]
    v = v[v <= np.quantile(v, 0.995)]
    z = (th - v.mean()) / (v.std() + 1e-3) if score == "z" else th
    # 국소 최댓값: 7×7 창 최댓값(팽창)과 같은 화소. 제품 영역 안이고 최소 점수 이상인 것만 후보로 둔다
    peak = (z == cv2.dilate(z, np.ones((7, 7), np.uint8))) & pm & (z >= CAND_MIN[score])
    ys, xs = np.nonzero(peak)   # 행(y), 열(x) 순서로 나온다
    bw, bh = box_wh
    # 화소 (x, y) 의 중심은 연속 좌표로 (x + 0.5, y + 0.5) 이다. 박스는 그 중심에 맞춘다
    return pd.DataFrame({"x0": xs - bw / 2 + 0.5, "y0": ys - bh / 2 + 0.5,
                         "x1": xs + bw / 2 + 0.5, "y1": ys + bh / 2 + 0.5, "score": z[ys, xs]})


def run(images, ids, prm, box_by_machine, machine):
    """여러 영상에 detect 를 돌려 한 표로 모은다.

    images: {id: 흑백 배열}, ids: 처리할 영상 id 목록, prm: {"se", "sigma", "score"},
    box_by_machine: {호기: (너비, 높이) px}, machine: {id: 호기}.
    반환: 열 id, x0, y0, x1, y1, score 인 표.
    """
    out = []
    for i in ids:
        d = detect(images[i], **prm, box_wh=box_by_machine[machine[i]])
        d.insert(0, "id", i)
        out.append(d)
    return pd.concat(out, ignore_index=True)


def main():
    """격자 탐색(train) → 임계값 결정(val) → 전 분할 채점 → 결과 저장.

    results/baseline_<variant>/ 에 grid_train.csv, pred_<분할>.csv, metrics.json, pr_test.png 를 남긴다.
    metrics.json 의 params 는 cnn.py · judge.py 가 다시 읽어 쓴다.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "data.yaml"))
    ap.add_argument("--variant", default="clean", help="clean(정제본) / raw(표시 남은 원본, 흑백 변환)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    data = ROOT / cfg["out_dir"]
    out = ROOT / "results" / f"baseline_{args.variant}"
    out.mkdir(parents=True, exist_ok=True)

    # 정답(라벨)이 있는 영상만 쓴다
    man = pd.read_csv(data / "manifest.csv")
    man = man[man["labeled"]].set_index("id")
    sizes = {i: (r.w, r.h) for i, r in man.iterrows()}                 # {id: (너비, 높이) px}
    machine = {i: int(m) for i, m in man["machine"].items()}           # {id: 호기}
    split = {s: man.index[man["split"] == s].tolist() for s in ["train", "val", "test"]}
    label_dir = data / args.variant / "labels"
    gt = {s: M.load_gt(ids, label_dir, sizes) for s, ids in split.items()}   # {분할: {id: (N, 4) xyxy 픽셀}}
    images = {i: np.asarray(Image.open(data / args.variant / "images" / f"{i}.png").convert("L"))
              for i in tqdm(man.index, desc="영상 읽기")}

    # 박스 크기 = train 정답의 호기별 중앙값
    # 이 방법은 점의 위치만 찾고 크기는 재지 않으므로, 모든 후보에 같은 크기 박스를 씌운다
    box_by_machine = {}
    for m in sorted(man["machine"].unique()):
        b = np.concatenate([gt["train"][i] for i in split["train"] if machine[i] == m])
        box_by_machine[int(m)] = (float(np.median(b[:, 2] - b[:, 0])), float(np.median(b[:, 3] - b[:, 1])))

    # 1) train에서 구조요소·평활·점수 방식 선택
    rows = []
    n_train = sum(len(g) for g in gt["train"].values())
    for vals in tqdm(list(itertools.product(*GRID.values())), desc="격자 탐색"):   # 4 × 3 × 2 = 24 조합
        prm = dict(zip(GRID, vals))
        p = run(images, split["train"], prm, box_by_machine, machine)
        pm, _ = M.match(p, gt["train"])
        _, _, rec, apv = M.pr_curve(pm, n_train)
        # max_recall = 후보를 전부 받았을 때의 재현율, cand_per_img = 영상당 후보 수
        rows.append(dict(**prm, AP=round(apv, 4), max_recall=round(float(rec[-1]), 4),
                         cand_per_img=round(len(p) / len(split["train"]), 1)))
    # AP 가 같으면 GRID 에 적힌 순서가 앞선 조합을 고른다 (안정 정렬)
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
    # metrics 의 키는 "<분할>/<임계값 이름>/<맞춤 기준>" 이다
    res = {"params": {**prm, "box_by_machine": box_by_machine, "cand_min": CAND_MIN[prm["score"]]},
           "thresholds": thr, "metrics": {}}
    for s in ["train", "val", "test"]:
        for name, t in thr.items():
            for rule in ["center", "iou50"]:
                res["metrics"][f"{s}/{name}/{rule}"] = M.evaluate(preds[s], gt[s], t, rule)
    # 후보 전체를 임계값으로 자르지 않고 저장한다. tp · gt_idx 열은 중심 일치 기준이다
    for s, p in preds.items():
        pm, _ = M.match(p, gt[s])
        pm.to_csv(out / f"pred_{s}.csv", index=False, encoding="utf-8-sig")
    json.dump(res, open(out / "metrics.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # 4) test PR 곡선
    plt.rcParams["font.family"] = "Malgun Gothic"   # 그래프의 한글 글꼴
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

    # 화면 요약: 격자 탐색 상위 5개와 val · test 지표
    print(grid.head(5).to_string(index=False))
    for k, v in res["metrics"].items():
        if k.startswith(("val", "test")):
            print(k, {x: v[x] for x in ["AP", "thr", "TP", "FP", "FN", "precision", "recall", "F1"]})


if __name__ == "__main__":
    main()
