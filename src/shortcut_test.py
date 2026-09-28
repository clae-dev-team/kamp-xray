"""지름길 학습 검증: 색 표시가 남은 원본으로 학습한 모델이 '이물' 대신 '색 네모'를 배웠는지 본다.

두 모델(정제본 학습 / 원본 학습)을 같은 test 영상의 세 가지 버전에 돌린다.
  A 원본      : 표시가 남은 그대로
  B 정제본    : 표시를 지운 것 (현장 실제 입력과 같은 조건)
  C 미끼      : 정제본의 이물 없는 자리(가짜 사각형 위치)에 색 네모만 새로 그린 것
지름길을 배웠다면 원본 학습 모델은 A에서만 잘하고, B에서 성능이 떨어지고, C의 빈 네모에 반응한다.

실행: .venv\\Scripts\\python.exe src\\shortcut_test.py --clean y26s_640 --raw y26s_640_raw
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from PIL import Image

import metrics as M
from train_yolo import weights_path

ROOT = Path(__file__).resolve().parents[1]
# 원본 표시에서 실측한 색 빈도 (빨강이 대부분)
COLORS = [((255, 0, 0), .72), ((255, 255, 0), .12), ((0, 0, 255), .12), ((255, 0, 255), .04)]


def draw_bait(gray, rects, rng):
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    cols, prob = zip(*COLORS)
    for x0, y0, x1, y1 in rects:
        c = cols[rng.choice(len(cols), p=np.array(prob) / sum(prob))]
        cv2.rectangle(rgb, (int(x0), int(y0)), (int(x1) - 1, int(y1) - 1), c, 2)
    return rgb


def predict(model, imgs_rgb, ids, imgsz=640):
    rows = []
    for i, im in zip(ids, imgs_rgb):
        r = model.predict(cv2.cvtColor(im, cv2.COLOR_RGB2BGR), imgsz=imgsz, conf=0.001,
                          max_det=100, verbose=False)[0]
        for (x0, y0, x1, y1), s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            rows.append(dict(id=i, x0=x0, y0=y0, x1=x1, y1=y1, score=float(s)))
    return pd.DataFrame(rows, columns=["id", "x0", "y0", "x1", "y1", "score"])


def bait_hits(pred, bait, thr):
    """판정 임계값 이상 예측 중 중심이 미끼 네모 안에 떨어진 것 = 표시만 보고 '이물'이라 한 경우."""
    hit = 0
    top = []
    for r in bait.itertuples():
        q = pred[pred["id"] == r.id]
        cx, cy = (q.x0 + q.x1) / 2, (q.y0 + q.y1) / 2
        s = q[(cx >= r.x0 - 2) & (cx <= r.x1 + 2) & (cy >= r.y0 - 2) & (cy <= r.y1 + 2)]["score"]
        m = float(s.max()) if len(s) else 0.0
        top.append(m)
        hit += int(m >= thr)
    return hit, float(np.median(top)), float(np.max(top))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "data.yaml"))
    ap.add_argument("--clean", default="y26s_640")
    ap.add_argument("--raw", default="y26s_640_raw")
    ap.add_argument("--split", default="test")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from ultralytics import YOLO

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    data = ROOT / cfg["out_dir"]
    out = ROOT / "results" / "shortcut"
    out.mkdir(parents=True, exist_ok=True)

    man = pd.read_csv(data / "manifest.csv")
    man = man[man["labeled"] & (man["split"] == args.split)].set_index("id")
    ids = man.index.tolist()
    gt = M.load_gt(ids, data / "clean" / "labels", {i: (r.w, r.h) for i, r in man.iterrows()})
    bait = pd.read_csv(data / "marks.csv")
    bait = bait[(bait["kind"] == "fake") & bait["id"].isin(ids)]

    rng = np.random.default_rng(args.seed)
    clean = {i: np.asarray(Image.open(data / "clean/images" / f"{i}.png")) for i in ids}
    inputs = {
        "A_원본": [np.asarray(Image.open(data / "raw/images" / f"{i}.png").convert("RGB")) for i in ids],
        "B_정제본": [cv2.cvtColor(clean[i], cv2.COLOR_GRAY2RGB) for i in ids],
        "C_미끼": [draw_bait(clean[i], bait.loc[bait["id"] == i, ["x0", "y0", "x1", "y1"]].to_numpy(), rng)
                  for i in ids],
    }

    res, examples = {}, {}
    for tag, name in [("정제본학습", args.clean), ("원본학습", args.raw)]:
        model = YOLO(str(weights_path(name)))
        thr = json.load(open(ROOT / "results" / f"yolo_{name}" / "metrics.json", encoding="utf-8"))["thresholds"]["F1최대"]
        for cond, imgs in inputs.items():
            p = predict(model, imgs, ids)
            r = M.evaluate(p, gt, thr, "center")
            row = {k: r[k] for k in ["AP", "TP", "FP", "FN", "precision", "recall", "F1"]}
            row["thr"] = round(thr, 3)
            if cond == "C_미끼":
                h, med, mx = bait_hits(p, bait, thr)
                row.update(미끼수=len(bait), 미끼반응=h, 미끼반응률=round(h / len(bait), 3),
                           미끼점수_중앙=round(med, 3), 미끼점수_최대=round(mx, 3))
                examples[tag] = p
            res[f"{tag}/{cond}"] = row
            print(tag, cond, row)

    json.dump(res, open(out / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    pd.DataFrame(res).T.to_csv(out / "summary.csv", encoding="utf-8-sig")

    # 미끼 영상 예시: 왼쪽 정제본 학습 모델, 오른쪽 원본 학습 모델 (초록 네모=정답 이물, 주황 원=모델 검출)
    tiles = []
    for i in ids[::max(1, len(ids) // 4)][:4]:
        im = inputs["C_미끼"][ids.index(i)]
        row = []
        for tag, name in [("정제본학습", args.clean), ("원본학습", args.raw)]:
            thr = res[f"{tag}/C_미끼"]["thr"]
            v = im.copy()
            for b in gt[i]:
                cv2.rectangle(v, (int(b[0]) - 1, int(b[1]) - 1), (int(b[2]) + 1, int(b[3]) + 1), (0, 200, 0), 1)
            q = examples[tag]
            for r in q[(q["id"] == i) & (q["score"] >= thr)].itertuples():
                cv2.circle(v, (int((r.x0 + r.x1) / 2), int((r.y0 + r.y1) / 2)), 6, (255, 170, 0), 1)
            row.append(v)
        pair = np.hstack([row[0], np.full((row[0].shape[0], 6, 3), 255, np.uint8), row[1]])
        tiles.append(cv2.resize(pair, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
    w = max(t.shape[1] for t in tiles)
    tiles = [np.pad(t, ((0, 8), (0, w - t.shape[1]), (0, 0)), constant_values=255) for t in tiles]
    Image.fromarray(np.vstack(tiles)).save(out / "bait_examples.png")


if __name__ == "__main__":
    main()
