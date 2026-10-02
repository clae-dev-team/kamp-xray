"""회색 선 네모 미끼 재시험: 최종 AI가 색 네모 미끼에 27% 반응한 원인이 '색'인지 '네모 모양의 이상 구조'인지 가린다.

shortcut_test.py 의 C 미끼(정제본의 이물 없는 자리 = 가짜 사각형 자리에 네모만 그림)와 같은 자리·같은 선 두께로
선의 성질만 바꿔 네 가지를 비교한다.
  ① 색      : 원래 미끼 (빨강·노랑·파랑·자홍, 실측 비율)  → 9/30 결과 재현
  ② 같은 밝기 회색 : ①과 같은 색을 고른 뒤, 그 색의 밝기(Y = 0.299R + 0.587G + 0.114B)와 같은 회색으로 칠함
  ③ 어두운 회색   : 선 자리 밝기 × 0.6 (X선을 더 막는 얇은 철사 같은 물체, 곱셈)
  ④ 밝은 회색     : 선 자리 밝기 × 1.3 (덜 막는 틈 같은 구조)
반응 = 판정 임계값(val F1 최대) 이상 예측의 중심이 미끼 네모(±2px) 안 (shortcut_test.bait_hits 와 같음).
실제 이물 재현율도 함께 잰다 (미끼가 진짜 이물 검출을 방해하는지).

실행: .venv\\Scripts\\python.exe src\\bait_gray.py --models ratio3_e100 y26s_640
결과: results/bait_gray/ (summary.json, summary.csv, examples.png)
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
from shortcut_test import COLORS, bait_hits, predict
from train_yolo import weights_path

ROOT = Path(__file__).resolve().parents[1]


def draw(gray, rects, rng, mode):
    """mode: color / luma / dark / bright. 선 두께 2px, 같은 rng 순서로 색을 골라 ①②가 같은 선 배치를 갖게 한다."""
    cols, prob = zip(*COLORS)
    pr = np.array(prob) / sum(prob)
    f = gray.astype(np.float32)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    for x0, y0, x1, y1 in rects:
        c = cols[rng.choice(len(cols), p=pr)]
        m = np.zeros(gray.shape, np.uint8)
        cv2.rectangle(m, (int(x0), int(y0)), (int(x1) - 1, int(y1) - 1), 1, 2)
        m = m.astype(bool)
        if mode == "color":
            rgb[m] = c
        elif mode == "luma":
            rgb[m] = round(0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2])
        elif mode == "dark":
            rgb[m] = np.clip(f[m] * 0.6, 0, 255).round().astype(np.uint8)[:, None]
        elif mode == "bright":
            rgb[m] = np.clip(f[m] * 1.3, 0, 255).round().astype(np.uint8)[:, None]
    return rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "data.yaml"))
    ap.add_argument("--models", nargs="+", default=["ratio3_e100", "y26s_640"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from ultralytics import YOLO
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    data = ROOT / cfg["out_dir"]
    out = ROOT / "results" / "bait_gray"
    out.mkdir(parents=True, exist_ok=True)

    man = pd.read_csv(data / "manifest.csv")
    man = man[man["labeled"] & (man["split"] == "test")].set_index("id")
    ids = man.index.tolist()
    gt = M.load_gt(ids, data / "clean" / "labels", {i: (r.w, r.h) for i, r in man.iterrows()})
    bait = pd.read_csv(data / "marks.csv")
    bait = bait[(bait["kind"] == "fake") & bait["id"].isin(ids)]
    clean = {i: np.asarray(Image.open(data / "clean/images" / f"{i}.png").convert("L")) for i in ids}
    modes = {"① 색": "color", "② 같은 밝기 회색": "luma", "③ 어두운 회색(×0.6)": "dark", "④ 밝은 회색(×1.3)": "bright"}
    inputs = {}
    for tag, mode in modes.items():
        rng = np.random.default_rng(args.seed)          # 모든 미끼가 같은 색 선택 순서를 쓰도록 매번 새로
        inputs[tag] = [draw(clean[i], bait.loc[bait["id"] == i, ["x0", "y0", "x1", "y1"]].to_numpy(), rng, mode) for i in ids]

    rows = []
    for name in args.models:
        model = YOLO(str(weights_path(name)))
        thr = json.load(open(ROOT / "results" / f"yolo_{name}" / "metrics.json", encoding="utf-8"))["thresholds"]["F1최대"]
        for tag, imgs in inputs.items():
            p = predict(model, imgs, ids)
            h, med, mx = bait_hits(p, bait, thr)
            ev = M.evaluate(p, gt, thr, "center")
            rows.append(dict(model=name, 미끼=tag, 미끼수=len(bait), 미끼반응=h, 미끼반응률=round(h / len(bait), 3),
                             미끼점수_중앙=round(med, 3), 미끼점수_최대=round(mx, 3),
                             실제이물_재현율=ev["recall"], 실제이물_헛경보=ev["FP"], 임계값=round(thr, 3)))
            print(rows[-1])
    df = pd.DataFrame(rows)
    df.to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
    json.dump(rows, open(out / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # 예시: 한 사진의 미끼 하나를 네 가지로 확대 (글자 없음)
    i = ids[len(ids) // 3]
    r = bait[bait["id"] == i].iloc[0]
    cx, cy = int((r.x0 + r.x1) / 2), int((r.y0 + r.y1) / 2)
    tiles = []
    for tag in modes:
        im = inputs[tag][ids.index(i)]
        pad = np.pad(im, ((24, 24), (24, 24), (0, 0)), mode="edge")
        c = pad[cy:cy + 48, cx:cx + 48]
        tiles += [cv2.resize(c, (192, 192), interpolation=cv2.INTER_NEAREST), np.full((192, 8, 3), 255, np.uint8)]
    Image.fromarray(np.hstack(tiles[:-1])).save(out / "examples.png")


if __name__ == "__main__":
    main()
