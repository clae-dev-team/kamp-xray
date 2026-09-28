"""합성 이물 증강 학습셋: train 영상에만 Beer–Lambert 합성 이물을 넣어 위치 편향을 줄인다.

실제 이물은 모두 어두운 띠 왼쪽 끝에 있어, 그대로 학습한 모델은 '점 + 그 자리'를 함께 외운다
(results/location_test). 제품 안 여러 자리에 합성 이물을 넣어 '어두운 점' 자체를 보게 만든다.

  - train 350장만 쓴다. val·test 영상에는 넣지 않는다 (임계값 결정·채점은 실제 이물로만).
  - 영상마다 VARIANTS장을 만들고, 한 장에 합성 이물 2~4개를 넣는다. 실제 라벨은 그대로 둔다.
  - 진하기 c0, 지름 d 는 넓게 뽑고, 30%는 길쭉한 파편(가로세로비 2~4, 임의 각도)으로 만든다.
  - 평가셋(synth.py)은 구 모양·고정 격자라 학습 분포와 겹치지 않게 난수 시드를 따로 쓴다.

실행: .venv\\Scripts\\python.exe src\\augment.py
결과: data/aug/{images,labels}, data/aug/train.txt(원본 train + 합성), data/aug.yaml(val·test는 정제본 그대로)
"""
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm

from prepare import product_mask
from synth import band_mask, insert, noise_residual

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = 3
N_RANGE = (2, 4)             # 한 장에 넣는 합성 이물 수
C0_RANGE = (0.15, 0.85)      # 명목 대비
D_RANGE = (1.5, 4.0)         # 지름 px
SHARD_P = 0.30               # 길쭉한 파편 비율
ASPECT_RANGE = (2.0, 4.0)
BAND_P = 0.5                 # 띠 안에 놓을 확률 (나머지는 제품 안 아무 곳)
MIN_BOX = 10                 # 실제 라벨 중앙 크기에 맞춘 최소 박스
SEED_OFFSET = 7919           # 평가셋 난수와 겹치지 않게


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", action="store_true", help="X선 잡음 보정 (결과: data/aug_n, data/aug_n.yaml)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "configs" / "data.yaml", encoding="utf-8"))
    data = ROOT / cfg["out_dir"]
    tag = "aug_n" if args.noise else "aug"
    out = data / tag
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(data / "manifest.csv")
    tr = man[man["labeled"] & (man["split"] == "train")]

    rows, paths = [], []
    for r in tqdm(list(tr.itertuples()), desc="증강"):
        g = np.asarray(Image.open(data / "clean/images" / f"{r.id}.png"))
        h, w = g.shape
        real = np.loadtxt(data / "clean/labels" / f"{r.id}.txt", ndmin=2)
        pm = product_mask(g)
        band = band_mask(g, pm)
        forbid = np.zeros_like(pm)
        for b in real:
            cx, cy, bw, bh = b[1] * w, b[2] * h, b[3] * w, b[4] * h
            forbid[max(0, int(cy - bh / 2 - 10)):int(cy + bh / 2 + 10),
                   max(0, int(cx - bw / 2 - 10)):int(cx + bw / 2 + 10)] = 1
        cand = {"band": np.argwhere(band & (forbid == 0)), "any": np.argwhere((pm > 0) & (forbid == 0))}
        for v in range(VARIANTS):
            rng = np.random.default_rng([cfg["seed"] + SEED_OFFSET, int(r.sha1[:8], 16), v])
            f = g.astype(np.float32)
            res = noise_residual(g) if args.noise else None
            nrng = np.random.default_rng([cfg["seed"] + SEED_OFFSET, int(r.sha1[:8], 16), v, 99])
            placed, labels = [], [tuple(b) for b in real]
            for _ in range(rng.integers(N_RANGE[0], N_RANGE[1] + 1)):
                pool = cand["band"] if (rng.random() < BAND_P and len(cand["band"])) else cand["any"]
                for _try in range(100):
                    y, x = pool[rng.integers(len(pool))]
                    cx, cy = x + rng.random(), y + rng.random()
                    if all(np.hypot(cx - px, cy - py) >= 24 for px, py in placed):
                        break
                c0 = rng.uniform(*C0_RANGE)
                d = rng.uniform(*D_RANGE)
                shard = rng.random() < SHARD_P
                aspect = rng.uniform(*ASPECT_RANGE) if shard else 1.0
                angle = rng.uniform(0, np.pi)
                insert(f, cx, cy, d, c0, aspect, angle, residual=res, rng=nrng)
                # 박스: 모양이 차지하는 범위 + 여유, 최소 MIN_BOX
                ex = abs(d * aspect / 2 * np.cos(angle)) + abs(d / 2 * np.sin(angle))
                ey = abs(d * aspect / 2 * np.sin(angle)) + abs(d / 2 * np.cos(angle))
                bw, bh = max(MIN_BOX, 2 * ex + 6), max(MIN_BOX, 2 * ey + 6)
                labels.append((0, cx / w, cy / h, bw / w, bh / h))
                placed.append((cx, cy))
                sid = f"{r.id}__a{v}"
                rows.append(dict(img=sid, src=r.id, machine=r.machine, cx=cx, cy=cy, d=d, c0=c0,
                                 shard=shard, aspect=aspect, in_band=bool(band[int(cy), int(cx)])))
            sid = f"{r.id}__a{v}"
            Image.fromarray(np.clip(f.round(), 0, 255).astype(np.uint8)).save(out / "images" / f"{sid}.png")
            np.savetxt(out / "labels" / f"{sid}.txt", np.array(labels), fmt="%d %.6f %.6f %.6f %.6f")
            paths.append(str((out / "images" / f"{sid}.png").resolve()))

    orig = (data / "clean" / "train.txt").read_text(encoding="utf-8").split()
    (out / "train.txt").write_text("\n".join(orig + paths) + "\n", encoding="utf-8")
    ds = {"path": str((data / "clean").resolve()), "train": str((out / "train.txt").resolve()),
          "val": "val.txt", "test": "test.txt", "names": {0: "Defect"}}
    yaml.safe_dump(ds, open(data / f"{tag}.yaml", "w", encoding="utf-8"), allow_unicode=True)
    pd.DataFrame(rows).to_csv(out / "defects.csv", index=False, encoding="utf-8-sig")
    json.dump(dict(variants=VARIANTS, n_range=N_RANGE, c0_range=C0_RANGE, d_range=D_RANGE, shard_p=SHARD_P,
                   aspect_range=ASPECT_RANGE, band_p=BAND_P, n_images=len(paths), n_defects=len(rows),
                   n_train_total=len(orig) + len(paths)),
              open(out / "config.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(len(orig), "+", len(paths), "장, 합성 이물", len(rows), "개")


if __name__ == "__main__":
    main()
