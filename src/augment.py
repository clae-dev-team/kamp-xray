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
VARIANTS = 3                 # train 영상 한 장당 만드는 합성 영상 수 (--variants 기본값)
N_RANGE = (2, 4)             # 한 장에 넣는 합성 이물 수
C0_RANGE = (0.15, 0.85)      # 명목 대비
D_RANGE = (1.5, 4.0)         # 지름 px
SHARD_P = 0.30               # 길쭉한 파편 비율
ASPECT_RANGE = (2.0, 4.0)    # 파편의 가로세로비 (긴 지름 = 지름 × 이 값)
BAND_P = 0.5                 # 띠 안에 놓을 확률 (나머지는 제품 안 아무 곳)
MIN_BOX = 10                 # 실제 라벨 중앙 크기에 맞춘 최소 박스
SEED_OFFSET = 7919           # 평가셋 난수와 겹치지 않게


def main():
    """train 영상마다 합성 영상을 만들고, 원본 train 과 합친 학습 목록·데이터셋 yaml 을 쓴다.

    결과 폴더 이름: aug (기본), aug_n (--noise), 장수가 3이 아니면 뒤에 _x<N>.
    defects.csv 는 합성 이물별 img, src, machine, cx, cy(px), d(px), c0, shard, aspect, in_band.
    """
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", action="store_true", help="X선 잡음 보정 (결과: data/aug_n, data/aug_n.yaml)")
    ap.add_argument("--variants", type=int, default=VARIANTS, help="train 영상 한 장당 합성 영상 수 (3이 아니면 결과 폴더 aug_x<N>)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "configs" / "data.yaml", encoding="utf-8"))
    data = ROOT / cfg["out_dir"]
    tag = "aug_n" if args.noise else "aug"
    if args.variants != VARIANTS:
        tag += f"_x{args.variants}"
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
        # 실제 이물 박스와 그 둘레 10px 에는 합성 이물을 놓지 않는다
        forbid = np.zeros_like(pm)
        for b in real:
            cx, cy, bw, bh = b[1] * w, b[2] * h, b[3] * w, b[4] * h
            forbid[max(0, int(cy - bh / 2 - 10)):int(cy + bh / 2 + 10),
                   max(0, int(cx - bw / 2 - 10)):int(cx + bw / 2 + 10)] = 1
        # 후보 자리 두 종류: 띠 안 / 제품 안 아무 곳. argwhere 결과는 (행 y, 열 x) 순서다
        cand = {"band": np.argwhere(band & (forbid == 0)), "any": np.argwhere((pm > 0) & (forbid == 0))}
        for v in range(args.variants):
            # 난수는 (시드 + SEED_OFFSET, 영상 해시, 변형 번호)로 고정해 다시 돌려도 같은 파일이 나온다
            rng = np.random.default_rng([cfg["seed"] + SEED_OFFSET, int(r.sha1[:8], 16), v])
            f = g.astype(np.float32)
            res = noise_residual(g) if args.noise else None
            # 잡음 보정용 난수를 따로 둬서, 보정을 켜도 이물 위치·조건을 뽑는 난수 순서가 바뀌지 않게 한다
            nrng = np.random.default_rng([cfg["seed"] + SEED_OFFSET, int(r.sha1[:8], 16), v, 99])
            # 라벨은 실제 이물 박스에서 시작해 합성 이물 박스를 덧붙인다
            placed, labels = [], [tuple(b) for b in real]
            for _ in range(rng.integers(N_RANGE[0], N_RANGE[1] + 1)):
                # 띠 안 후보가 하나도 없으면 제품 안 아무 곳에서 뽑는다
                pool = cand["band"] if (rng.random() < BAND_P and len(cand["band"])) else cand["any"]
                # 먼저 넣은 합성 이물과 중심이 24px 이상 떨어진 자리를 100번까지 뽑는다 (끝내 못 찾으면 마지막에 뽑은 자리를 쓴다)
                for _try in range(100):
                    y, x = pool[rng.integers(len(pool))]
                    # 픽셀 안에서의 위치도 무작위로 둔다 (소수 좌표)
                    cx, cy = x + rng.random(), y + rng.random()
                    if all(np.hypot(cx - px, cy - py) >= 24 for px, py in placed):
                        break
                c0 = rng.uniform(*C0_RANGE)
                d = rng.uniform(*D_RANGE)
                shard = rng.random() < SHARD_P
                aspect = rng.uniform(*ASPECT_RANGE) if shard else 1.0
                # 각도는 파편이 아니어도 뽑는다 (구는 돌려도 모양이 같다)
                angle = rng.uniform(0, np.pi)
                insert(f, cx, cy, d, c0, aspect, angle, residual=res, rng=nrng)
                # 박스: 모양이 차지하는 범위 + 여유, 최소 MIN_BOX
                # ex, ey = 돌린 타원의 긴 반지름·짧은 반지름을 가로·세로 축에 비춘 길이의 합 (실제 범위보다 조금 넉넉하다). 여기에 양쪽 3px 씩 더한다
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

    # 학습 목록 = 원본 train 영상 + 합성 영상. val·test 는 정제본 폴더의 목록을 그대로 가리킨다
    orig = (data / "clean" / "train.txt").read_text(encoding="utf-8").split()
    (out / "train.txt").write_text("\n".join(orig + paths) + "\n", encoding="utf-8")
    ds = {"path": str((data / "clean").resolve()), "train": str((out / "train.txt").resolve()),
          "val": "val.txt", "test": "test.txt", "names": {0: "Defect"}}
    yaml.safe_dump(ds, open(data / f"{tag}.yaml", "w", encoding="utf-8"), allow_unicode=True)
    pd.DataFrame(rows).to_csv(out / "defects.csv", index=False, encoding="utf-8-sig")
    json.dump(dict(variants=args.variants, n_range=N_RANGE, c0_range=C0_RANGE, d_range=D_RANGE, shard_p=SHARD_P,
                   aspect_range=ASPECT_RANGE, band_p=BAND_P, n_images=len(paths), n_defects=len(rows),
                   n_train_total=len(orig) + len(paths)),
              open(out / "config.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(len(orig), "+", len(paths), "장, 합성 이물", len(rows), "개")


if __name__ == "__main__":
    main()
