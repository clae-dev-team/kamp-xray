"""이미지 단위 판정용 평가셋: 이물을 지운 '가짜 정상' + 실제 불량 + 합성 불량.

제공 데이터는 전부 불량(NG) 영상이라 정상 영상이 없다. 그래서 val·test 영상에서 실제 이물 점만
지워(location_test 의 E 조건과 같은 방법) 같은 제품의 정상 영상을 만든다. 이물 유무만 다른 짝이 된다.
실제 이물은 모두 선명해 판정 기준을 정하기엔 너무 쉬우므로, 가짜 정상 위에 눈에 보이는 범위의
합성 이물을 하나씩 넣은 '합성 불량'도 함께 만든다.

  normal/<split>/<id>.png          가짜 정상 (split = val, test)
  synth_ng/<split>/<id>__nK.png    가짜 정상 + 합성 이물 1개 (K = 0..VARIANTS-1)
  judge_<split>.csv                img, kind(real_ng/normal/synth_ng), path, 합성 조건
지운 자리가 제대로 지워졌는지(대비가 잡음 수준인지)는 results/normal_set/check.json 에 남긴다.

한계: 사람이 표시하지 않은 옅은 이물이 영상에 남아 있으면 가짜 정상에도 그대로 남는다.
실행: .venv\\Scripts\\python.exe src\\normal_set.py
"""
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm

from defect_stats import measure
from location_test import dot_mask
from prepare import product_mask, restore
from synth import band_mask, insert

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = 4
C0_RANGE = (0.20, 0.70)      # 눈에 보이는 범위 (실제 이물 측정 대비 5% 지점 0.27 근처까지 포함)
D_RANGE = (1.5, 4.0)
SHARD_P = 0.30
SEED_OFFSET = 104729         # 학습 증강·합성 평가셋과 다른 난수


def dot_centers(g, boxes, w, h):
    """라벨 박스 안 2×2 평균 최솟값 = 이물 점 중심."""
    out = []
    for b in boxes:
        bw, bh = b[3] * w, b[4] * h
        x0, y0 = int(max(0, b[1] * w - bw / 2)), int(max(0, b[2] * h - bh / 2))
        sub = cv2.blur(g[y0:int(b[2] * h + bh / 2) + 1, x0:int(b[1] * w + bw / 2) + 1].astype(np.float32), (2, 2))
        iy, ix = np.unravel_index(np.argmin(sub), sub.shape)
        out.append((x0 + ix + 0.5, y0 + iy + 0.5))
    return out


def main():
    cfg = yaml.safe_load(open(ROOT / "configs" / "data.yaml", encoding="utf-8"))
    data = ROOT / cfg["out_dir"]
    man = pd.read_csv(data / "manifest.csv")
    rows, checks = [], []
    for split in ["val", "test"]:
        (data / "normal" / split).mkdir(parents=True, exist_ok=True)
        (data / "synth_ng" / split).mkdir(parents=True, exist_ok=True)
        sub = man[man["labeled"] & (man["split"] == split)]
        for r in tqdm(list(sub.itertuples()), desc=f"가짜 정상 {split}"):
            g = np.asarray(Image.open(data / "clean/images" / f"{r.id}.png"))
            h, w = g.shape
            boxes = np.loadtxt(data / "clean/labels" / f"{r.id}.txt", ndmin=2)
            centers = dot_centers(g, boxes, w, h)
            mask = np.zeros_like(g)
            for cx, cy in centers:
                mask |= dot_mask(g, cx - 0.5, cy - 0.5)
            rng = np.random.default_rng([cfg["seed"] + SEED_OFFSET, int(r.sha1[:8], 16)])
            normal = restore(g, mask, rng)
            npath = data / "normal" / split / f"{r.id}.png"
            Image.fromarray(normal).save(npath)
            for cx, cy in centers:
                before = measure(g, cx - 0.5, cy - 0.5)["contrast"]
                after = measure(normal, cx - 0.5, cy - 0.5)["contrast"]
                checks.append(dict(split=split, id=r.id, before=before, after=after))
            rows.append(dict(split=split, img=r.id, src=r.id, machine=r.machine, kind="real_ng",
                             path=str(data / "clean/images" / f"{r.id}.png")))
            rows.append(dict(split=split, img=f"{r.id}__normal", src=r.id, machine=r.machine, kind="normal",
                             path=str(npath)))

            # 합성 불량: 가짜 정상 위 제품 안 아무 곳(띠 안 절반)에 이물 1개
            pm = product_mask(normal)
            band = band_mask(normal, pm)
            pools = {"band": np.argwhere(band), "any": np.argwhere(pm > 0)}
            for k in range(VARIANTS):
                rk = np.random.default_rng([cfg["seed"] + SEED_OFFSET, int(r.sha1[:8], 16), k + 1])
                pool = pools["band"] if (rk.random() < 0.5 and len(pools["band"])) else pools["any"]
                y, x = pool[rk.integers(len(pool))]
                cx, cy = x + rk.random(), y + rk.random()
                c0, d = rk.uniform(*C0_RANGE), rk.uniform(*D_RANGE)
                shard = rk.random() < SHARD_P
                aspect = rk.uniform(2, 4) if shard else 1.0
                f = normal.astype(np.float32)
                insert(f, cx, cy, d, c0, aspect, rk.uniform(0, np.pi))
                img = np.clip(f.round(), 0, 255).astype(np.uint8)
                spath = data / "synth_ng" / split / f"{r.id}__n{k}.png"
                Image.fromarray(img).save(spath)
                rows.append(dict(split=split, img=f"{r.id}__n{k}", src=r.id, machine=r.machine, kind="synth_ng",
                                 path=str(spath), cx=cx, cy=cy, c0=c0, d=d, shard=shard,
                                 in_band=bool(band[int(cy), int(cx)]),
                                 c_meas=measure(img, cx - 0.5, cy - 0.5)["contrast"]))

    df = pd.DataFrame(rows)
    for split in ["val", "test"]:
        df[df["split"] == split].to_csv(data / f"judge_{split}.csv", index=False, encoding="utf-8-sig")
    ck = pd.DataFrame(checks)
    out = ROOT / "results" / "normal_set"
    out.mkdir(parents=True, exist_ok=True)
    real = pd.read_csv(ROOT / "results/defect_stats/real_defects.csv")
    summary = {
        "지운_이물수": len(ck),
        "지우기전_대비_중앙": round(float(ck["before"].median()), 3),
        "지운뒤_대비_중앙": round(float(ck["after"].median()), 3),
        "지운뒤_대비_최대": round(float(ck["after"].max()), 3),
        "실제이물_최소대비": round(float(real["contrast"].min()), 3),
        "영상수": df.groupby(["split", "kind"]).size().rename(lambda t: "/".join(t)).to_dict()
                   if False else {f"{s}/{k}": int(n) for (s, k), n in df.groupby(["split", "kind"]).size().items()},
    }
    ck.to_csv(out / "erase_check.csv", index=False, encoding="utf-8-sig")
    json.dump(summary, open(out / "check.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
