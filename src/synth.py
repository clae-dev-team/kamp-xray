"""Beer–Lambert 기반 합성 이물 생성 (미탐지 조건 분석용 평가셋).

X선 투과 세기는 I = I0·exp(-μt) 이므로, 물체 뒤에 이물이 하나 더 겹치면 그 자리 픽셀은
이물 투과율 T = exp(-μ_이물·t_이물) 만큼 곱해져 어두워진다. 밝은 곳에 놓이면 많이, 어두운 띠 위면 적게
어두워지는 실제 X-ray 성질이 그대로 따라온다. (픽셀 값이 투과 세기에 비례한다고 가정)

이물 모양은 지름 d px의 구: 두께 t(r) = sqrt(1-(2r/d)²). 8배 세분화 격자에서 그려 픽셀 평균을 내므로
지름이 1~2px로 작으면 부분 체적 효과로 실제 대비가 명목값보다 옅어진다 (현실과 같다).
  c0 = 이물 중심 한 줄기에서 줄어드는 투과율 비율 (재질·두께가 정하는 명목 대비)

test 분할 영상에만 넣는다 (학습에 쓰인 적 없는 영상). 실제 이물 주변과 서로 간에는 거리를 둔다.
실행: .venv\\Scripts\\python.exe src\\synth.py
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
from prepare import product_mask

ROOT = Path(__file__).resolve().parents[1]
CONTRASTS = [0.03, 0.05, 0.08, 0.12, 0.16, 0.20, 0.25, 0.30, 0.40, 0.55, 0.70]
DIAMETERS = [1.0, 1.5, 2.0, 3.0, 4.0]
PER_IMAGE = 4        # 영상 한 장에 넣는 합성 이물 수
VARIANTS = 24        # test 영상 한 장당 만드는 합성 영상 수
BOX = 10             # 채점용 박스 한 변 (실제 라벨 중앙값)
SS = 8               # 세분화 배율


def transmission(d, c0, fx, fy):
    """이물 투과율 맵 (작은 패치)과 패치 왼쪽 위 좌표 오프셋. fx, fy = 중심의 소수부."""
    r = int(np.ceil(d / 2)) + 1
    n = (2 * r + 1) * SS
    g = (np.arange(n) + 0.5) / SS - r - 0.5
    xx, yy = np.meshgrid(g - (fx - 0.5), g - (fy - 0.5))
    rho = np.sqrt(xx ** 2 + yy ** 2) / (d / 2)
    t = np.sqrt(np.clip(1 - rho ** 2, 0, None))
    T = np.exp(np.log(1 - c0) * t)
    T = T.reshape(2 * r + 1, SS, 2 * r + 1, SS).mean((1, 3))
    return T, r


def insert(gray_f, cx, cy, d, c0):
    """gray_f(float32)에 (cx, cy) 중심 이물을 곱해 넣는다. 제자리 수정."""
    ix, iy = int(np.floor(cx)), int(np.floor(cy))
    T, r = transmission(d, c0, cx - ix, cy - iy)
    y0, x0 = iy - r, ix - r
    gray_f[y0:y0 + T.shape[0], x0:x0 + T.shape[1]] *= T


def site_features(gray, pm, band, dist, cx, cy):
    x, y = int(cx), int(cy)
    f = gray.astype(np.float32)
    hp = f - cv2.GaussianBlur(f, (0, 0), 3)
    y0, y1, x0, x1 = max(0, y - 7), y + 8, max(0, x - 7), x + 8
    return dict(bg_mean=float(f[y0:y1, x0:x1].mean()), texture=float(hp[y0:y1, x0:x1].std()),
                in_band=bool(band[y, x]), edge_dist=float(dist[y, x]))


def band_mask(gray, pm):
    """제품 안에서 한 번 더 Otsu → 어두운 띠 영역."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    v = blur[pm > 0]
    t, _ = cv2.threshold(v.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (blur < t) & (pm > 0)


def main():
    cfg = yaml.safe_load(open(ROOT / "configs" / "data.yaml", encoding="utf-8"))
    data = ROOT / cfg["out_dir"]
    out = data / "synth"
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(data / "manifest.csv")
    man = man[man["labeled"] & (man["split"] == "test")]
    grid = [(c, d) for c in CONTRASTS for d in DIAMETERS]

    rows, img_rows = [], []
    for r in tqdm(list(man.itertuples()), desc="합성"):
        g = np.asarray(Image.open(data / "clean/images" / f"{r.id}.png"))
        h, w = g.shape
        pm_full = cv2.threshold(cv2.GaussianBlur(g, (9, 9), 0), 0, 1,
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        dist = cv2.distanceTransform(pm_full, cv2.DIST_L2, 3)
        pm = product_mask(g)
        band = band_mask(g, pm)
        forbid = np.zeros_like(pm)
        for b in np.loadtxt(data / "clean/labels" / f"{r.id}.txt", ndmin=2):
            cx, cy, bw, bh = b[1] * w, b[2] * h, b[3] * w, b[4] * h
            forbid[max(0, int(cy - bh / 2 - 10)):int(cy + bh / 2 + 10),
                   max(0, int(cx - bw / 2 - 10)):int(cx + bw / 2 + 10)] = 1
        ys, xs = np.nonzero(pm & (forbid == 0))
        for v in range(VARIANTS):
            rng = np.random.default_rng([cfg["seed"], int(r.sha1[:8], 16), v])
            f = g.astype(np.float32)
            placed, labels = [], []
            for _ in range(PER_IMAGE):
                for _try in range(100):
                    k = rng.integers(len(xs))
                    cx, cy = xs[k] + rng.random(), ys[k] + rng.random()
                    if all(np.hypot(cx - px, cy - py) >= 24 for px, py in placed):
                        break
                c0, d = grid[rng.integers(len(grid))]
                feats = site_features(g, pm, band, dist, cx, cy)
                insert(f, cx, cy, d, c0)
                placed.append((cx, cy))
                labels.append((cx, cy))
                rows.append(dict(img=f"{r.id}__s{v:02d}", src=r.id, machine=r.machine, cx=cx, cy=cy,
                                 d=d, c0=c0, **feats))
            img = np.clip(f.round(), 0, 255).astype(np.uint8)
            sid = f"{r.id}__s{v:02d}"
            Image.fromarray(img).save(out / "images" / f"{sid}.png")
            np.savetxt(out / "labels" / f"{sid}.txt",
                       [(0, cx / w, cy / h, BOX / w, BOX / h) for cx, cy in labels], fmt="%d %.6f %.6f %.6f %.6f")
            # 넣은 뒤 실제 대비를 실제 이물과 같은 방식으로 잰다
            for (cx, cy), row in zip(labels, rows[-PER_IMAGE:]):
                m = measure(img, cx - 0.5, cy - 0.5)
                row.update(c_meas=m["contrast"], area_meas=m["area"])
            img_rows.append(dict(img=sid, src=r.id, machine=r.machine, w=w, h=h))

    pd.DataFrame(rows).to_csv(out / "defects.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(img_rows).to_csv(out / "images.csv", index=False, encoding="utf-8-sig")
    json.dump(dict(contrasts=CONTRASTS, diameters=DIAMETERS, per_image=PER_IMAGE, variants=VARIANTS,
                   box=BOX, n_images=len(img_rows), n_defects=len(rows)),
              open(out / "config.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(len(img_rows), "장,", len(rows), "개")


if __name__ == "__main__":
    main()
