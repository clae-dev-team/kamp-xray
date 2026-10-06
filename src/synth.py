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
# 평가 격자: 명목 대비 11단계 × 지름 5단계 = 55칸. 이물마다 이 가운데 한 칸을 무작위로 고른다
CONTRASTS = [0.03, 0.05, 0.08, 0.12, 0.16, 0.20, 0.25, 0.30, 0.40, 0.55, 0.70]      # 명목 대비 c0 (0~1, 중심에서 줄어드는 투과율 비율)
DIAMETERS = [1.0, 1.5, 2.0, 3.0, 4.0]                                               # 지름 d (px)
PER_IMAGE = 4        # 영상 한 장에 넣는 합성 이물 수
VARIANTS = 24        # test 영상 한 장당 만드는 합성 영상 수
BOX = 10             # 채점용 박스 한 변 (실제 라벨 중앙값)
SS = 8               # 세분화 배율 (한 픽셀을 가로·세로 8칸씩 나눠 그린 뒤 평균한다)


def transmission(d, c0, fx, fy, aspect=1.0, angle=0.0):
    """이물 투과율 맵 (작은 패치)과 패치 반폭 r. fx, fy = 중심의 소수부.

    aspect > 1 이면 긴 지름 d·aspect, 짧은 지름 d 인 회전 타원체(길쭉한 파편), angle 은 라디안.

    d: 지름(px), c0: 명목 대비(0~1). fx, fy 는 0 이상 1 미만이고 0.5 가 픽셀 한가운데다.
    반환 T: (2r+1, 2r+1) 배열, 이물이 없는 곳은 1 · 중심 줄기에서 1-c0. 가운데 칸이 이물 중심이 든 픽셀이다.
    """
    # 패치 반폭 = 긴 반지름을 올림한 값 + 여유 1px
    r = int(np.ceil(d * aspect / 2)) + 1
    n = (2 * r + 1) * SS
    # 세분화 칸 중심의 좌표(px 단위). 원점은 가운데 픽셀의 한가운데
    g = (np.arange(n) + 0.5) / SS - r - 0.5
    # 이물 중심이 픽셀 한가운데에서 (fx-0.5, fy-0.5) 만큼 벗어난 것을 반영해 원점을 이물 중심으로 옮긴다
    xx, yy = np.meshgrid(g - (fx - 0.5), g - (fy - 0.5))
    # angle 만큼 돌린 좌표: u = 긴 축 방향, v = 짧은 축 방향
    u = xx * np.cos(angle) + yy * np.sin(angle)
    v = -xx * np.sin(angle) + yy * np.cos(angle)
    # rho = 각 축의 반지름으로 나눈 거리 (타원 둘레에서 1)
    rho = np.sqrt((u / (d * aspect / 2)) ** 2 + (v / (d / 2)) ** 2)
    # 구(회전 타원체)를 위에서 본 두께. 중심 1, 둘레 0, 바깥 0 으로 정규화한 값
    t = np.sqrt(np.clip(1 - rho ** 2, 0, None))
    # T = exp(-μ·두께) 에서 중심(t=1)의 투과율이 1-c0 이 되게 잡으면 T = (1-c0)^t
    T = np.exp(np.log(1 - c0) * t)
    # 세분화한 SS×SS 칸을 평균해 픽셀 격자로 줄인다 (부분 체적 효과)
    T = T.reshape(2 * r + 1, SS, 2 * r + 1, SS).mean((1, 3))
    return T, r


def insert(gray_f, cx, cy, d, c0, aspect=1.0, angle=0.0, residual=None, rng=None):
    """gray_f(float32)에 (cx, cy) 중심 이물을 곱해 넣는다. 제자리 수정. 영상 밖으로 나가는 부분은 자른다.

    residual 을 주면 X선 잡음 보정을 한다. 광자 잡음은 밝기의 제곱근에 비례하는데, 곱셈 합성은 잡음까지
    T배로 줄여 실제(√T배)보다 매끈해진다. 모자란 분산 σ²·T(1-T) 만큼, 6~12px 떨어진 곳의 잡음(residual)을
    √(T(1-T)) 배로 옮겨 더한다. 이웃 픽셀 간 상관(결)도 그대로 따라온다.

    cx, cy: 픽셀 경계 기준 좌표(px). 픽셀 k 는 k 이상 k+1 미만을 차지하고 한가운데가 k+0.5 다.
    d, c0, aspect, angle 은 transmission 과 같다. residual: noise_residual 결과 (H, W), rng: 잡음을 가져올 자리를 뽑는 난수 발생기.
    반환값은 없다.
    """
    ix, iy = int(np.floor(cx)), int(np.floor(cy))
    T, r = transmission(d, c0, cx - ix, cy - iy, aspect, angle)
    h, w = gray_f.shape
    # 패치 왼쪽 위가 놓일 영상 좌표 (음수면 영상 밖)
    y0, x0 = iy - r, ix - r
    # 영상 밖으로 나간 만큼 패치 쪽 범위(ty, tx)를 잘라 낸다
    ty0, tx0 = max(0, -y0), max(0, -x0)
    ty1, tx1 = T.shape[0] - max(0, y0 + T.shape[0] - h), T.shape[1] - max(0, x0 + T.shape[1] - w)
    Tc = T[ty0:ty1, tx0:tx1]
    ys, xs = slice(y0 + ty0, y0 + ty1), slice(x0 + tx0, x0 + tx1)
    gray_f[ys, xs] *= Tc
    if residual is not None:
        ph, pw = Tc.shape
        # 가로·세로 각각 6~12px 떨어진 같은 크기의 자리에서 잡음을 가져온다. 영상 안에 드는 자리를 50번까지 찾고, 못 찾으면 보정하지 않는다
        for _ in range(50):
            dy, dx = rng.integers(6, 13, 2) * rng.choice([-1, 1], 2)
            sy, sx = ys.start + dy, xs.start + dx
            if 0 <= sy and sy + ph <= h and 0 <= sx and sx + pw <= w:
                gray_f[ys, xs] += residual[sy:sy + ph, sx:sx + pw] * np.sqrt(Tc * (1 - Tc))
                break


def noise_residual(gray):
    """잡음 성분 = 영상 - 가우시안(σ=2) 흐림. 이물 크기(수 px)보다 가는 결을 담는다.

    gray: (H, W) uint8. 반환: (H, W) float32, 평균이 0 근처인 밝기 차.
    """
    f = gray.astype(np.float32)
    return f - cv2.GaussianBlur(f, (0, 0), 2)


def site_features(gray, pm, band, dist, cx, cy):
    """이물을 넣을 자리의 조건 (미탐지 조건 분석에서 검출률을 나눠 보는 축).

    gray: 이물을 넣기 전 영상, band: 띠 마스크, dist: 제품 가장자리까지 거리 지도(px), (cx, cy): 이물 중심. pm 은 쓰지 않는다.
    반환: dict(bg_mean 둘레 15×15 평균 밝기, texture 같은 범위의 고주파 성분 표준편차, in_band 띠 안 여부, edge_dist 제품 가장자리까지 거리 px).
    """
    x, y = int(cx), int(cy)
    f = gray.astype(np.float32)
    # 고주파 성분 = 영상 - 가우시안(σ=3) 흐림. 완만한 밝기 변화를 빼고 잔결만 남긴다
    hp = f - cv2.GaussianBlur(f, (0, 0), 3)
    y0, y1, x0, x1 = max(0, y - 7), y + 8, max(0, x - 7), x + 8
    return dict(bg_mean=float(f[y0:y1, x0:x1].mean()), texture=float(hp[y0:y1, x0:x1].std()),
                in_band=bool(band[y, x]), edge_dist=float(dist[y, x]))


def band_mask(gray, pm):
    """제품 안에서 한 번 더 Otsu → 어두운 띠 영역.

    gray: (H, W) uint8, pm: 제품 마스크(product_mask). 반환: (H, W) bool, 띠 안 True.
    """
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # 배경을 뺀 제품 픽셀만 모아 임계값을 구하고, 그보다 어두운 제품 픽셀을 띠로 본다
    v = blur[pm > 0]
    t, _ = cv2.threshold(v.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (blur < t) & (pm > 0)


def main():
    """test 영상마다 합성 영상 VARIANTS 장을 만들고 영상·라벨·이물 목록을 저장한다.

    결과 (data/synth 또는 --noise 일 때 data/synth_n)
      images/<id>__sNN.png, labels/<id>__sNN.txt (합성 이물만, 한 변 BOX px 의 박스)
      defects.csv  이물별 img, src, machine, cx, cy, d, c0, 자리 조건(site_features), c_meas(넣은 뒤 잰 대비), area_meas(반치폭 면적 px)
      images.csv   영상별 img, src, machine, w, h
      config.json  격자와 장수 설정
    """
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", action="store_true", help="X선 잡음 보정 (결과: data/synth_n). 이물 위치·조건은 보정 없는 판과 같다")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "configs" / "data.yaml", encoding="utf-8"))
    data = ROOT / cfg["out_dir"]
    out = data / ("synth_n" if args.noise else "synth")
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(data / "manifest.csv")
    man = man[man["labeled"] & (man["split"] == "test")]
    grid = [(c, d) for c in CONTRASTS for d in DIAMETERS]

    rows, img_rows = [], []
    for r in tqdm(list(man.itertuples()), desc="합성"):
        g = np.asarray(Image.open(data / "clean/images" / f"{r.id}.png"))
        h, w = g.shape
        # 침식하지 않은 제품 마스크로 '제품 가장자리까지 거리'를 잰다 (product_mask 는 7px 안쪽으로 깎여 있다)
        pm_full = cv2.threshold(cv2.GaussianBlur(g, (9, 9), 0), 0, 1,
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        dist = cv2.distanceTransform(pm_full, cv2.DIST_L2, 3)
        pm = product_mask(g)
        band = band_mask(g, pm)
        # 실제 이물 박스와 그 둘레 10px 에는 합성 이물을 놓지 않는다
        forbid = np.zeros_like(pm)
        for b in np.loadtxt(data / "clean/labels" / f"{r.id}.txt", ndmin=2):
            cx, cy, bw, bh = b[1] * w, b[2] * h, b[3] * w, b[4] * h
            forbid[max(0, int(cy - bh / 2 - 10)):int(cy + bh / 2 + 10),
                   max(0, int(cx - bw / 2 - 10)):int(cx + bw / 2 + 10)] = 1
        # 놓을 수 있는 자리 = 제품 안이면서 금지 영역 밖인 픽셀
        ys, xs = np.nonzero(pm & (forbid == 0))
        for v in range(VARIANTS):
            # 난수는 (시드, 영상 해시, 변형 번호)로 고정해 다시 돌려도 같은 자리·조건이 나온다
            rng = np.random.default_rng([cfg["seed"], int(r.sha1[:8], 16), v])
            f = g.astype(np.float32)
            res = noise_residual(g) if args.noise else None
            # 잡음 보정용 난수를 따로 둬서, 보정을 켜도 이물 위치·조건을 뽑는 난수 순서가 바뀌지 않게 한다
            nrng = np.random.default_rng([cfg["seed"], int(r.sha1[:8], 16), v, 99])
            placed, labels = [], []
            for _ in range(PER_IMAGE):
                # 먼저 넣은 합성 이물과 중심이 24px 이상 떨어진 자리를 100번까지 뽑는다 (끝내 못 찾으면 마지막에 뽑은 자리를 쓴다)
                for _try in range(100):
                    k = rng.integers(len(xs))
                    cx, cy = xs[k] + rng.random(), ys[k] + rng.random()
                    if all(np.hypot(cx - px, cy - py) >= 24 for px, py in placed):
                        break
                c0, d = grid[rng.integers(len(grid))]
                # 자리 조건은 이물을 넣기 전의 원래 영상(g)에서 잰다
                feats = site_features(g, pm, band, dist, cx, cy)
                insert(f, cx, cy, d, c0, residual=res, rng=nrng)
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
            # measure 는 픽셀 한가운데가 정수인 좌표를 쓰므로 0.5 를 뺀다. rows 의 마지막 PER_IMAGE 개가 이 영상에 넣은 이물이다
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
