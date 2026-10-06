"""실제 이물의 대비·크기 측정. 합성 이물 실험과 같은 정의로 재서 둘을 한 축에 놓을 수 있게 한다.

대비 c = 1 - (이물 중심 어두운 값) / (주변 고리 중앙값)
         Beer–Lambert 기준으로 이물이 X선을 흡수해 줄인 투과율 비율과 같다 (c=0.3 → 30% 더 어두움).
         중심 값은 잡음 영향을 줄이려고 2×2 평균의 최솟값을 쓴다.
크기  = 중심 주변에서 (주변값 - 절반 대비)보다 어두운 연결 픽셀 수 (반치폭 면적)

실행: .venv\\Scripts\\python.exe src\\defect_stats.py
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def measure(gray, cx, cy, r_in=5, r_out=9):
    """(cx, cy) 근처 이물의 대비·반치폭 면적·주변 밝기. 영상 가장자리는 잘라서 잰다.

    cx, cy: 픽셀 한가운데가 정수인 좌표(px). r_in, r_out: 주변 고리의 안쪽·바깥쪽 반폭(px, 정사각형 고리).
    반환: dict(contrast 대비 0~1, area 반치폭 면적 픽셀 수, bg 주변 고리 밝기 중앙값 0~255).
    """
    f = gray.astype(np.float32)
    h, w = f.shape
    # 중심 둘레 (2·r_out+1) 정사각형 조각. yy, xx 는 영상 전체 기준 좌표다
    x0, x1 = max(0, int(cx) - r_out), min(w, int(cx) + r_out + 1)
    y0, y1 = max(0, int(cy) - r_out), min(h, int(cy) + r_out + 1)
    patch = f[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    # d = 가로·세로 거리 중 큰 쪽 (체비쇼프 거리). 같은 d 인 픽셀은 정사각형 테두리를 이룬다
    d = np.maximum(np.abs(xx - cx), np.abs(yy - cy))
    # 주변 밝기 = 중심에서 r_in~r_out 떨어진 고리의 중앙값 (이물 자신은 들어가지 않는다)
    bg = float(np.median(patch[(d >= r_in) & (d <= r_out)]))
    # 중심 값 = 중심 3px 안에서 2×2 평균이 가장 어두운 값
    core = cv2.blur(patch, (2, 2))[d <= 3]
    lo = float(core.min())
    # 대비 = 주변보다 줄어든 밝기의 비율. 주변이 0 이어도 나눌 수 있게 분모는 1 이상
    c = 1 - lo / max(bg, 1)
    # 반치폭 면적: 중심 3px 안의 최저점에서 연결된 '절반 이상 어두운' 픽셀
    half = (patch < bg - (bg - lo) / 2).astype(np.uint8)
    n, lab = cv2.connectedComponents(half, connectivity=8)
    # 중심 3px 안에 걸친 덩어리만 이물로 치고(여러 개면 모두), 그 덩어리 전체의 픽셀 수를 센다. 없으면 0
    near = lab[(d <= 3) & (half > 0)]
    area = int(np.isin(lab, np.unique(near[near > 0])).sum()) if len(near) else 0
    return dict(contrast=c, area=area, bg=bg)


def main():
    """라벨 있는 영상의 실제 이물을 모두 재서 results/defect_stats/real_defects.csv 로 저장하고 분위수를 출력한다.

    열: id, k(영상 안 박스 번호), machine, split, box_w, box_h(라벨 박스 크기 px), contrast, area, bg.
    """
    cfg = yaml.safe_load(open(ROOT / "configs" / "data.yaml", encoding="utf-8"))
    data = ROOT / cfg["out_dir"]
    man = pd.read_csv(data / "manifest.csv")
    # 분할과 상관없이 라벨 있는 영상 전부 (train · val · test)
    man = man[man["labeled"]]
    rows = []
    for r in man.itertuples():
        g = np.asarray(Image.open(data / "clean/images" / f"{r.id}.png"))
        for k, b in enumerate(np.loadtxt(data / "clean/labels" / f"{r.id}.txt", ndmin=2)):
            cx, cy = b[1] * r.w, b[2] * r.h
            # 라벨 박스 중심이 이물 점과 1~2px 어긋나 있어 박스 안 가장 어두운 곳을 중심으로 삼는다
            bw, bh = b[3] * r.w, b[4] * r.h
            x0, y0 = int(max(0, cx - bw / 2)), int(max(0, cy - bh / 2))
            sub = cv2.blur(g[y0:int(cy + bh / 2) + 1, x0:int(cx + bw / 2) + 1].astype(np.float32), (2, 2))
            iy, ix = np.unravel_index(np.argmin(sub), sub.shape)
            # 잘라 낸 조각 안의 위치(ix, iy)에 조각 시작점을 더해 영상 좌표로 되돌린다
            m = measure(g, x0 + ix, y0 + iy)
            rows.append(dict(id=r.id, k=k, machine=r.machine, split=r.split, box_w=bw, box_h=bh, **m))
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "defect_stats"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "real_defects.csv", index=False, encoding="utf-8-sig")
    # 화면 출력: 전체 분위수(1 · 5 · 25 · 50 · 75 · 95%)와 호기별 중앙값
    q = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95]
    print(df[["contrast", "area", "bg"]].quantile(q).round(3).to_string())
    print(df.groupby("machine")[["contrast", "area"]].median().round(3).to_string())


if __name__ == "__main__":
    main()
