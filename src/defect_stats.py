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
    """(cx, cy) 근처 이물의 대비·반치폭 면적·주변 밝기. 영상 가장자리는 잘라서 잰다."""
    f = gray.astype(np.float32)
    h, w = f.shape
    x0, x1 = max(0, int(cx) - r_out), min(w, int(cx) + r_out + 1)
    y0, y1 = max(0, int(cy) - r_out), min(h, int(cy) + r_out + 1)
    patch = f[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.maximum(np.abs(xx - cx), np.abs(yy - cy))
    bg = float(np.median(patch[(d >= r_in) & (d <= r_out)]))
    core = cv2.blur(patch, (2, 2))[d <= 3]
    lo = float(core.min())
    c = 1 - lo / max(bg, 1)
    # 반치폭 면적: 중심 3px 안의 최저점에서 연결된 '절반 이상 어두운' 픽셀
    half = (patch < bg - (bg - lo) / 2).astype(np.uint8)
    n, lab = cv2.connectedComponents(half, connectivity=8)
    near = lab[(d <= 3) & (half > 0)]
    area = int(np.isin(lab, np.unique(near[near > 0])).sum()) if len(near) else 0
    return dict(contrast=c, area=area, bg=bg)


def main():
    cfg = yaml.safe_load(open(ROOT / "configs" / "data.yaml", encoding="utf-8"))
    data = ROOT / cfg["out_dir"]
    man = pd.read_csv(data / "manifest.csv")
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
            m = measure(g, x0 + ix, y0 + iy)
            rows.append(dict(id=r.id, k=k, machine=r.machine, split=r.split, box_w=bw, box_h=bh, **m))
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "defect_stats"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "real_defects.csv", index=False, encoding="utf-8-sig")
    q = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95]
    print(df[["contrast", "area", "bg"]].quantile(q).round(3).to_string())
    print(df.groupby("machine")[["contrast", "area"]].median().round(3).to_string())


if __name__ == "__main__":
    main()
