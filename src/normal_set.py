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
VARIANTS = 4                 # 가짜 정상 한 장당 만드는 합성 불량 영상 수
C0_RANGE = (0.20, 0.70)      # 눈에 보이는 범위 (실제 이물 측정 대비 5% 지점 0.27 근처까지 포함)
D_RANGE = (1.5, 4.0)         # 지름 px
SHARD_P = 0.30               # 길쭉한 파편 비율
SEED_OFFSET = 104729         # 학습 증강·합성 평가셋과 다른 난수


def dot_centers(g, boxes, w, h):
    """라벨 박스 안 2×2 평균 최솟값 = 이물 점 중심.

    g: (H, W) uint8, boxes: (N, 5) YOLO 형식, w · h: 영상 크기(px).
    반환: [(cx, cy)] 픽셀 좌표. 찾은 픽셀의 한가운데(k+0.5)를 주므로, 한가운데가 정수인 좌표를 받는
    measure · dot_mask 에는 0.5 를 빼서 넘긴다.
"""
    out = []
    for b in boxes:
        bw, bh = b[3] * w, b[4] * h
        # 박스의 왼쪽 위 픽셀. 라벨 중심이 이물 점과 조금 어긋나 있어 박스 안에서 가장 어두운 곳을 다시 찾는다 (defect_stats.py 와 같은 방법)
        x0, y0 = int(max(0, b[1] * w - bw / 2)), int(max(0, b[2] * h - bh / 2))
        # 한 픽셀짜리 잡음에 끌리지 않게 2×2 평균을 낸 뒤 최솟값 자리를 찾는다
        sub = cv2.blur(g[y0:int(b[2] * h + bh / 2) + 1, x0:int(b[1] * w + bw / 2) + 1].astype(np.float32), (2, 2))
        iy, ix = np.unravel_index(np.argmin(sub), sub.shape)
        out.append((x0 + ix + 0.5, y0 + iy + 0.5))
    return out


def main():
    """val · test 영상마다 가짜 정상 1장과 합성 불량 VARIANTS 장을 만들고 판정용 목록을 쓴다.

    data/judge_<split>.csv: split, img, src, machine, kind, path, 합성 불량은 cx, cy(px), c0, d(px), shard, in_band, c_meas(넣은 뒤 잰 대비).
    results/normal_set: erase_check.csv (이물별 지우기 전후 대비), check.json (요약).
    results/defect_stats/real_defects.csv 를 읽으므로 defect_stats.py 를 먼저 돌려야 한다.
"""
    cfg = yaml.safe_load(open(ROOT / "configs" / "data.yaml", encoding="utf-8"))
    data = ROOT / cfg["out_dir"]
    man = pd.read_csv(data / "manifest.csv")
    # rows = 판정용 영상 목록, checks = 지운 자리의 전후 대비
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
            # 가짜 정상: 이물 점의 반치폭 영역(+1px)만 모아, 전처리의 표시 제거와 같은 방법(restore)으로 메운다
            mask = np.zeros_like(g)
            for cx, cy in centers:
                mask |= dot_mask(g, cx - 0.5, cy - 0.5)
            rng = np.random.default_rng([cfg["seed"] + SEED_OFFSET, int(r.sha1[:8], 16)])
            normal = restore(g, mask, rng)
            npath = data / "normal" / split / f"{r.id}.png"
            Image.fromarray(normal).save(npath)
            # 지우기 확인: 같은 자리의 대비를 지우기 전과 뒤에 같은 방법으로 잰다
            for cx, cy in centers:
                before = measure(g, cx - 0.5, cy - 0.5)["contrast"]
                after = measure(normal, cx - 0.5, cy - 0.5)["contrast"]
                checks.append(dict(split=split, id=r.id, before=before, after=after))
            # 실제 불량(정제본 그대로)과 가짜 정상을 한 짝으로 목록에 올린다. src 가 같으면 같은 제품이다
            rows.append(dict(split=split, img=r.id, src=r.id, machine=r.machine, kind="real_ng",
                             path=str(data / "clean/images" / f"{r.id}.png")))
            rows.append(dict(split=split, img=f"{r.id}__normal", src=r.id, machine=r.machine, kind="normal",
                             path=str(npath)))

            # 합성 불량: 가짜 정상 위 제품 안 아무 곳(띠 안 절반)에 이물 1개
            pm = product_mask(normal)
            band = band_mask(normal, pm)
            pools = {"band": np.argwhere(band), "any": np.argwhere(pm > 0)}
            for k in range(VARIANTS):
                # 변형마다 난수를 따로 둔다 (지우기에 쓴 난수와 겹치지 않게 k+1 을 덧붙인다)
                rk = np.random.default_rng([cfg["seed"] + SEED_OFFSET, int(r.sha1[:8], 16), k + 1])
                pool = pools["band"] if (rk.random() < 0.5 and len(pools["band"])) else pools["any"]
                y, x = pool[rk.integers(len(pool))]
                cx, cy = x + rk.random(), y + rk.random()
                c0, d = rk.uniform(*C0_RANGE), rk.uniform(*D_RANGE)
                shard = rk.random() < SHARD_P
                aspect = rk.uniform(2, 4) if shard else 1.0
                # 변형마다 가짜 정상에서 새로 시작하므로 한 영상에는 합성 이물이 하나만 들어간다
                f = normal.astype(np.float32)
                insert(f, cx, cy, d, c0, aspect, rk.uniform(0, np.pi))
                img = np.clip(f.round(), 0, 255).astype(np.uint8)
                spath = data / "synth_ng" / split / f"{r.id}__n{k}.png"
                Image.fromarray(img).save(spath)
                rows.append(dict(split=split, img=f"{r.id}__n{k}", src=r.id, machine=r.machine, kind="synth_ng",
                                 path=str(spath), cx=cx, cy=cy, c0=c0, d=d, shard=shard,
                                 in_band=bool(band[int(cy), int(cx)]),
                                 c_meas=measure(img, cx - 0.5, cy - 0.5)["contrast"]))

    # 분할별 판정 목록 저장
    df = pd.DataFrame(rows)
    for split in ["val", "test"]:
        df[df["split"] == split].to_csv(data / f"judge_{split}.csv", index=False, encoding="utf-8-sig")
    ck = pd.DataFrame(checks)
    out = ROOT / "results" / "normal_set"
    out.mkdir(parents=True, exist_ok=True)
    # 지운 뒤 대비를 실제 이물의 최소 대비와 나란히 적어, 지운 자리가 이물 수준으로 남지 않았는지 볼 수 있게 한다
    real = pd.read_csv(ROOT / "results/defect_stats/real_defects.csv")
    summary = {
        "지운_이물수": len(ck),
        "지우기전_대비_중앙": round(float(ck["before"].median()), 3),
        "지운뒤_대비_중앙": round(float(ck["after"].median()), 3),
        "지운뒤_대비_최대": round(float(ck["after"].max()), 3),
        "실제이물_최소대비": round(float(real["contrast"].min()), 3),
        # 'if False' 라 앞쪽 식은 실행되지 않고 뒤쪽 dict 가 쓰인다: {"분할/종류": 영상 수}
        "영상수": df.groupby(["split", "kind"]).size().rename(lambda t: "/".join(t)).to_dict()
                   if False else {f"{s}/{k}": int(n) for (s, k), n in df.groupby(["split", "kind"]).size().items()},
    }
    ck.to_csv(out / "erase_check.csv", index=False, encoding="utf-8-sig")
    json.dump(summary, open(out / "check.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
