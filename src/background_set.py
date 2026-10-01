"""학습용 '이물 없음' 배경 사진: train 분할 실제 불량 350장에서 이물 점만 지운 가짜 정상.

정상 사진에 가까운 영상을 '이물 없음'(빈 라벨)으로 함께 보여 주면, 띠 모서리·잡음 같은 정상 부위에
높은 점수를 주는 버릇(정상 제품 재검사)이 줄어드는지 본다. 만드는 법은 normal_set.py 와 같다
(val·test 가짜 정상과 같은 방법, 같은 사진이 아니라 train 분할만 쓰므로 판정 기준을 정하는 val·test 와 겹치지 않는다).

한계: 사람이 표시하지 않은 옅은 이물이 남아 있으면 그것도 '이물 없음'으로 배운다 → 옅은 이물 감도가
떨어질 수 있어 합성 평가(synth_eval)와 테스트피스로 함께 확인한다.

결과: data/bg/{images,labels} (빈 라벨), data/aug_bg/train.txt (= data/aug/train.txt + 배경 350장), data/aug_bg.yaml
실행: .venv\\Scripts\\python.exe src\\background_set.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm

from location_test import dot_mask
from normal_set import SEED_OFFSET, dot_centers
from prepare import restore

ROOT = Path(__file__).resolve().parents[1]


def main():
    cfg = yaml.safe_load(open(ROOT / "configs" / "data.yaml", encoding="utf-8"))
    data = ROOT / cfg["out_dir"]
    man = pd.read_csv(data / "manifest.csv")
    sub = man[man["labeled"] & (man["split"] == "train")]
    (data / "bg" / "images").mkdir(parents=True, exist_ok=True)
    (data / "bg" / "labels").mkdir(parents=True, exist_ok=True)
    paths = []
    for r in tqdm(list(sub.itertuples()), desc="학습용 가짜 정상"):
        g = np.asarray(Image.open(data / "clean/images" / f"{r.id}.png"))
        h, w = g.shape
        mask = np.zeros_like(g)
        for cx, cy in dot_centers(g, np.loadtxt(data / "clean/labels" / f"{r.id}.txt", ndmin=2), w, h):
            mask |= dot_mask(g, cx - 0.5, cy - 0.5)
        rng = np.random.default_rng([cfg["seed"] + SEED_OFFSET, int(r.sha1[:8], 16), 7])
        p = data / "bg" / "images" / f"{r.id}__bg.png"
        Image.fromarray(restore(g, mask, rng)).save(p)
        (data / "bg" / "labels" / f"{r.id}__bg.txt").write_text("", encoding="utf-8")
        paths.append(str(p))
    base = [l.strip() for l in open(data / "aug" / "train.txt", encoding="utf-8") if l.strip()]
    (data / "aug_bg").mkdir(exist_ok=True)
    (data / "aug_bg" / "train.txt").write_text("\n".join(base + paths) + "\n", encoding="utf-8")
    ds = yaml.safe_load(open(data / "aug.yaml", encoding="utf-8"))
    ds["train"] = str((data / "aug_bg" / "train.txt").resolve())
    yaml.safe_dump(ds, open(data / "aug_bg.yaml", "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
    print("배경", len(paths), "장, 학습 목록", len(base) + len(paths), "장")


if __name__ == "__main__":
    main()
