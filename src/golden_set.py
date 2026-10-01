"""골든 세트: 호기별로 '가장 정상에 가까운' 가짜 정상 사진 N장을 고른다 (운영 중 점검의 기준 사진).

정상 사진이 없어 val 가짜 정상(이물 점만 지운 사진, normal_set.py)에서 고른다. test 는 기준 결정에 쓰지 않는다.
  1. 라벨 누락 의심 사진(judge.py suspects.csv, 정상인데 불합격선 이상) 제외
  2. 지운 자리 대비(normal_set erase_check)가 실제 이물 최소 대비보다 낮은 사진만 (지운 흔적이 이물처럼 보이지 않게)
  3. 남은 사진 중 최종 모델 최고 점수가 낮은 순으로 N장
한 장으로는 정상의 흔들림을 알 수 없어 호기마다 여러 장을 고른다.

실행: .venv\\Scripts\\python.exe src\\golden_set.py --yolo ratio3_e100
결과: results/golden/ (golden.csv, thumbs.png)
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
N_PER = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo", default="ratio3_e100")
    ap.add_argument("--n", type=int, default=N_PER)
    args = ap.parse_args()
    out = ROOT / "results" / "golden"
    out.mkdir(parents=True, exist_ok=True)

    sc = pd.read_csv(ROOT / "results/judge/scores_val.csv")
    sc = sc[sc["kind"] == "normal"].copy()
    sus = pd.read_csv(ROOT / "results/judge/suspects.csv")
    bad = set(sus.loc[(sus["split"] == "val") & (sus["model"] == args.yolo), "img"])
    er = pd.read_csv(ROOT / "results/normal_set/erase_check.csv")
    er = er[er["split"] == "val"].groupby("id")["after"].max()
    real_min = float(pd.read_csv(ROOT / "results/defect_stats/real_defects.csv")["contrast"].min())

    sc["지운자리_최대대비"] = sc["src"].map(er)
    sc["의심"] = sc["img"].isin(bad)
    ok = sc[~sc["의심"] & (sc["지운자리_최대대비"] < real_min)]
    gold = (ok.sort_values(args.yolo).groupby("machine", group_keys=False).head(args.n)
              .sort_values(["machine", args.yolo]))
    gold = gold[["img", "src", "machine", "path", args.yolo, "지운자리_최대대비"]].rename(columns={args.yolo: "최고점수"})
    gold.to_csv(out / "golden.csv", index=False, encoding="utf-8-sig")
    print(f"후보 {len(sc)}장 → 의심 제외 {int(sc['의심'].sum())}장, 지운 흔적 기준 통과 {len(ok)}장 → 골든 {len(gold)}장")
    print(gold.round(4).to_string(index=False))

    # 썸네일: 행 = 호기, 열 = 골든 사진 (제품 부분만, 높이 맞춤)
    rows = []
    for m, g in gold.groupby("machine"):
        tiles = []
        for p in g["path"]:
            a = np.asarray(Image.open(p).convert("L"))
            mk = cv2.threshold(cv2.GaussianBlur(a, (9, 9), 0), 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            ys, xs = np.nonzero(mk)
            c = a[max(0, ys.min() - 8):ys.max() + 8, max(0, xs.min() - 8):xs.max() + 8]
            c = cv2.resize(c, (int(c.shape[1] * 220 / c.shape[0]), 220), interpolation=cv2.INTER_AREA)
            tiles += [c, np.full((220, 8), 255, np.uint8)]
        rows.append(np.hstack(tiles))
    w = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 10), (0, w - r.shape[1])), constant_values=255) for r in rows]
    Image.fromarray(np.vstack(rows)).save(out / "thumbs.png")


if __name__ == "__main__":
    main()
