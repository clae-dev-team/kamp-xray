"""기준 정상 영상: 호기별로 '가장 정상에 가까운' 가짜 정상 사진 N장을 고른다 (운영 중 점검의 기준 사진).

제조 현장의 골든 샘플·골든 이미지와 같은 역할이지만, 실물 양품으로 확인된 사진은 아니다(이물을 지운 가짜 정상).

정상 사진이 없어 val 가짜 정상(이물 점만 지운 사진, normal_set.py)에서 고른다. test 는 기준 결정에 쓰지 않는다.
  1. 라벨 누락 의심 사진(judge.py suspects.csv, 정상인데 불합격선 이상) 제외
  2. 지운 자리 대비(normal_set erase_check)가 실제 이물 최소 대비보다 낮은 사진만 (지운 흔적이 이물처럼 보이지 않게)
  3. 남은 사진 중 최종 모델 최고 점수가 낮은 순으로 N장
한 장으로는 정상의 흔들림을 알 수 없어 호기마다 여러 장을 고른다.

실행: .venv\\Scripts\\python.exe src\\reference_set.py --yolo ratio3_e100
결과: results/reference/ (reference.csv, thumbs.png)
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
N_PER = 5        # 호기마다 고르는 기준 정상 영상 수 (--n 기본값)


def main():
    """호기별 기준 정상 영상을 골라 목록과 썸네일을 저장한다.

    judge.py (scores_val.csv, suspects.csv), normal_set.py (erase_check.csv), defect_stats.py (real_defects.csv) 의 결과를 읽는다.
    reference.csv 열: img, src, machine, path, 최고점수(--yolo 모델의 영상 최고 점수), 지운자리_최대대비.
"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo", default="ratio3_e100")
    ap.add_argument("--n", type=int, default=N_PER)
    args = ap.parse_args()
    out = ROOT / "results" / "reference"
    out.mkdir(parents=True, exist_ok=True)

    # 후보 = val 가짜 정상 전체. 모델 이름(--yolo)과 같은 이름의 열에 그 모델의 영상 최고 점수가 들어 있다
    sc = pd.read_csv(ROOT / "results/judge/scores_val.csv")
    sc = sc[sc["kind"] == "normal"].copy()
    # 조건 1: 이 모델 기준으로 라벨 누락이 의심되는 사진 목록
    sus = pd.read_csv(ROOT / "results/judge/suspects.csv")
    bad = set(sus.loc[(sus["split"] == "val") & (sus["model"] == args.yolo), "img"])
    # 조건 2: 사진마다 지운 자리 대비의 최댓값 (이물이 여러 개면 가장 덜 지워진 자리 기준)
    er = pd.read_csv(ROOT / "results/normal_set/erase_check.csv")
    er = er[er["split"] == "val"].groupby("id")["after"].max()
    real_min = float(pd.read_csv(ROOT / "results/defect_stats/real_defects.csv")["contrast"].min())

    # erase_check 의 id 는 원본 영상 id 라 가짜 정상의 src 열로 잇는다
    sc["지운자리_최대대비"] = sc["src"].map(er)
    sc["의심"] = sc["img"].isin(bad)
    ok = sc[~sc["의심"] & (sc["지운자리_최대대비"] < real_min)]
    # 조건 3: 점수 오름차순으로 세운 뒤 호기별로 앞에서 n장. 저장할 때는 호기, 점수 순으로 다시 정렬한다
    ref = (ok.sort_values(args.yolo).groupby("machine", group_keys=False).head(args.n)
              .sort_values(["machine", args.yolo]))
    ref = ref[["img", "src", "machine", "path", args.yolo, "지운자리_최대대비"]].rename(columns={args.yolo: "최고점수"})
    ref.to_csv(out / "reference.csv", index=False, encoding="utf-8-sig")
    print(f"후보 {len(sc)}장 → 의심 제외 {int(sc['의심'].sum())}장, 지운 흔적 기준 통과 {len(ok)}장 → 기준 정상 영상 {len(ref)}장")
    print(ref.round(4).to_string(index=False))

    # 썸네일: 행 = 호기, 열 = 기준 정상 영상 (제품 부분만, 높이 맞춤)
    rows = []
    for m, g in ref.groupby("machine"):
        tiles = []
        for p in g["path"]:
            a = np.asarray(Image.open(p).convert("L"))
            # Otsu 로 제품(배경보다 어두운 쪽)을 잡아 그 범위에 8px 여유를 두고 자른 뒤, 높이 220px 로 비율을 지켜 줄인다
            mk = cv2.threshold(cv2.GaussianBlur(a, (9, 9), 0), 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            ys, xs = np.nonzero(mk)
            c = a[max(0, ys.min() - 8):ys.max() + 8, max(0, xs.min() - 8):xs.max() + 8]
            c = cv2.resize(c, (int(c.shape[1] * 220 / c.shape[0]), 220), interpolation=cv2.INTER_AREA)
            tiles += [c, np.full((220, 8), 255, np.uint8)]
        rows.append(np.hstack(tiles))
    # 행마다 폭이 다르므로 가장 넓은 행에 맞춰 오른쪽을 흰색으로 채우고, 행 사이에 10px 흰 띠를 둔다
    w = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 10), (0, w - r.shape[1])), constant_values=255) for r in rows]
    Image.fromarray(np.vstack(rows)).save(out / "thumbs.png")


if __name__ == "__main__":
    main()
