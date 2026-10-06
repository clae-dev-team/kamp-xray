"""공정 점검 신호: 검출 결과를 호기 · 날짜별로 모아 '이물의 성격이 평소와 달라진 날'을 표시한다.

검사기의 판정은 제품 하나하나에 대한 것이지만, 검출 결과를 쌓으면 공정 쪽에 알릴 신호가 된다.
이물이 평소와 다른 자리에 나오기 시작했거나, 한 제품에 여러 개씩 나오거나, 크기 · 진하기가 달라졌다면 원인이 바뀌었을 수 있다.

입력: 모든 사진(2,532장)의 예측 박스 (unlabeled_check.py 가 쓰는 all_boxes.csv). 합격선 이상 박스만 이물로 본다.
이물마다 제품 기준 정보(conditions.Img: 제품 마스크 · 어두운 띠 · 가장자리 거리)와 대비 · 크기(defect_stats.measure)를 붙인다.

날짜별 지표 (호기별로 따로)
  이물_사진당   : 이물이 검출된 사진 한 장당 이물 수
  띠밖_비율     : 이물 가운데 어두운 띠 밖에 있는 비율 (평소 자리는 띠 끝)
  대비_중앙     : 주변보다 어두운 정도의 중앙값
경보: 호기별로 처음 BASE_FRAC 의 날짜를 기준 구간으로 삼아, 띠밖_비율이 기준 비율의 3시그마 상한(p 관리도)을 넘는 날,
      이물_사진당이 3시그마 한계(u 관리도)를 위나 아래로 벗어난 날을 표시한다.
      자리 경보는 하루 이물이 MIN_N 개 미만인 날, 개수 경보는 사진이 MIN_PHOTOS 장 미만인 날에는 판단하지 않는다.

한계 (반드시 함께 읽을 것)
  - 받은 사진은 모두 불량으로 분류된 것이라 '불량률'은 계산할 수 없다. 여기서 보는 것은 불량의 성격 변화뿐이다.
  - 무엇이 원인인지(어느 공정인지)는 이 데이터로 알 수 없다. 표시된 날은 '확인해 볼 날'이지 원인 진단이 아니다.
  - 기준 구간을 데이터의 앞부분으로 잡은 것은 시연을 위한 가정이다.

실행: .venv\\Scripts\\python.exe src\\process_signal.py
결과: results/process_signal/ (daily.csv, defects.csv, summary.json, signal.png)
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from conditions import Img
from defect_stats import measure

ROOT = Path(__file__).resolve().parents[1]
BASE_FRAC = 0.3           # 기준 구간: 호기별로 날짜 순서 앞쪽 이 비율의 날짜 (최소 이틀)
MIN_N = 20                # 자리 경보: 하루 이물 수가 이보다 적으면 판단하지 않는다
MIN_PHOTOS = 8            # 개수 경보: 하루 사진 수가 이보다 적으면 판단하지 않는다


def main():
    """이물별 특징표(defects.csv)를 만들고 호기 · 날짜별로 모아 관리 한계와 경보를 붙인다 (daily.csv, summary.json, signal.png).

    defects.csv : id machine date labeled score in_product in_band edge(가장자리까지 px) u v(제품 외곽 상자 기준 자리) contrast area
    daily.csv   : machine date 이물수 이물사진수 띠밖 대비_중앙 크기_중앙 전체사진수 이물_사진당 띠밖_비율
                  기준구간 경보_자리 경보_개수 상한_띠밖 상한_사진당 하한_사진당
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--boxes", default="results/unlabeled_check/all_boxes.csv")
    args = ap.parse_args()
    data = ROOT / "data"
    out = ROOT / "results" / "process_signal"
    out.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(data / "manifest.csv").set_index("id")
    t_low = json.load(open(ROOT / "results/risk_threshold/summary.json", encoding="utf-8"))["채택"]["합격선"]
    boxes = pd.read_csv(ROOT / args.boxes)
    boxes = boxes[boxes.score >= t_low]

    # 이물마다 자리와 모양의 특징을 붙인다. 자리는 박스 중심 화소에서 읽는다
    rows = []
    for i, g in tqdm(boxes.groupby("id"), desc="이물 특징"):
        I = Img.get(data / "clean/images" / f"{i}.png")
        h, w = I["g"].shape
        ys, xs = np.nonzero(I["pm"])
        if len(xs) == 0:                                # 제품 영역을 찾지 못한 사진은 뺀다
            continue
        px0, px1, py0, py1 = xs.min(), xs.max(), ys.min(), ys.max()     # 제품 외곽 상자. u, v 는 이 상자 안에서의 0~1 자리다
        for b in g.itertuples():
            cx, cy = (b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2
            xi, yi = int(np.clip(cx, 0, w - 1)), int(np.clip(cy, 0, h - 1))
            m = measure(I["g"], cx - 0.5, cy - 0.5)
            rows.append(dict(id=i, machine=int(man.machine[i]), date=int(man.date[i]), labeled=bool(man.labeled[i]), score=b.score,
                             in_product=bool(I["pm"][yi, xi]), in_band=bool(I["band"][yi, xi]), edge=float(I["dist"][yi, xi]),
                             u=float((cx - px0) / max(px1 - px0, 1)), v=float((cy - py0) / max(py1 - py0, 1)),
                             contrast=m["contrast"], area=m["area"]))
    d = pd.DataFrame(rows)
    d.to_csv(out / "defects.csv", index=False, encoding="utf-8-sig")

    # 호기 · 날짜별 집계. 이물사진수 = 이물이 하나라도 검출된 사진 수, 전체사진수 = 그날 그 호기의 모든 사진 수
    daily = d.groupby(["machine", "date"]).agg(이물수=("id", "size"), 이물사진수=("id", "nunique"), 띠밖=("in_band", lambda s: int((~s).sum())),
                                              대비_중앙=("contrast", "median"), 크기_중앙=("area", "median")).reset_index()
    daily["전체사진수"] = [int(((man.machine == m) & (man.date == dt)).sum()) for m, dt in zip(daily.machine, daily.date)]
    daily["이물_사진당"] = daily["이물수"] / daily["이물사진수"]
    daily["띠밖_비율"] = daily["띠밖"] / daily["이물수"]

    summary = {"합격선": t_low, "이물수": len(d), "사진수": int(d["id"].nunique()), "기준구간_비율": BASE_FRAC, "하루_최소_이물수": MIN_N, "호기별": {}}
    flags = []
    for c in ("기준구간", "경보_자리", "경보_개수"):
        daily[c] = False
    daily["상한_띠밖"], daily["상한_사진당"], daily["하한_사진당"] = np.nan, np.nan, np.nan
    for mc, g in daily.groupby("machine"):
        g = g.sort_values("date")
        nb = max(2, int(round(len(g) * BASE_FRAC)))     # 기준 구간의 날짜 수
        base = g.iloc[:nb]
        p0 = (base["띠밖"].sum() + 0.5) / (base["이물수"].sum() + 1)          # 기준 구간에 0건이어도 상한이 0 이 되지 않게
        u0 = base["이물수"].sum() / base["이물사진수"].sum()                   # 기준 구간의 사진당 이물 수
        # p 관리도: 그날 이물 n 개 가운데 띠 밖 비율의 표준편차는 √(p0(1-p0)/n). 날마다 n 이 달라 한계도 날마다 다르다
        ucl_p = p0 + 3 * np.sqrt(p0 * (1 - p0) / g["이물수"])
        # u 관리도: 사진 n 장의 사진당 개수의 표준편차는 √(u0/n) (포아송 가정). 하한은 0 아래로 내려가지 않게 자른다
        ucl_u = u0 + 3 * np.sqrt(u0 / g["이물사진수"])
        ok = g["이물수"] >= MIN_N
        lcl_u = np.clip(u0 - 3 * np.sqrt(u0 / g["이물사진수"]), 0, None)
        ok_u = g["이물사진수"] >= MIN_PHOTOS
        # 자리 경보 = 띠 밖 비율이 상한을 넘은 날, 개수 경보 = 사진당 이물 수가 상한을 넘거나 하한 아래로 내려간 날
        f_p, f_u = ok & (g["띠밖_비율"] > ucl_p), ok_u & ((g["이물_사진당"] > ucl_u) | (g["이물_사진당"] < lcl_u))
        daily.loc[g.index, "기준구간"] = [k < nb for k in range(len(g))]
        daily.loc[g.index, "상한_띠밖"] = ucl_p.round(4)
        daily.loc[g.index, "상한_사진당"] = ucl_u.round(4)
        daily.loc[g.index, "하한_사진당"] = lcl_u.round(4)
        daily.loc[g.index, "경보_자리"] = f_p.to_numpy()
        daily.loc[g.index, "경보_개수"] = f_u.to_numpy()
        summary["호기별"][int(mc)] = {"날짜수": len(g), "기준_날짜수": nb, "기준_띠밖_비율": round(float(p0), 4), "기준_이물_사진당": round(float(u0), 3),
                                    "판단한_날짜수": int(ok.sum()), "경보_자리_날짜": g.loc[f_p, "date"].astype(int).tolist(),
                                    "경보_개수_날짜": g.loc[f_u, "date"].astype(int).tolist(),
                                    "경보_개수_첫날": int(g.loc[f_u, "date"].min()) if f_u.any() else None,
                                    "이물_사진당(경보 전/후)": [round(float(g.loc[ok_u & ~f_u, "이물_사진당"].mean()), 2),
                                                        round(float(g.loc[f_u, "이물_사진당"].mean()), 2) if f_u.any() else None],
                                    "기준구간_경보": int((f_p | f_u).iloc[:nb].sum()),
                                    "대비_중앙_범위": [round(float(g.loc[ok, "대비_중앙"].min()), 3), round(float(g.loc[ok, "대비_중앙"].max()), 3)]}
        flags.append(g)
    daily.round(4).to_csv(out / "daily.csv", index=False, encoding="utf-8-sig")
    json.dump(summary, open(out / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # 그림: 호기마다 한 줄. 왼쪽 = 사진당 이물 수, 오른쪽 = 띠 밖 이물 비율(%).
    # 회색 바탕 = 기준 구간, 점선 = 관리 한계, 초록 점 = 판단한 날, 주황 테두리 = 경보가 난 날
    plt.rcParams["font.family"] = "Malgun Gothic"
    fig, axes = plt.subplots(3, 2, figsize=(7.4, 5.4), dpi=170)
    for row, (mc, g) in zip(axes, daily.groupby("machine")):
        g = g.sort_values("date").reset_index(drop=True)
        x = np.arange(len(g))
        ok_p, ok_u = (g["이물수"] >= MIN_N).to_numpy(), (g["이물사진수"] >= MIN_PHOTOS).to_numpy()
        ticks = x[::max(1, len(x) // 7)]                # 가로축은 날짜 순번. 눈금 글자는 날짜(YYYYMMDD)에서 월/일만 떼어 쓴다
        for ax, col, lims, fcol, scale, name in ((row[0], "이물_사진당", ("하한_사진당", "상한_사진당"), "경보_개수", 1, "사진당 이물 수"),
                                                 (row[1], "띠밖_비율", ("상한_띠밖",), "경보_자리", 100, "띠 밖 이물 (%)")):
            ok = ok_u if col == "이물_사진당" else ok_p
            ax.axvspan(-0.5, g["기준구간"].sum() - 0.5, color="#eef1f4")
            ax.plot(x, g[col] * scale, color="#b9c2cb", lw=1)
            ax.scatter(x[ok], g.loc[ok, col] * scale, s=11, color="#0a8f86", zorder=3)
            for lim in lims:
                ax.step(x, g[lim] * scale, where="mid", color="#18212b", lw=0.8, ls="--")
            fl = g[fcol].astype(bool).to_numpy()
            ax.scatter(x[fl], g.loc[fl, col] * scale, s=44, facecolors="none", edgecolors="#c96f24", linewidths=1.5, zorder=4)
            ax.set_ylabel(f"{mc}호기 · {name}", fontsize=7.5)
            ax.set_xticks(ticks, [str(v)[4:6] + "/" + str(v)[6:] for v in g["date"].iloc[ticks]], fontsize=6.5)
            ax.tick_params(axis="y", labelsize=6.5)
            ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(out / "signal.png")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
