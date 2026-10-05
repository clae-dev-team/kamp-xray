"""정답(TXT) 없는 2,032장으로 하는 사후 대조: 사람이 그려 둔 색 표시 자리를 모델이 찾았는가.

정답 파일이 있는 사진은 500장뿐이라 시험 이물이 139개에 그친다. 나머지 2,032장에도 사람이 그린 색 표시(3,424개)가 있다.
이 표시는 **학습에도, 기준선을 정하는 데에도 쓰지 않았다** (prepare.py 가 지우기만 했고, 위치는 data/marks.csv 에 기록).
그래서 표시를 지운 사진을 최종 모델로 추론한 뒤, 기록해 둔 표시 자리와 사후에 맞춰 볼 수 있다.

  찾음   : 표시 사각형(±2px) 안에 합격선 이상 예측의 중심이 있음
  표시 밖 : 어떤 표시에도 들지 않는 합격선 이상 예측 (오경보 후보. 사람이 표시하지 않은 이물일 수도 있다)
  가짜 표시: 전처리 때 이물 없는 곳에 그렸다 지운 자리(kind=fake)에 반응한 비율 → 지운 흔적에 반응하는지

읽을 때 주의
  - 표시는 TXT 정답이 아니다. 한 변 18~20px 사각형이라 이물이 그 안에 있다는 것만 알려 준다.
    그래서 정답 있는 500장에서 같은 방법(표시 기준)과 TXT 기준의 재현율을 나란히 내어 방법의 차이를 먼저 본다.
  - 정답 없는 사진은 학습 사진과 같은 호기 · 같은 날 찍힌 것이 많다. 완전히 독립된 시험이 아니다 (날짜가 겹치지 않는 사진만 따로 집계한다).

입력: 모든 사진의 예측 박스 CSV (id, x0, y0, x1, y1, score). --predict 를 주면 이 PC 에서 추론해 만든다.
실행: .venv\\Scripts\\python.exe src\\unlabeled_check.py [--boxes results/unlabeled_check/all_boxes.csv] [--predict]
결과: results/unlabeled_check/ (summary.json, marks_scored.csv, by_machine_date.csv)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

import metrics as M
from train_yolo import predict, weights_path

ROOT = Path(__file__).resolve().parents[1]
MARGIN = 2


def ci(k, n):
    """Clopper–Pearson 95% 구간."""
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return [round(lo, 4), round(hi, 4)]


def score_marks(marks, boxes):
    """표시마다 그 안에 중심이 든 예측의 최고 점수. boxes 에는 cx, cy 열이 있어야 한다."""
    by = dict(tuple(boxes.groupby("id")))
    best = np.zeros(len(marks))
    for k, r in enumerate(marks.itertuples()):
        b = by.get(r.id)
        if b is None:
            continue
        inside = (b.cx >= r.x0 - MARGIN) & (b.cx <= r.x1 + MARGIN) & (b.cy >= r.y0 - MARGIN) & (b.cy <= r.y1 + MARGIN)
        if inside.any():
            best[k] = b.score[inside].max()
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo", default="ratio3_e100")
    ap.add_argument("--boxes", default="results/unlabeled_check/all_boxes.csv")
    ap.add_argument("--predict", action="store_true", help="박스 CSV 를 이 PC 에서 새로 만든다 (GPU)")
    args = ap.parse_args()
    data = ROOT / "data"
    out = ROOT / "results" / "unlabeled_check"
    out.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(data / "manifest.csv")
    th = json.load(open(ROOT / "results/risk_threshold/summary.json", encoding="utf-8"))["채택"]
    t_low, t_high = th["합격선"], th["불합격선"]

    if args.predict:
        from ultralytics import YOLO
        ids = man["id"].tolist()
        predict(YOLO(str(weights_path(args.yolo))), [data / "clean/images" / f"{i}.png" for i in ids], ids, 640).to_csv(
            ROOT / args.boxes, index=False)
    boxes = pd.read_csv(ROOT / args.boxes)
    assert boxes["id"].nunique() <= len(man) and set(boxes["id"]) <= set(man["id"])
    boxes["cx"], boxes["cy"] = (boxes.x0 + boxes.x1) / 2, (boxes.y0 + boxes.y1) / 2
    marks = pd.read_csv(data / "marks.csv").merge(man[["id", "machine", "date", "labeled", "split"]], on="id")
    marks["score"] = score_marks(marks, boxes)
    marks["found"] = marks["score"] >= t_low

    def recall(m):
        k, n = int(m["found"].sum()), len(m)
        return {"표시수": n, "찾음": k, "재현율": round(k / max(n, 1), 4), "95%구간": ci(k, n) if n else None}

    # 색이 있는 화소 덩어리 가운데 '이물 표시 사각형'만 센다. 1호기 2020-07-27 사진 112장에는 세로로 긴 노란 선(폭 2px)이
    # 그려져 있는데 이물 표시가 아니다 (한 변 12~34px 의 사각형이 아닌 것은 대조에서 빼고 수만 남긴다).
    mw, mh = marks.x1 - marks.x0, marks.y1 - marks.y0
    marks["box_like"] = (mw.between(12, 34) & mh.between(12, 34)) | (marks.kind == "fake")
    not_box = marks[(marks.kind == "real") & ~marks.box_like]
    real = marks[(marks.kind == "real") & marks.box_like]
    marks.to_csv(out / "marks_scored.csv", index=False, encoding="utf-8-sig")
    un, lab = real[~real.labeled], real[real.labeled]
    train_dates = set(map(tuple, man[man.labeled & (man.split == "train")][["machine", "date"]].drop_duplicates().to_numpy()))
    un_new = un[[(m, d) not in train_dates for m, d in zip(un.machine, un.date)]]
    summary = {"모델": args.yolo, "합격선": t_low, "불합격선": t_high,
               "사각형이_아닌_색표시(대조_제외)": {"건수": len(not_box), "정답없는_사진": int((~not_box.labeled).sum()),
                                        "호기·날짜": {f"{int(m)}호기 {d}": int(n) for (m, d), n in not_box.groupby(["machine", "date"]).size().items()}}, "사진수": {"정답없음": int((~man.labeled).sum()), "정답있음": int(man.labeled.sum())},
               "정답없는_사진": {"전체": recall(un), "호기별": {int(k): recall(v) for k, v in un.groupby("machine")},
                            "학습과_날짜가_겹치지_않는_사진만": {**recall(un_new), "사진수": int(un_new["id"].nunique()),
                                                     "호기별": {int(k): recall(v) for k, v in un_new.groupby("machine")}}}}

    # 대조 방법 보정: 정답 있는 사진에서 표시 기준 vs TXT 기준 (분할별)
    lm = man[man.labeled].set_index("id")
    cal = {}
    for sp in ["train", "val", "test"]:
        ids = lm.index[lm.split == sp].tolist()
        gt = M.load_gt(ids, data / "clean/labels", {i: (lm.w[i], lm.h[i]) for i in ids})
        ev = M.evaluate(boxes[boxes["id"].isin(ids)][["id", "x0", "y0", "x1", "y1", "score"]], gt, t_low, "center")
        cal[sp] = {"TXT기준": {"이물수": ev["n_gt"], "찾음": ev["TP"], "재현율": ev["recall"]}, "표시기준": recall(lab[lab.split == sp])}
    summary["정답있는_사진(방법_보정)"] = cal

    # 표시 밖 예측과 가짜 표시 반응 (정답 없는 사진)
    hi = boxes[(boxes.score >= t_low) & boxes["id"].isin(man.loc[~man.labeled, "id"])].copy()
    rb = dict(tuple(real.groupby("id")))
    outside = []
    for r in hi.itertuples():
        m = rb.get(r.id)
        outside.append(m is None or not ((m.x0 - MARGIN <= r.cx) & (r.cx <= m.x1 + MARGIN) & (m.y0 - MARGIN <= r.cy) & (r.cy <= m.y1 + MARGIN)).any())
    hi["outside"] = outside
    n_un = int((~man.labeled).sum())
    fake = marks[(marks.kind == "fake") & ~marks.labeled]
    summary["표시_밖_예측(정답없는_사진)"] = {"건수": int(hi.outside.sum()), "사진당": round(float(hi.outside.sum()) / n_un, 4),
                                    "그런_사진수": int(hi[hi.outside]["id"].nunique()), "불합격선_이상": int((hi.outside & (hi.score >= t_high)).sum())}
    summary["가짜표시_자리_반응(정답없는_사진)"] = {"자리수": len(fake), "반응": int(fake["found"].sum()),
                                      "비율": round(float(fake["found"].mean()), 4) if len(fake) else None}
    hi[hi.outside].to_csv(out / "outside_boxes.csv", index=False, encoding="utf-8-sig")

    # 사진 단위 판정 (전부 불량 폴더)
    top = boxes.groupby("id")["score"].max().reindex(man["id"]).fillna(0)
    top.index = man.index
    v = np.where(top >= t_high, "불합격", np.where(top >= t_low, "재검사", "합격"))
    um = man[~man.labeled].assign(판정=v[~man.labeled.to_numpy()])
    has_mark = um["id"].isin(un["id"])
    summary["사진_판정(정답없는_사진)"] = {"표시가_있는_사진": um[has_mark]["판정"].value_counts().to_dict(),
                                 "표시가_없는_사진": um[~has_mark]["판정"].value_counts().to_dict()}

    # 놓친 표시의 호기 · 날짜 분포
    g = un.groupby(["machine", "date"]).agg(표시수=("found", "size"), 찾음=("found", "sum"), 사진수=("id", "nunique")).reset_index()
    g["재현율"] = (g["찾음"] / g["표시수"]).round(4)
    g["학습_날짜와_겹침"] = [(m, d) in train_dates for m, d in zip(g.machine, g.date)]
    g.to_csv(out / "by_machine_date.csv", index=False, encoding="utf-8-sig")
    summary["놓친_표시"] = {"건수": int((~un.found).sum()), "점수0(예측없음)": int((un.score == 0).sum()),
                        "점수_0.3~합격선": int(((un.score >= 0.3) & (un.score < t_low)).sum()),
                        "호기별": {int(k): int((~v_.found).sum()) for k, v_ in un.groupby("machine")}}
    json.dump(summary, open(out / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=int)
    print(json.dumps(summary, ensure_ascii=False, indent=1, default=int))


if __name__ == "__main__":
    main()
