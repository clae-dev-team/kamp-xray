"""놓침 · 헛경보가 생기는 조건 정리 (최종 모델, 판정 합격선 기준).

판정 합격선(risk_threshold 채택값)은 '사진의 최고 점수가 이 아래면 합격'이므로,
이물 하나의 점수가 합격선 아래면 그 이물만 든 사진은 합격 = 놓침, 정상 부위 예측이 합격선 이상이면 = 헛경보로 본다.

1. 놓침 조건 (가상 시험편 14,400개, test, results/testpiece/defects_scored.csv 의 YOLO 점수)
   이물마다 호기 · 지름 · 측정 대비 · 제품 가장자리까지 거리 · 어두운 띠 안/밖 · 주변 밝기 · 주변 결(잔무늬) 을 붙이고
   (a) 조건별 놓침률 표, (b) 표준화 로지스틱 회귀(대비를 함께 넣어 '대비가 같을 때' 각 조건의 영향),
   (c) 깊이 3 결정나무로 사람이 읽는 규칙을 만든다.
   사양 안 시험편(val 로 정한 호기·지름별 90% 보장 진하기 이상) 중 놓친 것은 따로 모은다.
2. 헛경보 조건 (froc.py 가 저장한 예측: 시험 가짜 정상 73장 · 실제 불량 사진 73장 · 시험편 3,600장)
   정답(실제 이물 · 넣은 시험편)에 맞지 않은 예측 중 점수 ≥ 합격선 을 헛경보로 보고, 0.3 이상은 '헛경보 후보' 로 넓혀 본다.
   예측 자리마다 지운 자리(가짜 정상의 원래 이물 자리)까지 거리 · 가장자리 거리 · 띠 · 그 자리의 측정 대비 를 붙이고,
   같은 사진에서 무작위로 뽑은 제품 안 지점과 비율을 비교한다. 점수 높은 순 확대 그림을 만들어 눈으로 확인한다.
   주의: 시험편 사진 3,600장은 가짜 정상 73장을 약 49번씩 배경으로 다시 쓴 것이라, 배경의 같은 자리가 사본마다
   헛경보로 반복해 잡힌다. 그래서 (원본 사진, 6px 격자 자리)로 묶은 '고유 자리' 기준 통계를 함께 낸다.
3. 실제 이물 139개 중 점수가 낮은 것들의 조건.

실행: .venv\\Scripts\\python.exe src\\conditions.py   (froc.py 먼저)
결과: results/conditions/ (summary.json, miss_*.csv, fp_*.csv, fp_top.png, miss_insepc.png)
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text

import metrics as M
from defect_stats import measure
from judge import in_spec
from prepare import product_mask
from synth import band_mask
from synth_eval import MARGIN
from testpiece import HALF

ROOT = Path(__file__).resolve().parents[1]
FP_WIDE = 0.3
ERASE_NEAR = 6       # 지운 자리 근처 (px)
EDGE_NEAR = 4        # 제품 가장자리 근처 (px)
RNG_PTS = 200        # 사진마다 무작위 비교 지점


class Img:
    """사진 한 장의 제품 마스크 · 가장자리 거리 · 띠 · 결 지도 (캐시)."""
    cache = {}

    @classmethod
    def get(cls, path):
        if path not in cls.cache:
            g = np.asarray(Image.open(path).convert("L"))
            full = cv2.threshold(cv2.GaussianBlur(g, (9, 9), 0), 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            pm = product_mask(g)
            f = g.astype(np.float32)
            hp = f - cv2.GaussianBlur(f, (0, 0), 3)
            tex = np.sqrt(cv2.blur(hp * hp, (15, 15)))
            bg = cv2.blur(f, (15, 15))
            cls.cache[path] = dict(g=g, pm=pm, dist=cv2.distanceTransform(full, cv2.DIST_L2, 3),
                                   band=band_mask(g, pm), tex=tex, bg=bg)
        return cls.cache[path]


def feats(I, x, y):
    h, w = I["g"].shape
    xi, yi = int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))
    return dict(edge_dist=float(I["dist"][yi, xi]), in_band=bool(I["band"][yi, xi]), in_product=bool(I["pm"][yi, xi]),
                bg=float(I["bg"][yi, xi]), texture=float(I["tex"][yi, xi]))


def rate_table(df, col, bins=None, labels=None):
    k = pd.cut(df[col], bins, labels=labels, include_lowest=True) if bins is not None else df[col]
    t = df.groupby(k, observed=True)["miss"].agg(["mean", "size"])
    return {str(i): {"놓침률": round(float(r["mean"]), 3), "수": int(r["size"])} for i, r in t.iterrows()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo", default="ratio3_e100")
    args = ap.parse_args()
    data = ROOT / "data"
    out = ROOT / "results" / "conditions"
    out.mkdir(parents=True, exist_ok=True)
    thr = json.load(open(ROOT / "results/risk_threshold/summary.json", encoding="utf-8"))["채택"]["합격선"]
    man = pd.read_csv(data / "manifest.csv").set_index("id")
    summary = {"합격선": thr}

    # ---------- 1. 놓침 조건 (시험편) ----------
    d = pd.read_csv(ROOT / "results/testpiece/defects_scored.csv")
    for i, r in enumerate(d.itertuples()):
        f = feats(Img.get(data / "normal/test" / f"{r.src}.png"), r.cx, r.cy)
        d.loc[i, ["bg", "texture"]] = f["bg"], f["texture"]
    d["miss"] = (d["YOLO_score"] < thr).astype(int)
    spec = pd.read_csv(ROOT / f"results/testpiece_val_{args.yolo}/spec.csv")
    spec = spec[spec["model"] == "YOLO"]
    d["사양내"] = in_spec(d.assign(kind="synth_ng"), spec)
    d.to_csv(out / "miss_testpiece.csv", index=False, encoding="utf-8-sig")
    tex_q = np.quantile(d["texture"], [0, 1 / 3, 2 / 3, 1])
    bg_q = np.quantile(d["bg"], [0, 1 / 3, 2 / 3, 1])
    mt = {"전체": {"놓침률": round(float(d.miss.mean()), 3), "수": len(d)},
          "호기": rate_table(d, "machine"), "지름": rate_table(d, "d"),
          "측정대비": rate_table(d, "c_meas", [-1, 0.05, 0.10, 0.15, 0.20, 0.30, 1]),
          "가장자리거리": rate_table(d, "edge_dist", [0, 4, 8, 15, 1e9], ["<4px", "4-8px", "8-15px", "15px+"]),
          "띠": rate_table(d.assign(띠=np.where(d.in_band, "띠 안", "띠 밖")), "띠"),
          "주변밝기": rate_table(d, "bg", bg_q, ["어두움", "중간", "밝음"]),
          "주변결": rate_table(d, "texture", tex_q, ["매끈", "중간", "거침"])}
    # 대비가 같은 조건에서 비교: 측정 대비 0.10~0.20 구간만
    mid = d[(d.c_meas >= 0.10) & (d.c_meas < 0.20)]
    mt["대비0.10~0.20_안에서"] = {"가장자리거리": rate_table(mid, "edge_dist", [0, 4, 8, 15, 1e9], ["<4px", "4-8px", "8-15px", "15px+"]),
                             "띠": rate_table(mid.assign(띠=np.where(mid.in_band, "띠 안", "띠 밖")), "띠"),
                             "주변결": rate_table(mid, "texture", tex_q, ["매끈", "중간", "거침"]),
                             "호기": rate_table(mid, "machine")}
    X = pd.DataFrame({"측정대비": d.c_meas, "지름": d.d, "가장자리거리(log)": np.log1p(d.edge_dist),
                      "띠안": d.in_band.astype(float), "주변밝기": d.bg, "주변결": d.texture,
                      "2호기": (d.machine == 2).astype(float), "3호기": (d.machine == 3).astype(float)})
    Z = (X - X.mean()) / X.std()
    lr = LogisticRegression(C=10, max_iter=2000).fit(Z, d.miss)
    mt["로지스틱_오즈비(1표준편차당, >1이면 놓침 증가)"] = {c: round(float(np.exp(b)), 3) for c, b in zip(Z.columns, lr.coef_[0])}
    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=200, random_state=0).fit(X, d.miss)
    rules = export_text(tree, feature_names=list(X.columns), show_weights=True, decimals=3)
    (out / "miss_tree.txt").write_text(rules, encoding="utf-8")
    ins = d[d["사양내"]]
    mi = ins[ins.miss == 1]
    mt["사양안"] = {"수": len(ins), "놓침": len(mi), "놓침률": round(float(mi.shape[0] / max(len(ins), 1)), 4),
                  "놓친것_호기": mi.machine.value_counts().sort_index().to_dict(),
                  "놓친것_가장자리<4px_비율": round(float((mi.edge_dist < 4).mean()), 3) if len(mi) else None,
                  "전체사양안_가장자리<4px_비율": round(float((ins.edge_dist < 4).mean()), 3),
                  "놓친것_띠안_비율": round(float(mi.in_band.mean()), 3) if len(mi) else None,
                  "전체사양안_띠안_비율": round(float(ins.in_band.mean()), 3),
                  "놓친것_주변결_중앙": round(float(mi.texture.median()), 2) if len(mi) else None,
                  "전체사양안_주변결_중앙": round(float(ins.texture.median()), 2)}
    mi.to_csv(out / "miss_inspec.csv", index=False, encoding="utf-8-sig")
    summary["놓침"] = mt
    print("놓침", json.dumps(mt["사양안"], ensure_ascii=False))

    # ---------- 2. 헛경보 조건 ----------
    fr = ROOT / "results/froc"
    ids = man.index[man.labeled & (man.split == "test")].tolist()
    gt = M.load_gt(ids, data / "clean/labels", {i: (man.w[i], man.h[i]) for i in ids})
    rows = []
    # (a) 실제 사진 + 가짜 정상
    pr = pd.read_csv(fr / f"preds_실제_{args.yolo}.csv")
    q = pd.DataFrame({"id": pr["img"], "x0": pr.px, "y0": pr.py, "x1": pr.px, "y1": pr.py, "score": pr.score})
    pm, _ = M.match(q, gt, "center")
    for r in pm[(pm.tp == 0) & (pm.score >= FP_WIDE)].itertuples():
        normal = r.id.endswith("__normal")
        src = r.id.replace("__normal", "")
        rows.append(dict(set="가짜 정상" if normal else "실제 불량 사진", img=r.id, src=src, x=r.x0, y=r.y0, score=r.score,
                         path=str(data / ("normal/test" if normal else "clean/images") / f"{src}.png")))
    # (b) 시험편 사진 (넣은 시험편 ±7px 밖)
    pt = pd.read_csv(fr / f"preds_시험편_{args.yolo}.csv")
    td = pd.read_csv(data / "testpiece/defects.csv")
    by = dict(tuple(td.groupby("img")))
    pt = pt[pt.score >= FP_WIDE]
    for r in pt.itertuples():
        g = by[r.img]
        if ((np.abs(g.cx - r.px) <= HALF + MARGIN) & (np.abs(g.cy - r.py) <= HALF + MARGIN)).any():
            continue
        src = g.src.iloc[0]
        rows.append(dict(set="시험편 사진", img=r.img, src=src, x=r.px, y=r.py, score=r.score,
                         path=str(data / "testpiece/images" / f"{r.img}.png")))
    fp = pd.DataFrame(rows)
    rnd = []
    rng = np.random.default_rng(0)
    for i, r in fp.iterrows():
        I = Img.get(r.path)
        f = feats(I, r.x, r.y)
        e = gt[r.src] if r.src in gt else np.zeros((0, 4))
        dist_e = float(np.min(np.hypot((e[:, 0] + e[:, 2]) / 2 - r.x, (e[:, 1] + e[:, 3]) / 2 - r.y))) if len(e) else np.inf
        c = measure(I["g"], r.x, r.y)
        fp.loc[i, list(f) + ["지운자리거리", "측정대비"]] = list(f.values()) + [dist_e, c["contrast"]]
        fp.loc[i, "machine"] = int(man.machine[r.src])
    fp["판정헛경보"] = fp.score >= thr
    fp["자리"] = fp.src + "@" + (fp.x / 6).round().astype(int).astype(str) + "," + (fp.y / 6).round().astype(int).astype(str)
    uniq = fp.sort_values("score", ascending=False).drop_duplicates("자리")
    # 같은 사진들의 무작위 제품 안 지점 (비교 기준)
    for path in fp.path.unique():
        I = Img.get(path)
        src = Path(path).stem.split("__")[0]
        ys, xs = np.nonzero(I["pm"])
        e = gt.get(src, np.zeros((0, 4)))
        for j in rng.integers(len(xs), size=RNG_PTS):
            x, y = xs[j], ys[j]
            dist_e = float(np.min(np.hypot((e[:, 0] + e[:, 2]) / 2 - x, (e[:, 1] + e[:, 3]) / 2 - y))) if len(e) else np.inf
            rnd.append(dict(feats(I, x, y), 지운자리거리=dist_e))
    rnd = pd.DataFrame(rnd)

    def describe(df):
        return {"수": len(df),
                "지운자리_6px안": round(float((df.지운자리거리 <= ERASE_NEAR).mean()), 3),
                "가장자리_4px안": round(float((df.edge_dist < EDGE_NEAR).mean()), 3),
                "제품밖": round(float((~df.in_product.astype(bool)).mean()), 3),
                "띠안": round(float(df.in_band.astype(bool).mean()), 3),
                "주변결_중앙": round(float(df.texture.median()), 2)}
    ft = {"정의": f"정답에 맞지 않은 예측, 점수 ≥ {FP_WIDE} (후보) / ≥ 합격선 {thr} (판정 헛경보)",
          "후보_전체": describe(fp), "판정헛경보": describe(fp[fp.판정헛경보]), "무작위_제품안_지점": describe(rnd),
          "고유자리_후보": describe(uniq), "고유자리_판정헛경보": describe(uniq[uniq.판정헛경보]),
          "판정헛경보_사진종류별": fp[fp.판정헛경보].groupby("set").size().to_dict(),
          "판정헛경보_사진수": {"가짜 정상": 73, "실제 불량 사진": 73, "시험편 사진": 3600},
          "판정헛경보_호기별": fp[fp.판정헛경보].groupby("machine").size().astype(int).to_dict()}
    fx = uniq[uniq.판정헛경보]
    cat = np.select([fx.지운자리거리 <= ERASE_NEAR, fx.edge_dist < EDGE_NEAR, fx.측정대비 >= 0.10],
                    ["지운 자리", "제품 가장자리", "어두운 점(대비 0.10 이상)"], "기타")
    ft["판정헛경보_분류(고유 자리, 앞 조건 우선)"] = pd.Series(cat).value_counts().to_dict()
    ft["고유자리_판정헛경보_목록"] = [dict(자리=r.자리, 호기=int(r.machine), 최고점수=round(float(r.score), 3),
                                       측정대비=round(float(r.측정대비), 3), 반복수=int((fp.판정헛경보 & (fp.자리 == r.자리)).sum()))
                                  for r in fx.itertuples()]
    cand = uniq[~uniq.판정헛경보]
    ft["고유자리_후보_분류(합격선 미만, 앞 조건 우선)"] = pd.Series(np.select(
        [cand.지운자리거리 <= ERASE_NEAR, cand.edge_dist < EDGE_NEAR, cand.측정대비 >= 0.10],
        ["지운 자리", "제품 가장자리", "어두운 점(대비 0.10 이상)"], "기타")).value_counts().to_dict()
    fp.sort_values("score", ascending=False).to_csv(out / "fp.csv", index=False, encoding="utf-8-sig")
    summary["헛경보"] = ft
    print("헛경보", json.dumps(ft["고유자리_판정헛경보_목록"], ensure_ascii=False), ft["고유자리_후보"])

    # 확대 그림: 판정 헛경보 점수 높은 순 최대 24개 (중앙 십자 없음, 48px → 144px)
    tiles = []
    top = uniq.head(24)            # 고유 자리, 합격선 미만 후보 포함 점수 높은 순
    for r in top.itertuples():
        g = np.pad(Img.get(r.path)["g"], 24, mode="edge")
        c = g[int(r.y):int(r.y) + 48, int(r.x):int(r.x) + 48]
        tiles.append(cv2.resize(c, (144, 144), interpolation=cv2.INTER_NEAREST))
    if tiles:
        while len(tiles) % 6:
            tiles.append(np.full((144, 144), 255, np.uint8))
        rowsimg = [np.hstack(sum([[t, np.full((144, 6), 255, np.uint8)] for t in tiles[k:k + 6]], [])[:-1])
                   for k in range(0, len(tiles), 6)]
        grid = np.vstack(sum([[r_, np.full((6, r_.shape[1]), 255, np.uint8)] for r_ in rowsimg], [])[:-1])
        Image.fromarray(grid).save(out / "fp_top.png")
        top[["set", "img", "x", "y", "score", "판정헛경보", "지운자리거리", "edge_dist", "측정대비"]] \
            .to_csv(out / "fp_top.csv", index=False, encoding="utf-8-sig")

    # ---------- 3. 실제 이물 중 점수 낮은 것 ----------
    q2 = pm[pm.tp == 1]
    best = {}
    for r in q2.itertuples():
        best[(r.id, r.gt_idx)] = max(best.get((r.id, r.gt_idx), 0), r.score)
    real = []
    for i, g in gt.items():
        I = Img.get(data / "clean/images" / f"{i}.png")
        for j, b in enumerate(g):
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            real.append(dict(img=i, machine=int(man.machine[i]), score=best.get((i, j), 0.0),
                             측정대비=measure(I["g"], cx, cy)["contrast"], **feats(I, cx, cy)))
    real = pd.DataFrame(real).sort_values("score")
    real.to_csv(out / "real_scores.csv", index=False, encoding="utf-8-sig")
    low = real.head(10)
    summary["실제이물_낮은점수10"] = {"점수범위": [round(float(low.score.min()), 3), round(float(low.score.max()), 3)],
                                "호기": low.machine.value_counts().sort_index().to_dict(),
                                "측정대비_중앙(하위10 / 전체)": [round(float(low.측정대비.median()), 3), round(float(real.측정대비.median()), 3)],
                                "가장자리4px안(하위10 / 전체)": [round(float((low.edge_dist < 4).mean()), 2), round(float((real.edge_dist < 4).mean()), 2)],
                                "주변결_중앙(하위10 / 전체)": [round(float(low.texture.median()), 2), round(float(real.texture.median()), 2)]}
    json.dump(summary, open(out / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=str)


if __name__ == "__main__":
    main()
