"""놓침 위험 지도: 합격으로 판정된 사진에서도 "옅은 이물이 있었다면 AI가 놓쳤을 수 있는 구역"을 제품 구조로 설명한다.

지금 AI 는 '이물 있음'에는 위치와 근거 지도를 낼 수 있지만, '이물 없음(합격)'에는 아무 설명도 못 한다.
conditions.py 에서 놓침이 대비 다음으로 주변 결(배경의 거친 정도) · 호기 · 밝기에 따라 달라진다는 것을 확인했으므로,
그 관계를 모형으로 만들어 사진의 모든 자리에 '같은 옅은 이물을 여기 두면 놓칠 확률'을 칠한다.

1. 모형 (검증 시험편 14,400개, val 만 사용)
   놓침(점수 < 판정 합격선) ~ 측정 대비 + 지름 + 가장자리 거리(log) + 띠 안 + 주변 밝기 + 주변 결 + 호기  (로지스틱 회귀)
2. 위험 지도: 사진의 제품 안 모든 화소에 기준 이물(지름 REF_D, 측정 대비 REF_C)을 둔다고 보고 놓침 확률을 계산.
   '고위험 구역' = 검증 사진 제품 화소의 위험 상위 HIGH_Q 지점 이상 (기준값을 val 에서 고정)
3. 검증 (시험 시험편 14,400개, test 는 여기서 처음 봄)
   (a) 모형 전체 vs 대비 · 지름만 쓴 모형의 놓침 예측 AUC  → 구조 정보가 더해 주는 몫
   (b) 대비 0.10~0.20 으로 세기를 묶은 시험편에서, 위치 정보만 쓴 위험(기준 이물)의 AUC
   (c) 고위험 구역이 제품 면적의 몇 % 이고, 놓친 시험편의 몇 % 가 그 안에 있는가 (재검사 때 볼 곳을 얼마나 줄이나)
   (d) 위험 구간별 예측 놓침률과 실제 놓침률 (보정)
4. 사진별 리포트 (시험 실제 불량 73장 + 가짜 정상 73장): 판정, 박스마다 그 자리의 구조(띠 · 가장자리 거리 · 주변 결),
   고위험 구역 면적 비율, 예시 그림. 박스 자리의 주변 결은 이물을 뺀 고리 영역(7~14px)에서 잰다
   (이물 자체가 결을 거칠게 만들어, 그대로 재면 모든 박스가 '거침'으로 나온다).
   위험 지도는 이물이 보이지 않은 합격 사진의 '놓쳤을 수 있는 곳'을 위한 것이라, 이물이 있는 사진에서는 이물 주변이 함께 칠해진다.

실행: .venv\\Scripts\\python.exe src\\miss_risk.py
결과: results/miss_risk/ (summary.json, report_test.csv, boxes_test.csv, examples.png, model.json)
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from conditions import Img

ROOT = Path(__file__).resolve().parents[1]
REF_D, REF_C = 2.0, 0.15      # 위험 지도의 기준 이물: 지름 2px, 주변보다 15% 어두움 (검출 사양 근처의 옅은 이물)
HIGH_Q = 0.80                 # 고위험 구역: 검증 사진 제품 화소 위험의 상위 20%
FEATS = ["측정대비", "지름", "가장자리거리(log)", "띠안", "주변밝기", "주변결", "2호기", "3호기"]


def design(c, d, edge, band, bg, tex, machine):
    """특징 표 (배열 모양 그대로 받아 열로 편다)."""
    m = np.broadcast_to(machine, np.shape(edge))
    return np.stack([np.broadcast_to(c, np.shape(edge)), np.broadcast_to(d, np.shape(edge)), np.log1p(edge),
                     band.astype(float), bg, tex, (m == 2).astype(float), (m == 3).astype(float)], -1).reshape(-1, len(FEATS))


class Model:
    def __init__(self, X, y):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        self.lr = LogisticRegression(C=10, max_iter=3000).fit((X - self.mu) / self.sd, y)

    def p(self, X):
        return self.lr.predict_proba((X - self.mu) / self.sd)[:, 1]


def defect_table(scored, normal_dir, thr, man):
    rows = []
    for r in scored.itertuples():
        I = Img.get(normal_dir / f"{r.src}.png")
        h, w = I["g"].shape
        x, y = int(np.clip(r.cx, 0, w - 1)), int(np.clip(r.cy, 0, h - 1))
        rows.append((r.c_meas, r.d, I["dist"][y, x], I["band"][y, x], I["bg"][y, x], I["tex"][y, x], int(man.machine[r.src]),
                     int(r.YOLO_score < thr)))
    t = pd.DataFrame(rows, columns=["c", "d", "edge", "band", "bg", "tex", "machine", "miss"])
    X = design(t.c.to_numpy(), t.d.to_numpy(), t.edge.to_numpy(), t.band.to_numpy(), t.bg.to_numpy(), t.tex.to_numpy(), t.machine.to_numpy())
    return t, X


def ring_texture(I, cx, cy, r_in=7, r_out=14):
    """박스 자리의 주변 결: 이물 자체가 결을 거칠게 재지 않도록 중심 r_in px 안을 빼고 고리 영역에서 고역 성분의 표준편차."""
    f = I["g"].astype(np.float32)
    hp = f - cv2.GaussianBlur(f, (0, 0), 3)
    h, w = f.shape
    yy, xx = np.mgrid[max(0, cy - r_out):min(h, cy + r_out + 1), max(0, cx - r_out):min(w, cx + r_out + 1)]
    d = np.hypot(xx - cx, yy - cy)
    m = (d >= r_in) & (d <= r_out)
    return float(hp[yy[m], xx[m]].std())


def risk_map(model, I, machine):
    """제품 안 화소마다 기준 이물의 놓침 확률. 제품 밖은 NaN."""
    X = design(REF_C, REF_D, I["dist"], I["band"], I["bg"], I["tex"], machine)
    r = model.p(X).reshape(I["g"].shape)
    r[I["pm"] == 0] = np.nan
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo", default="ratio3_e100")
    args = ap.parse_args()
    data = ROOT / "data"
    out = ROOT / "results" / "miss_risk"
    out.mkdir(parents=True, exist_ok=True)
    thr = json.load(open(ROOT / "results/risk_threshold/summary.json", encoding="utf-8"))["채택"]["합격선"]
    man = pd.read_csv(data / "manifest.csv").set_index("id")

    # 1. 모형 (val)
    tv, Xv = defect_table(pd.read_csv(ROOT / f"results/testpiece_val_{args.yolo}/defects_scored.csv"), data / "normal/val", thr, man)
    full = Model(Xv, tv.miss.to_numpy())
    base = Model(Xv[:, :2], tv.miss.to_numpy())                       # 대비 · 지름만
    summary = {"합격선": thr, "기준이물": {"지름": REF_D, "측정대비": REF_C},
               "모형_오즈비(1표준편차당)": {f: round(float(np.exp(b)), 3) for f, b in zip(FEATS, full.lr.coef_[0])},
               "val_놓침률": round(float(tv.miss.mean()), 3)}
    json.dump({"특징": FEATS, "평균": full.mu.tolist(), "표준편차": full.sd.tolist(), "계수": full.lr.coef_[0].tolist(),
               "절편": float(full.lr.intercept_[0]), "기준이물": [REF_D, REF_C]}, open(out / "model.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    # 고위험 기준값: val 사진들의 제품 화소 위험 분포 상위 20%
    vals = []
    for s in sorted(p.stem for p in (data / "normal/val").glob("*.png")):
        r = risk_map(full, Img.get(data / "normal/val" / f"{s}.png"), int(man.machine[s]))
        vals.append(r[~np.isnan(r)])
    high = float(np.quantile(np.concatenate(vals), HIGH_Q))
    summary["고위험_기준값(val 상위20%)"] = round(high, 4)

    # 2. 검증 (test)
    tt, Xt = defect_table(pd.read_csv(ROOT / "results/testpiece/defects_scored.csv"), data / "normal/test", thr, man)
    y = tt.miss.to_numpy()
    pf, pb = full.p(Xt), base.p(Xt[:, :2])
    ev = {"(a) 전체_모형_AUC": round(float(roc_auc_score(y, pf)), 4), "(a) 대비·지름만_AUC": round(float(roc_auc_score(y, pb)), 4)}
    mid = ((tt.c >= 0.10) & (tt.c < 0.20)).to_numpy()
    Xref = Xt.copy()
    Xref[:, 0], Xref[:, 1] = REF_C, REF_D                              # 세기를 지우고 위치 정보만 남긴 위험
    pr = full.p(Xref)
    ev["(b) 대비0.10~0.20_위치정보만_AUC"] = round(float(roc_auc_score(y[mid], pr[mid])), 4)
    ev["(b) 해당_시험편수 · 놓침수"] = [int(mid.sum()), int(y[mid].sum())]
    inhigh = pr >= high
    ev["(c) 고위험구역_안_놓침비율(대비0.10~0.20)"] = round(float(inhigh[mid & (y == 1)].mean()), 3)
    ev["(c) 고위험구역_안_놓침비율(전체)"] = round(float(inhigh[y == 1].mean()), 3)
    ev["(c) 고위험구역_안_시험편비율(전체)"] = round(float(inhigh.mean()), 3)
    area = []
    for s in sorted(p.stem for p in (data / "normal/test").glob("*.png")):
        r = risk_map(full, Img.get(data / "normal/test" / f"{s}.png"), int(man.machine[s]))
        area.append(float((r[~np.isnan(r)] >= high).mean()))
    ev["(c) 고위험구역_제품면적비율(test 사진 평균)"] = round(float(np.mean(area)), 3)
    q = pd.qcut(pf, 5, labels=False, duplicates="drop")
    ev["(d) 위험5구간_예측vs실제놓침률"] = [{"예측": round(float(pf[q == k].mean()), 3), "실제": round(float(y[q == k].mean()), 3),
                                    "수": int((q == k).sum())} for k in sorted(set(q))]
    for mc in [1, 2, 3]:
        sel = mid & (tt.machine == mc).to_numpy()
        if y[sel].sum() > 0 and (1 - y[sel]).sum() > 0:
            ev[f"(b) {mc}호기_위치정보만_AUC"] = round(float(roc_auc_score(y[sel], pr[sel])), 4)
    summary["검증(test 시험편)"] = ev
    print(json.dumps(ev, ensure_ascii=False, indent=1))

    # 3. 사진별 리포트 (test 실제 불량 + 가짜 정상)
    sub = pd.read_csv(ROOT / "results/submission/test_images.csv").set_index("id")
    ids = sub.index.tolist()
    box = pd.read_csv(ROOT / "results/submission/test_boxes.csv")
    # 주변 결 등급 기준: 검증 가짜 정상의 제품 안 무작위 지점에서 같은 고리 방식으로 잰 값의 3분위
    rng = np.random.default_rng(0)
    ring_vals = []
    for s_ in sorted(p.stem for p in (data / "normal/val").glob("*.png")):
        Iv = Img.get(data / "normal/val" / f"{s_}.png")
        ys, xs = np.nonzero(Iv["pm"])
        ring_vals += [ring_texture(Iv, int(xs[j]), int(ys[j])) for j in rng.integers(len(xs), size=60)]
    tex_q = np.quantile(ring_vals, [1 / 3, 2 / 3])
    summary["주변결_등급기준(고리, val 3분위)"] = [round(float(v), 2) for v in tex_q]
    rep, brow = [], []
    for i in ids:
        I = Img.get(data / "clean/images" / f"{i}.png")
        r = risk_map(full, I, int(man.machine[i]))
        ok = ~np.isnan(r)
        rep.append(dict(id=i, 호기=int(man.machine[i]), 판정=sub.loc[i, "판정"], 최고점수=sub.loc[i, "최고점수"],
                        고위험구역_면적비율=round(float((r[ok] >= high).mean()), 3), 위험_중앙=round(float(np.median(r[ok])), 3)))
        for b in box[box.id == i].itertuples():
            cx, cy = int((b.x0 + b.x1) / 2), int((b.y0 + b.y1) / 2)
            t = ring_texture(I, cx, cy)
            brow.append(dict(id=i, x=cx, y=cy, 점수=b.score, 박스판정=b.박스판정, 띠=("띠 안" if I["band"][cy, cx] else "띠 밖"),
                             가장자리거리=round(float(I["dist"][cy, cx]), 1),
                             주변결=("매끈" if t < tex_q[0] else "중간" if t < tex_q[1] else "거침"), 주변결_값=round(t, 2)))
    rep = pd.DataFrame(rep)

    # 실제 이물은 어떤 자리에 있나: 이물을 지운 가짜 정상으로 위험 지도를 계산해(이물 자신의 영향 제외) 정답 중심이 고위험 구역에 드는 비율
    import metrics as M
    gt = M.load_gt(ids, data / "clean/labels", {i: (man.w[i], man.h[i]) for i in ids})
    inside, rings, n_gt = 0, [], 0
    for i, g in gt.items():
        In = Img.get(data / "normal/test" / f"{i}.png")
        r = risk_map(full, In, int(man.machine[i]))
        for b_ in g:
            cx, cy = int((b_[0] + b_[2]) / 2), int((b_[1] + b_[3]) / 2)
            inside += int(np.nan_to_num(r[cy, cx]) >= high)
            rings.append(ring_texture(In, cx, cy))
            n_gt += 1
    band_rand = []
    for s_ in sorted(p.stem for p in (data / "normal/test").glob("*.png")):
        In = Img.get(data / "normal/test" / f"{s_}.png")
        ys, xs = np.nonzero(In["band"])
        band_rand += [ring_texture(In, int(xs[j]), int(ys[j])) for j in rng.integers(len(xs), size=30)]
    summary["실제이물_자리"] = {"이물수": n_gt, "고위험구역_안": inside, "비율": round(inside / n_gt, 3),
                           "비교_고위험구역_제품면적비율": ev["(c) 고위험구역_제품면적비율(test 사진 평균)"],
                           "고리결_중앙(실제 이물 자리)": round(float(np.median(rings)), 2),
                           "고리결_중앙(띠 안 무작위)": round(float(np.median(band_rand)), 2),
                           "고리결_중앙(제품 안 무작위, val)": round(float(np.median(ring_vals)), 2),
                           "고리결_거침_비율(실제 이물 자리)": round(float((np.array(rings) >= tex_q[1]).mean()), 3)}
    print("실제이물_자리", summary["실제이물_자리"])
    rep.to_csv(out / "report_test.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(brow).to_csv(out / "boxes_test.csv", index=False, encoding="utf-8-sig")
    summary["사진별_리포트"] = {"사진수": len(rep), "고위험구역_면적비율_중앙": round(float(rep.고위험구역_면적비율.median()), 3),
                           "호기별_고위험면적_중앙": rep.groupby("호기").고위험구역_면적비율.median().round(3).to_dict()}

    # 예시 그림: 호기별 한 장씩, 원본 | 위험 지도 겹침 (글자 없음, 박스는 청록 테두리)
    tiles = []
    for mc in [1, 2, 3]:
        i = rep[rep.호기 == mc].sort_values("고위험구역_면적비율").iloc[len(rep[rep.호기 == mc]) // 2]["id"]
        I = Img.get(data / "clean/images" / f"{i}.png")
        r = risk_map(full, I, mc)
        g = cv2.cvtColor(I["g"], cv2.COLOR_GRAY2RGB)
        heat = np.zeros_like(g)
        hi = np.nan_to_num(r) >= high
        heat[hi] = (232, 93, 4)
        over = np.where(hi[..., None], (0.55 * g + 0.45 * heat).astype(np.uint8), g)
        for b in box[box.id == i].itertuples():
            cv2.rectangle(over, (int(b.x0) - 2, int(b.y0) - 2), (int(b.x1) + 2, int(b.y1) + 2), (15, 118, 110), 1)
        both = np.hstack([g, np.full((g.shape[0], 6, 3), 255, np.uint8), over])
        both = cv2.resize(both, (int(both.shape[1] * 360 / both.shape[0]), 360), interpolation=cv2.INTER_AREA)
        tiles.append(both)
        summary.setdefault("예시사진", []).append(i)
    wmax = max(t.shape[1] for t in tiles)
    canvas = np.vstack(sum([[np.pad(t, ((0, 0), (0, wmax - t.shape[1]), (0, 0)), constant_values=255),
                             np.full((10, wmax, 3), 255, np.uint8)] for t in tiles], [])[:-1])
    Image.fromarray(canvas).save(out / "examples.png")
    json.dump(summary, open(out / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=str)


if __name__ == "__main__":
    main()
