"""FROC: 사진 한 장당 헛경보 수(FPPI)에 따른 이물 검출률(민감도) 곡선.

이물 하나하나를 찾는 문제라 사진 단위 ROC 대신, 의료 영상 결절 검출에서 쓰는 FROC 로 모델을 비교한다.
요약값은 LUNA16(Setio et al. 2017) 의 CPM: FPPI 1/8 · 1/4 · 1/2 · 1 · 2 · 4 · 8 에서 민감도의 평균.

두 가지 시험 (모두 test split, 기준은 이미 val 로 정했으므로 여기서는 재기만 한다)
  ① 실제: 시험 실제 불량 사진 73장의 이물 139개 + 같은 사진의 가짜 정상 73장 (헛경보 = 이물에 맞지 않은 예측)
  ② 시험편: 가짜 정상 위에 넣은 가상 시험편 14,400개 (testpiece.py, 3,600장)
맞춤: 예측 중심이 정답 박스(±2px) 안 (metrics.match 'center'). 시험편은 중심 ±7px (testpiece.py 사양표 채점과 같은 HALF 5 + MARGIN 2).
운영점: 박스 기준선(val F1 최대) 과 판정 합격선(risk_threshold 채택값) 에서의 민감도 · FPPI 도 함께 적는다.

실행: .venv\\Scripts\\python.exe src\\froc.py
결과: results/froc/ (summary.json, froc_real.png, froc_testpiece.png, preds_*.csv)
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import metrics as M
from synth_eval import MARGIN, baseline_preds, yolo_preds
from testpiece import HALF

ROOT = Path(__file__).resolve().parents[1]
FPPI_POINTS = [0.125, 0.25, 0.5, 1, 2, 4, 8]
LABEL = {"베이스라인": "규칙 기반", "cnn_aug": "조각 분류 CNN", "y26s_640": "YOLO (합성 전)", "ratio3_e100": "최종 YOLO (합성 3배)"}
plt.rcParams["font.family"] = "Malgun Gothic"


def preds(model, paths, ids, machines, bl):
    if model == "베이스라인":
        return baseline_preds(bl, paths, ids, machines)
    return yolo_preds(model, paths, ids, machines=machines)


def froc_curve(scores, tp, n_gt, n_img):
    """scores/tp: 모든 예측(이미 정답 배정됨). 반환: 기준선 내림차순의 (민감도, FPPI, 기준선)."""
    o = np.argsort(-scores, kind="stable")
    s, t = scores[o], tp[o]
    sens = np.cumsum(t) / n_gt
    fppi = np.cumsum(1 - t) / n_img
    return sens, fppi, s


def sens_at(sens, fppi, f):
    ok = fppi <= f
    return float(sens[ok].max()) if ok.any() else 0.0


def at_thr(scores, tp, n_gt, n_img, thr):
    m = scores >= thr
    return {"기준선": round(float(thr), 4), "민감도": round(float(tp[m].sum() / n_gt), 4),
            "FPPI": round(float((1 - tp[m]).sum() / n_img), 4), "헛경보": int((1 - tp[m]).sum())}


def match_real(p, gt):
    """p: img, px, py, score → metrics.match 로 정답 배정 (가짜 정상 사진은 정답 없음)."""
    q = pd.DataFrame({"id": p["img"], "x0": p.px, "y0": p.py, "x1": p.px, "y1": p.py, "score": p.score})
    pm, _ = M.match(q, gt, "center")
    return pm["score"].to_numpy(), pm["tp"].to_numpy()


def match_tp(p, defects):
    """시험편: 이물별로 가장 높은 점수의 근처 예측 하나만 TP, 나머지는 FP."""
    p = p.sort_values("score", ascending=False).reset_index(drop=True)
    tp = np.zeros(len(p), int)
    by = dict(tuple(defects.groupby("img")))
    for img, q in p.groupby("img", sort=False):
        g = by.get(img)
        if g is None:
            continue
        used = np.zeros(len(g), bool)
        half = HALF + MARGIN
        for idx, r in zip(q.index, q[["px", "py"]].to_numpy()):
            ok = (np.abs(g.cx.to_numpy() - r[0]) <= half) & (np.abs(g.cy.to_numpy() - r[1]) <= half) & ~used
            if ok.any():
                j = np.where(ok)[0][np.argmin(np.hypot(g.cx.to_numpy()[ok] - r[0], g.cy.to_numpy()[ok] - r[1]))]
                used[j] = True
                tp[idx] = 1
    return p["score"].to_numpy(), tp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["베이스라인", "cnn_aug", "y26s_640", "ratio3_e100"])
    ap.add_argument("--final", default="ratio3_e100")
    ap.add_argument("--reuse", action="store_true", help="저장된 preds_*.csv 를 다시 씀")
    args = ap.parse_args()
    data = ROOT / "data"
    out = ROOT / "results" / "froc"
    out.mkdir(parents=True, exist_ok=True)
    bl = json.load(open(ROOT / "results/baseline_clean/metrics.json", encoding="utf-8"))["params"]

    man = pd.read_csv(data / "manifest.csv").set_index("id")
    ids = man.index[man.labeled & (man.split == "test")].tolist()
    gt = M.load_gt(ids, data / "clean/labels", {i: (man.w[i], man.h[i]) for i in ids})
    n_gt = sum(len(g) for g in gt.values())
    real_paths = [data / "clean/images" / f"{i}.png" for i in ids] + [data / "normal/test" / f"{i}.png" for i in ids]
    real_ids = ids + [f"{i}__normal" for i in ids]
    real_m = [int(man.machine[i]) for i in ids] * 2
    tp_d = pd.read_csv(data / "testpiece/defects.csv")
    tp_i = pd.read_csv(data / "testpiece/images.csv")
    tp_paths = [data / "testpiece/images" / f"{i}.png" for i in tp_i.img]

    thr_box = json.load(open(ROOT / f"results/yolo_{args.final}/metrics.json", encoding="utf-8"))["thresholds"]["F1최대"]
    thr_img = json.load(open(ROOT / "results/risk_threshold/summary.json", encoding="utf-8"))["채택"]["합격선"]

    summary = {"설정": {"실제_이물": n_gt, "실제_사진": len(real_ids), "시험편_이물": len(tp_d), "시험편_사진": len(tp_i),
                      "FPPI_점": FPPI_POINTS, "박스기준선": round(thr_box, 4), "판정합격선": thr_img}}
    curves = {}
    for test, paths, iid, mach in [("실제", real_paths, real_ids, real_m),
                                   ("시험편", tp_paths, tp_i.img.tolist(), tp_i.machine.tolist())]:
        n_img = len(paths)
        for m in args.models:
            f = out / f"preds_{test}_{m}.csv"
            if args.reuse and f.exists():
                p = pd.read_csv(f)
            else:
                p = preds(m, paths, iid, mach, bl)
                p.to_csv(f, index=False, encoding="utf-8-sig")
            if test == "실제":
                s, t = match_real(p, gt)
                ng = n_gt
            else:
                s, t = match_tp(p, tp_d)
                ng = len(tp_d)
            sens, fppi, th = froc_curve(s, t, ng, n_img)
            pts = {str(x): round(sens_at(sens, fppi, x), 4) for x in FPPI_POINTS}
            res = {"CPM": round(float(np.mean(list(pts.values()))), 4), "FPPI별_민감도": pts,
                   "최대_민감도": round(float(sens[-1]), 4)}
            if m == args.final:
                res["운영점_박스기준선"] = at_thr(s, t, ng, n_img, thr_box)
                res["운영점_판정합격선"] = at_thr(s, t, ng, n_img, thr_img)
                if test == "시험편":      # 호기별 곡선 요약
                    res["호기별_CPM"] = {}
                    for mc in sorted(tp_i.machine.unique()):
                        ims = set(tp_i.img[tp_i.machine == mc])
                        sel = p["img"].isin(ims).to_numpy()
                        ps, pt = match_tp(p[sel], tp_d[tp_d.img.isin(ims)])
                        se, fp_, _ = froc_curve(ps, pt, int(tp_d.img.isin(ims).sum()), len(ims))
                        res["호기별_CPM"][f"{mc}호기"] = round(float(np.mean([sens_at(se, fp_, x) for x in FPPI_POINTS])), 4)
            summary.setdefault(test, {})[m] = res
            curves[(test, m)] = (sens, fppi)
            print(test, m, res["CPM"], res.get("운영점_판정합격선", ""))

    colors = {"베이스라인": "#9aa5b1", "cnn_aug": "#e0a458", "y26s_640": "#6c8ebf", "ratio3_e100": "#0f766e"}
    for test, xmin in [("실제", 1 / 160), ("시험편", 1 / 100)]:
        fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=200)
        for m in args.models:
            sens, fppi = curves[(test, m)]
            ok = fppi > 0
            ax.step(np.r_[xmin, fppi[ok]], np.r_[sens[~ok].max() if (~ok).any() else 0, sens[ok]], where="post",
                    color=colors.get(m, "k"), lw=2.2 if m == args.final else 1.4,
                    label=f"{LABEL.get(m, m)}  CPM {summary[test][m]['CPM']:.3f}")
        op = summary[test][args.final].get("운영점_판정합격선")
        if op and op["FPPI"] > 0:
            ax.plot(op["FPPI"], op["민감도"], "o", ms=6, mfc="white", mec=colors.get(args.final, "k"), mew=1.8, zorder=5)
            ax.annotate(f"판정 합격선 {op['기준선']:.3f}", (op["FPPI"], op["민감도"]), xytext=(8, 7),
                        textcoords="offset points", fontsize=8, color="#374151")
        for x in FPPI_POINTS:
            ax.axvline(x, color="#e5e7eb", lw=0.8, zorder=0)
        ax.set_xscale("log")
        ax.set_xlim(xmin, 8)
        ax.set_ylim(0, 1.01)
        ax.set_xticks([0.01, 1 / 32] + FPPI_POINTS)
        ax.set_xticklabels(["1/100", "1/32", "1/8", "1/4", "1/2", "1", "2", "4", "8"])
        ax.minorticks_off()
        ax.set_xlabel("사진 한 장당 헛경보 수 (FPPI)")
        ax.set_ylabel("이물 검출률")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        fig.tight_layout()
        fig.savefig(out / f"froc_{'real' if test == '실제' else 'testpiece'}.png")
        plt.close(fig)
    json.dump(summary, open(out / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
