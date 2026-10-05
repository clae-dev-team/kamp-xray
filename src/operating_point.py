"""현장 의사결정용 분석 두 가지: 비용에 맞춘 운영점과 재검사 우선순위. 저장된 영상 점수만 다시 계산한다(추론 없음).

1. 비용 기반 운영점
   제품 하나의 기대 비용(재검사 1건의 비용 = 1 로 둔 상대값)
     불량 제품: 합격 → R (놓침 비용), 재검사 → 1, 불합격 → 0
     정상 제품: 합격 → 0, 재검사 → 1, 불합격 → S (정상을 사람 확인 없이 폐기한 비용)
   실제 금액은 알 수 없으므로 R(놓침/재검사 비용 비)과 불량률 p 를 바꿔 가며, 검증 영상에서 기대 비용이 최소인
   합격선 · 불합격선을 찾고 시험 영상에서 그 결과를 본다. 지금 쓰는 보장 기준선이 어느 조건에서 최적에 가까운지도 함께 낸다.
   불량 = 실제 불량 + 검출 사양 이상의 합성 불량 (judge.in_spec), 정상 = 이물을 지운 정상 영상.
2. 재검사 우선순위
   제품을 영상 점수가 높은 순으로 사람이 볼 때, 전체의 x% 를 보면 불량의 몇 % 를 잡는가 (불량률 p 로 가중).
   무작위로 보면 x% 를 봐야 x% 를 잡는다.
3. 우선순위 세 단
   1순위 재검사 구간(합격선 이상 불합격선 미만, 반드시 사람이 봄) → 2순위 합격했지만 약한 신호가 있는 제품(WEAK 이상 합격선 미만,
   여유가 있을 때 보는 표본) → 3순위 나머지(무작위 표본). 불합격선 이상은 자동 배출이라 사람이 보지 않는다.
   단계마다 '사람이 보는 제품 비율'과 '그때까지 걸러진 불량 비율'을 낸다. WEAK 는 zone_rules.py 의 놓침 후보 기준(val 에서 정한 0.3)을 그대로 쓴다.
   정답 없는 사진(unlabeled_check.py 의 결과가 있으면)에서도 같은 단계를 센다: 색 표시가 있는 사진이 어느 단계에서 걸리는지.

실행: .venv\\Scripts\\python.exe src\\operating_point.py
결과: results/operating_point/summary.json, cost_table.csv, priority.csv, tiers.csv, priority.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from judge import in_spec

ROOT = Path(__file__).resolve().parents[1]
RATIOS = [10, 100, 1000, 10000]          # R = 놓침 비용 / 재검사 비용
PREVALENCE = [0.0001, 0.001, 0.01]       # 불량률
SCRAP = 5                                # S = 정상 자동 폐기 비용 / 재검사 비용 (가정)
WEAK = 0.3                               # 2순위(약한 신호) 하한: zone_rules.py 가 val 에서 정한 놓침 후보 기준


def tier_shares(x, t_low, t_high):
    """점수 배열을 (자동 배출, 1순위, 2순위, 3순위) 비율로."""
    x = np.asarray(x)
    return [float((x >= t_high).mean()), float(((x >= t_low) & (x < t_high)).mean()),
            float(((x >= WEAK) & (x < t_low)).mean()), float((x < WEAK).mean())]


def rates(d, n, t_low, t_high):
    """불량 점수 d, 정상 점수 n 에서 (불량 합격, 불량 재검사, 정상 재검사, 정상 불합격) 비율."""
    return (float((d < t_low).mean()), float(((d >= t_low) & (d < t_high)).mean()),
            float(((n >= t_low) & (n < t_high)).mean()), float((n >= t_high).mean()))


def cost(r, p, R, S=SCRAP):
    d_pass, d_re, n_re, n_rej = r
    return p * (d_pass * R + d_re) + (1 - p) * (n_re + n_rej * S)


def best_thresholds(d, n, p, R):
    """검증 점수에서 기대 비용이 최소인 (합격선, 불합격선). 후보는 관측 점수 사이의 중간값."""
    s = np.unique(np.r_[d, n])
    cand = np.r_[0.0, (s[1:] + s[:-1]) / 2, 1.0001]
    best = (np.inf, 0.0, 1.0001)
    for tl in cand:
        d_pass = (d < tl).mean()
        for th in cand[cand >= tl]:
            c = cost((d_pass, ((d >= tl) & (d < th)).mean(), ((n >= tl) & (n < th)).mean(), (n >= th).mean()), p, R)
            if c < best[0] - 1e-12:
                best = (c, float(tl), float(th))
    return best[1], best[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo", default="ratio3_e100")
    args = ap.parse_args()
    m = args.yolo
    out = ROOT / "results" / "operating_point"
    out.mkdir(parents=True, exist_ok=True)
    val = pd.read_csv(ROOT / "results/judge/scores_val.csv")
    test = pd.read_csv(ROOT / "results/judge/scores_test.csv")
    spec = pd.read_csv(ROOT / f"results/testpiece_val_{m}/spec.csv")
    spec = spec[spec["model"] == "YOLO"]
    sets = {}
    for name, df in (("val", val), ("test", test)):
        ok = in_spec(df, spec)
        sets[name] = dict(d=df.loc[ok, m].to_numpy(), n=df.loc[df.kind == "normal", m].to_numpy(),
                          real=df.loc[df.kind == "real_ng", m].to_numpy())
    g = json.load(open(ROOT / "results/risk_threshold/summary.json", encoding="utf-8"))["채택"]
    g_low, g_high = g["합격선"], g["불합격선"]

    # 1. 비용 기반 운영점
    rows = []
    for p in PREVALENCE:
        for R in RATIOS:
            tl, th = best_thresholds(sets["val"]["d"], sets["val"]["n"], p, R)
            rt = rates(sets["test"]["d"], sets["test"]["n"], tl, th)
            rg = rates(sets["test"]["d"], sets["test"]["n"], g_low, g_high)
            c_opt, c_g = cost(rt, p, R), cost(rg, p, R)
            rows.append({"불량률": p, "놓침/재검사_비용비": R, "최적_합격선": round(tl, 3), "최적_불합격선": round(min(th, 1.0), 3),
                         "최적_1만개당_재검사": round(1e4 * (p * rt[1] + (1 - p) * rt[2]), 1),
                         "최적_1만개당_정상폐기": round(1e4 * (1 - p) * rt[3], 1),
                         "최적_1만개당_놓침": round(1e4 * p * rt[0], 3), "최적_비용": round(1e4 * c_opt, 1),
                         "보장기준_1만개당_재검사": round(1e4 * (p * rg[1] + (1 - p) * rg[2]), 1),
                         "보장기준_1만개당_정상폐기": round(1e4 * (1 - p) * rg[3], 1),
                         "보장기준_1만개당_놓침": round(1e4 * p * rg[0], 3), "보장기준_비용": round(1e4 * c_g, 1),
                         "보장기준/최적": round(c_g / c_opt, 2) if c_opt > 0 else None})
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "cost_table.csv", index=False, encoding="utf-8-sig")

    # 2. 재검사 우선순위 (시험 영상, 점수 높은 순)
    d, n, real = sets["test"]["d"], sets["test"]["n"], sets["test"]["real"]
    pr_rows, curves = [], {}
    for p in PREVALENCE:
        s = np.r_[d, n]
        w = np.r_[np.full(len(d), p / len(d)), np.full(len(n), (1 - p) / len(n))]
        isdef = np.r_[np.ones(len(d)), np.zeros(len(n))]
        o = np.argsort(-s, kind="stable")
        frac, rec = np.cumsum(w[o]), np.cumsum((w * isdef)[o]) / p
        curves[p] = (frac, rec)
        row = {"불량률": p}
        for target in (0.90, 0.95, 0.99, 1.0):
            k = int(np.searchsorted(rec, target - 1e-12))
            row[f"불량 {int(target * 100)}% 포착에 필요한 검사 비율"] = round(float(frac[min(k, len(frac) - 1)]), 4)
        for x in (0.01, 0.02, 0.05):
            k = int(np.searchsorted(frac, x, side="right")) - 1
            row[f"상위 {int(x * 100)}% 검사 시 포착률"] = round(float(rec[k]) if k >= 0 else 0.0, 4)
        pr_rows.append(row)
    pr = pd.DataFrame(pr_rows)
    pr.to_csv(out / "priority.csv", index=False, encoding="utf-8-sig")

    # 3. 우선순위 세 단 (시험 판정 세트)
    sd, sn, sr = tier_shares(d, g_low, g_high), tier_shares(n, g_low, g_high), tier_shares(real, g_low, g_high)
    tier_rows = []
    for p in PREVALENCE:
        look1 = p * sd[1] + (1 - p) * sn[1]
        look2 = look1 + p * sd[2] + (1 - p) * sn[2]
        tier_rows.append({"불량률": p, "자동배출_제품비율": round(p * sd[0] + (1 - p) * sn[0], 4),
                          "1순위까지_사람이_보는_비율": round(look1, 4), "1순위까지_걸러진_불량": round(sd[0] + sd[1], 4),
                          "2순위까지_사람이_보는_비율": round(look2, 4), "2순위까지_걸러진_불량": round(sd[0] + sd[1] + sd[2], 4),
                          "3순위에_남는_불량": round(sd[3], 4)})
    tiers = pd.DataFrame(tier_rows)
    tiers.to_csv(out / "tiers.csv", index=False, encoding="utf-8-sig")
    tier_summary = {"약한신호_하한": WEAK,
                    "시험_판정세트": {"불량(실제+사양안 합성)_단계별": dict(zip(["자동배출", "1순위", "2순위", "3순위"], [round(v, 4) for v in sd])),
                                 "실제불량_단계별": dict(zip(["자동배출", "1순위", "2순위", "3순위"], [round(v, 4) for v in sr])),
                                 "정상_단계별": dict(zip(["자동배출", "1순위", "2순위", "3순위"], [round(v, 4) for v in sn])),
                                 "불량률별": tier_rows}}
    # 정답 없는 사진: 색 표시가 있는 사진이 어느 단계에서 걸리는가 (unlabeled_check.py 결과가 있을 때)
    bf, mf = ROOT / "results/unlabeled_check/all_boxes.csv", ROOT / "results/unlabeled_check/marks_scored.csv"
    if bf.exists() and mf.exists():
        man = pd.read_csv(ROOT / "data/manifest.csv")
        top = pd.read_csv(bf).groupby("id")["score"].max()
        mk = pd.read_csv(mf)
        marked = set(mk[(mk.kind == "real") & mk.box_like & ~mk.labeled]["id"])
        un = man[~man.labeled]
        t_m = top.reindex([i for i in un["id"] if i in marked]).fillna(0).to_numpy()
        t_u = top.reindex([i for i in un["id"] if i not in marked]).fillna(0).to_numpy()
        cnt = lambda x: dict(zip(["자동배출", "1순위", "2순위", "3순위"], [int(round(v * len(x))) for v in tier_shares(x, g_low, g_high)]))
        tier_summary["정답없는_사진"] = {"이물표시가_있는_사진": {"사진수": len(t_m), **cnt(t_m)},
                                   "이물표시가_없는_사진": {"사진수": len(t_u), **cnt(t_u)}}

    plt.rcParams["font.family"] = "Malgun Gothic"
    fig, ax = plt.subplots(figsize=(6.2, 3.6), dpi=170)
    for p, c in zip(PREVALENCE, ["#8c5bb5", "#0a8f86", "#c96f24"]):
        f, r = curves[p]
        ax.step(np.r_[0, f] * 100, np.r_[0, r] * 100, where="post", color=c, lw=1.8, label=f"불량률 {p * 100:g}%")
    ax.plot([0, 100], [0, 100], color="#9aa5b1", lw=1, ls="--", label="무작위 순서")
    p_mark = 0.001                                   # 단계 경계는 불량률 0.1% 곡선 위에 표시
    for name, k in (("1순위 끝", 2), ("2순위 끝", 3)):
        x_end = 100 * (p_mark * sum(sd[:k]) + (1 - p_mark) * sum(sn[:k]))
        y_end = 100 * sum(sd[:k])
        ax.scatter([x_end], [y_end], s=34, facecolors="white", edgecolors="#18212b", linewidths=1.3, zorder=5)
        ax.annotate(name, (x_end, y_end), xytext=(0, -15), textcoords="offset points", ha="center", fontsize=7.5, color="#18212b")
    ax.set(xlabel="사람이 검사하는 비율 (점수 높은 순, %)", ylabel="잡아낸 불량 (%)", xlim=(0, 16), ylim=(0, 104))
    ax.grid(alpha=.3)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "priority.png")

    summary = {"모델": m, "보장기준선": {"합격선": g_low, "불합격선": g_high}, "가정": {"정상폐기/재검사_비용비": SCRAP},
               "표본": {s: {"불량(실제+사양안 합성)": int(len(v["d"])), "실제불량": int(len(v["real"])), "정상": int(len(v["n"]))} for s, v in sets.items()},
               "비용표": tab.to_dict("records"), "우선순위": pr.to_dict("records"), "우선순위_단계": tier_summary,
               "실제불량만_최저점수_test": round(float(real.min()), 4), "정상_최고점수_test": round(float(n.max()), 4)}
    json.dump(summary, open(out / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(tab[["불량률", "놓침/재검사_비용비", "최적_합격선", "최적_불합격선", "최적_1만개당_재검사", "최적_1만개당_정상폐기", "최적_1만개당_놓침",
               "보장기준_1만개당_재검사", "보장기준_1만개당_놓침", "보장기준/최적"]].to_string(index=False))
    print(pr.T.to_string())
    print(tiers.T.to_string())
    print(json.dumps({k: v for k, v in tier_summary.items() if k != "시험_판정세트"}, ensure_ascii=False))
    print(json.dumps({k: v for k, v in tier_summary["시험_판정세트"].items() if k != "불량률별"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
