"""수치의 오차 범위: 시험 사진이 73장뿐이라 F1 0.99 같은 값이 어느 정도 믿을 만한지 범위로 보인다.

1. 재표집 (부트스트랩, 학습 없이 저장된 예측으로 계산)
   - 실제 이물: 시험 사진 73장을 복원 추출해 정밀도·재현율·F1 의 95% 구간 (사진 단위).
     같은 호기·같은 날 사진은 서로 비슷하므로 (호기, 날짜) 묶음 단위로 뽑은 구간도 함께 낸다 (묶음 9개라 넓게 나온다).
   - 모델 간 차이: 같은 표본에서 (최종 YOLO − 비교 모델) 의 구간. 0 을 포함하면 차이가 있다고 말할 수 없다.
   - 합성 저대비 이물 검출률: 원본 사진 단위로 묶어 재표집 (한 사진에서 96개씩 만들었으므로 이물 단위로 뽑으면 구간이 지나치게 좁다).
   - 비율 하나짜리 수치(재현율, 정상 합격률, 놓침 0건)는 정확 구간(Clopper–Pearson)도 낸다.
2. 시드 반복 (--seeds): 같은 설정에서 학습 시드만 바꾼 모델들의 수치 범위.
   판정 기준선은 각 모델이 자기 val 에서 다시 정한다 (F1 최대). 이미지 단위는 시험 실제 불량 73장과 이물 지운 가짜 정상 73장의 최고 점수로 본다.

실행: .venv\\Scripts\\python.exe src\\uncertainty.py
      .venv\\Scripts\\python.exe src\\uncertainty.py --seeds ratio3_e100 ratio3_e100_s1 ratio3_e100_s2
결과: results/uncertainty/summary.json (시드 결과는 seeds.json)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

import metrics as M

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "uncertainty"
B = 10000
MODELS = {"규칙(베이스라인)": "baseline_clean", "조각 CNN": "cnn_cnn_aug", "최종 YOLO": "yolo_ratio3_e100"}
SYNTH = {"규칙(베이스라인)": ("synth_eval_ratio3_e100", "베이스라인_hit"), "조각 CNN": ("synth_eval_cnn_aug", "CNN_hit"),
         "최종 YOLO": ("synth_eval_ratio3_e100", "YOLO_hit")}


def exact_ci(k, n, level=0.95):
    """Clopper–Pearson 양쪽 구간."""
    a = (1 - level) / 2
    lo = 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k))
    return [round(lo, 4), round(hi, 4)]


def prf(tp, fp, fn):
    p = tp / np.maximum(tp + fp, 1)
    r = tp / np.maximum(tp + fn, 1)
    return p, r, 2 * tp / np.maximum(2 * tp + fp + fn, 1)


def counts(pred, gt, ids, thr, rule):
    """사진별 TP·FP·FN (metrics.evaluate 와 같은 맞춤 방식)."""
    pm, _ = M.match(pred[pred["id"].isin(gt.keys())][["id", "x0", "y0", "x1", "y1", "score"]], gt, rule)
    p = pm[pm["score"] >= thr]
    tp = p.groupby("id")["tp"].sum().reindex(ids, fill_value=0).to_numpy()
    n = p.groupby("id").size().reindex(ids, fill_value=0).to_numpy()
    ngt = np.array([len(gt[i]) for i in ids])
    return np.stack([tp, n - tp, ngt - tp], 1).astype(float)


def pct(x):
    return [round(float(v), 4) for v in np.percentile(x, [2.5, 97.5])]


def resample(c, idx):
    """c: (단위 수, 3) 합계표, idx: (B, 단위 수) → 재표집별 정밀도·재현율·F1."""
    s = c[idx].sum(1)
    return prf(s[:, 0], s[:, 1], s[:, 2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", default=None, help="시드만 다른 모델 이름들 (첫 번째가 최종 모델)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    data = ROOT / "data"
    man = pd.read_csv(data / "manifest.csv")
    man = man[man["labeled"]].set_index("id")
    if args.seeds:
        return seeds(args.seeds, data, man)

    ids = man.index[man["split"] == "test"].tolist()
    gt = M.load_gt(ids, data / "clean/labels", {i: (man.w[i], man.h[i]) for i in ids})
    grp = (man.loc[ids, "machine"].astype(str) + "_" + man.loc[ids, "date"].astype(str)).to_numpy()
    gnames = sorted(set(grp))
    rng = np.random.default_rng(0)
    idx_img = rng.integers(0, len(ids), (B, len(ids)))
    idx_grp = rng.integers(0, len(gnames), (B, len(gnames)))

    res = {"시험_사진수": len(ids), "시험_이물수": int(sum(len(g) for g in gt.values())), "묶음수(호기·날짜)": len(gnames),
           "재표집_횟수": B, "실제이물": {}}
    boot = {}
    for rule in ["center", "iou50"]:
        for name, folder in MODELS.items():
            thr = json.load(open(ROOT / "results" / folder / "metrics.json", encoding="utf-8"))["thresholds"]["F1최대"]
            c = counts(pd.read_csv(ROOT / "results" / folder / "pred_test.csv"), gt, ids, thr, rule)
            cg = np.stack([c[grp == g].sum(0) for g in gnames])
            tp, fp, fn = c.sum(0)
            p, r, f = prf(tp, fp, fn)
            bp, br, bf = resample(c, idx_img)
            _, _, gf = resample(cg, idx_grp)
            boot[(rule, name)] = bf
            res["실제이물"].setdefault(rule, {})[name] = {
                "TP": int(tp), "FP": int(fp), "FN": int(fn), "정밀도": round(float(p), 4), "재현율": round(float(r), 4),
                "F1": round(float(f), 4), "F1_95%구간(사진 단위)": pct(bf), "F1_95%구간(호기·날짜 묶음)": pct(gf),
                "재현율_95%구간(사진 단위)": pct(br), "정밀도_95%구간(사진 단위)": pct(bp),
                "재현율_정확구간(이물 단위)": exact_ci(int(tp), int(tp + fn))}
        res["실제이물"][rule]["차이(최종 YOLO − 비교)"] = {
            n: {"F1_차이": round(float(np.mean(boot[(rule, "최종 YOLO")] - boot[(rule, n)])), 4),
                "95%구간": pct(boot[(rule, "최종 YOLO")] - boot[(rule, n)])} for n in MODELS if n != "최종 YOLO"}

    # 합성 저대비 이물: 원본 사진 단위 묶음 재표집
    srcs, hits = None, {}
    for name, (folder, col) in SYNTH.items():
        d = pd.read_csv(ROOT / "results" / folder / "defects_scored.csv")
        srcs = sorted(d["src"].unique())
        g = d.groupby("src")[col]
        hits[name] = np.stack([g.sum().reindex(srcs).to_numpy(), g.size().reindex(srcs).to_numpy()], 1).astype(float)
    idx_src = rng.integers(0, len(srcs), (B, len(srcs)))
    rate = {n: h[idx_src].sum(1) for n, h in hits.items()}
    rate = {n: s[:, 0] / s[:, 1] for n, s in rate.items()}
    res["합성_저대비_이물"] = {"이물수": int(hits["최종 YOLO"][:, 1].sum()), "원본_사진수": len(srcs)}
    for n, h in hits.items():
        res["합성_저대비_이물"][n] = {"검출률": round(float(h[:, 0].sum() / h[:, 1].sum()), 4), "95%구간(사진 묶음)": pct(rate[n])}
    res["합성_저대비_이물"]["차이(최종 YOLO − 비교)"] = {
        n: {"검출률_차이": round(float(np.mean(rate["최종 YOLO"] - rate[n])), 4), "95%구간": pct(rate["최종 YOLO"] - rate[n])}
        for n in SYNTH if n != "최종 YOLO"}

    # 3단 판정 (보장 기준선): 비율 하나짜리 수치의 정확 구간
    th = json.load(open(ROOT / "results/risk_threshold/summary.json", encoding="utf-8"))["채택"]
    j = pd.read_csv(ROOT / "results/judge/judged_test.csv")
    s = j[th["모델"]]
    nor, ng = s[j["kind"] == "normal"], s[j["kind"] == "real_ng"]
    k_pass, k_miss = int((nor < th["합격선"]).sum()), int((ng < th["합격선"]).sum())
    res["3단_판정(보장 기준선)"] = {
        "합격선": th["합격선"], "불합격선": th["불합격선"],
        "정상_합격": {"수": f"{k_pass}/{len(nor)}", "비율": round(k_pass / len(nor), 4), "정확구간": exact_ci(k_pass, len(nor))},
        "실제불량_놓침": {"수": f"{k_miss}/{len(ng)}", "비율": round(k_miss / len(ng), 4), "정확구간": exact_ci(k_miss, len(ng))}}

    json.dump(res, open(OUT / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    for rule in ["center", "iou50"]:
        print(f"[{rule}]")
        for n in MODELS:
            v = res["실제이물"][rule][n]
            print(f"  {n}: F1 {v['F1']} {v['F1_95%구간(사진 단위)']} / 묶음 {v['F1_95%구간(호기·날짜 묶음)']} · 재현율 {v['재현율']} {v['재현율_정확구간(이물 단위)']}")
        print("  차이", res["실제이물"][rule]["차이(최종 YOLO − 비교)"])
    print("합성", {k: v for k, v in res["합성_저대비_이물"].items()})
    print("판정", res["3단_판정(보장 기준선)"])


def seeds(names, data, man):
    """시드만 다른 모델들: 박스 지표, 합성 검출률, 이미지 단위 분리."""
    from ultralytics import YOLO
    from sklearn.metrics import roc_auc_score
    from train_yolo import weights_path

    ids = man.index[man["split"] == "test"].tolist()
    rows = []
    for n in names:
        m = json.load(open(ROOT / f"results/yolo_{n}/metrics.json", encoding="utf-8"))
        thr = m["thresholds"]["F1최대"]
        c, i50 = m["metrics"]["test/F1최대/center"], m["metrics"]["test/F1최대/iou50"]
        pt = pd.read_csv(ROOT / f"results/yolo_{n}/pred_test.csv")
        row = {"모델": n, "시드": m["params"]["seed"], "기준선": round(thr, 4), "test_F1": c["F1"], "test_재현율": c["recall"],
               "test_정밀도": c["precision"], "test_놓침": c["FN"], "test_오검출": c["FP"], "test_AP": c["AP"], "test_F1_iou50": i50["F1"],
               "실제이물_최저점수": round(float(pt.loc[pt["tp"] == 1, "score"].min()), 4)}
        sf = ROOT / f"results/synth_eval_{n}/summary.json"
        if sf.exists():
            s = json.load(open(sf, encoding="utf-8"))["YOLO"]
            row.update({"합성_검출률": s["검출률_전체"], "합성_오검출_영상당": s["오검출_영상당"]})
        model = YOLO(str(weights_path(n)))
        top = {}
        for kind, folder in [("불량", data / "clean/images"), ("정상", data / "normal/test")]:
            res = model.predict([str(folder / f"{i}.png") for i in ids], imgsz=640, conf=0.001, max_det=100, verbose=False)
            top[kind] = np.array([float(r.boxes.conf.max()) if len(r.boxes) else 0.0 for r in res])
        row.update({"정상_기준선아래": f"{int((top['정상'] < thr).sum())}/{len(ids)}",
                    "불량_기준선이상": f"{int((top['불량'] >= thr).sum())}/{len(ids)}",
                    "정상_최고점수_최대": round(float(top["정상"].max()), 4), "불량_최고점수_최소": round(float(top["불량"].min()), 4),
                    "이미지_AUROC": round(float(roc_auc_score(np.r_[np.ones(len(ids)), np.zeros(len(ids))],
                                                             np.r_[top["불량"], top["정상"]])), 4)})
        rows.append(row)
    df = pd.DataFrame(rows)
    num = [c for c in df.columns if c not in ("모델", "시드", "정상_기준선아래", "불량_기준선이상")]
    spread = {c: {"최소": float(df[c].min()), "최대": float(df[c].max()), "평균": round(float(df[c].mean()), 4),
                  "표준편차": round(float(df[c].std(ddof=1)), 4)} for c in num}
    json.dump({"모델별": df.to_dict("records"), "범위": spread}, open(OUT / "seeds.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(df.T.to_string())


if __name__ == "__main__":
    main()
