"""합성 저대비 이물 평가: 베이스라인과 YOLO가 이물 대비·크기에 따라 어디서부터 놓치는가.

합성 이물 하나마다 '판정 임계값 이상 예측의 중심이 이물 박스(±2px) 안에 있으면 검출'로 본다.
임계값은 각 모델이 val에서 정한 F1 최대 값을 그대로 쓴다 (합성셋으로 다시 맞추지 않는다).
오검출 = 합성·실제 이물 어느 쪽에도 닿지 않은 판정 임계값 이상 예측.

실행: .venv\\Scripts\\python.exe src\\synth_eval.py --yolo y26s_640
결과: results/synth_eval/ (이물별 점수표, 조건별 검출률표, 히트맵, 대비 곡선)
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import baseline as B
import cnn as C
from train_yolo import weights_path

ROOT = Path(__file__).resolve().parents[1]
MARGIN = 2           # 예측 중심이 이물 박스에서 벗어나도 봐주는 여유 (px)


def yolo_preds(name, paths, ids, imgsz=640, batch=32, machines=None):
    """모델 name 으로 영상들을 예측해 박스 중심과 점수를 모은다. 이름은 yolo 지만 CNN 이면 CNN 으로 돌린다.

    paths: 영상 경로 목록, ids: 같은 순서의 영상 이름, machines: 호기 번호(CNN 만 씀, 호기별 박스 크기).
    반환: 열 img, px, py, score 인 표. px, py 는 박스 중심(픽셀), score 는 신뢰도.
    """
    # runs/<name>/cnn.pt 가 있으면 조각 분류 CNN, 없으면 YOLO 가중치로 본다
    if C.is_cnn(name):
        d = C.predict_paths(name, paths, ids, machines)
        return pd.DataFrame({"img": d["id"], "px": (d.x0 + d.x1) / 2, "py": (d.y0 + d.y1) / 2, "score": d["score"]})
    from ultralytics import YOLO
    model = YOLO(str(weights_path(name)))
    rows = []
    for k in tqdm(range(0, len(paths), batch), desc=f"YOLO {name}"):
        # conf 를 0.001 로 낮춰 낮은 점수 예측까지 받는다. 판정 임계값은 채점할 때 따로 적용한다
        res = model.predict([str(p) for p in paths[k:k + batch]], imgsz=imgsz, conf=0.001,
                            max_det=100, verbose=False)
        for i, r in zip(ids[k:k + batch], res):
            for (x0, y0, x1, y1), s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                rows.append((i, (x0 + x1) / 2, (y0 + y1) / 2, float(s)))
    return pd.DataFrame(rows, columns=["img", "px", "py", "score"])


def baseline_preds(prm, paths, ids, machines):
    """규칙 기반(black top-hat) 베이스라인으로 영상들을 예측한다.

    prm: baseline_clean/metrics.json 의 params (구조요소 se, 평활 sigma, 점수 방식 score, 호기별 박스 크기).
    반환: yolo_preds 와 같은 열 img, px, py, score.
    """
    # json 은 키가 문자열이라 호기 번호를 정수로 되돌린다
    boxes = {int(k): v for k, v in prm["box_by_machine"].items()}
    rows = []
    for p, i, m in tqdm(list(zip(paths, ids, machines)), desc="베이스라인"):
        d = B.detect(np.asarray(Image.open(p)), prm["se"], prm["sigma"], prm["score"], boxes[int(m)])
        for r in d.itertuples():
            rows.append((i, (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2, float(r.score)))
    return pd.DataFrame(rows, columns=["img", "px", "py", "score"])


def score_defects(pred, defects, real_boxes, thr, half):
    """이물별 최고 점수·검출 여부, 영상별 오검출 수.

    pred: 열 img, px, py, score. defects: 합성 이물 표(열 img, src, cx, cy). real_boxes: {원본 영상 이름: (N,4) xyxy 픽셀}.
    thr: 판정 임계값, half: 이물 박스 반폭(px).
    반환: (best, fp). best 는 defects 행 순서의 근처 예측 최고 점수(없으면 0), fp 는 {영상 이름: 오검출 수}.
    """
    best = np.zeros(len(defects))
    fp = {}
    by_img = dict(tuple(pred.groupby("img")))
    for img, grp in defects.groupby("img"):
        q = by_img.get(img)
        if q is None:
            # 예측이 하나도 없는 영상: 이물 점수는 0 으로 남고 오검출도 0
            fp[img] = 0
            continue
        used = np.zeros(len(q), bool)   # 합성 이물이나 실제 이물에 닿은 예측 표시
        for idx, r in grp.iterrows():
            # 이물 중심에서 가로·세로 모두 half + MARGIN 안에 중심이 든 예측. 일대일 배정은 하지 않는다
            near = (np.abs(q["px"] - r.cx) <= half + MARGIN) & (np.abs(q["py"] - r.cy) <= half + MARGIN)
            if near.any():
                best[defects.index.get_loc(idx)] = q.loc[near, "score"].max()
                used |= near.to_numpy()
        # 원본 영상에 있던 실제 이물에 맞은 예측도 오검출에서 뺀다 (한 영상의 합성 이물은 모두 같은 원본에서 나옴)
        for x0, y0, x1, y1 in real_boxes[grp["src"].iloc[0]]:
            used |= ((q["px"] >= x0 - MARGIN) & (q["px"] <= x1 + MARGIN) &
                     (q["py"] >= y0 - MARGIN) & (q["py"] <= y1 + MARGIN)).to_numpy()
        fp[img] = int(((q["score"] >= thr).to_numpy() & ~used).sum())
    return best, fp


def heatmap(ax, tab, title):
    """검출률 표(행 = 명목 대비 c0, 열 = 지름 px, 값 0~1)를 칸마다 % 숫자를 적은 히트맵으로 그린다. 반환: imshow 객체."""
    # origin="lower": 옅은 대비가 아래, 진한 대비가 위로 가게 한다
    im = ax.imshow(tab.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto", origin="lower")
    ax.set_xticks(range(tab.shape[1]), [f"{c:g}" for c in tab.columns])
    ax.set_yticks(range(tab.shape[0]), [f"{c:g}" for c in tab.index])
    for (i, j), v in np.ndenumerate(tab.to_numpy()):
        ax.text(j, i, f"{v * 100:.0f}", ha="center", va="center", fontsize=7.5,
                color="white" if v > 0.6 else "#1d2733")
    ax.set(xlabel="이물 지름 (px)", ylabel="명목 대비 c0", title=title)
    return im


def main():
    """합성 평가셋을 베이스라인과 지정 모델로 예측해 대비·크기별 검출률 표와 그림을 저장한다."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "data.yaml"))
    ap.add_argument("--yolo", default="y26s_640")
    ap.add_argument("--set", default="synth", help="평가셋 폴더: synth(잡음 보정 없음) / synth_n(잡음 보정)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    data = ROOT / cfg["out_dir"]
    syn = data / args.set
    # 기본 모델 결과는 synth_eval/, 다른 모델은 synth_eval_<이름>/ 에 따로 둔다
    out = ROOT / "results" / ("synth_eval" if args.yolo == "y26s_640" else f"synth_eval_{args.yolo}")
    if args.set != "synth":
        out = out.with_name(out.name + f"__{args.set}")
    out.mkdir(parents=True, exist_ok=True)
    scfg = json.load(open(syn / "config.json", encoding="utf-8"))
    half = scfg["box"] / 2      # 합성할 때 쓴 채점용 박스 한 변의 절반 (px)

    imgs = pd.read_csv(syn / "images.csv")
    defects = pd.read_csv(syn / "defects.csv")
    man = pd.read_csv(data / "manifest.csv").set_index("id")
    # 원본 영상의 실제 이물 박스: YOLO 라벨(class cx cy w h, 0~1 비율)을 픽셀 xyxy 로 바꾼다
    real_boxes = {}
    for s in imgs["src"].unique():
        b = np.loadtxt(data / "clean/labels" / f"{s}.txt", ndmin=2)
        w, h = man.loc[s, "w"], man.loc[s, "h"]
        real_boxes[s] = np.stack([(b[:, 1] - b[:, 3] / 2) * w, (b[:, 2] - b[:, 4] / 2) * h,
                                  (b[:, 1] + b[:, 3] / 2) * w, (b[:, 2] + b[:, 4] / 2) * h], 1)
    paths = [syn / "images" / f"{i}.png" for i in imgs["img"]]

    bl = json.load(open(ROOT / "results/baseline_clean/metrics.json", encoding="utf-8"))
    yo = json.load(open(C.metrics_file(args.yolo), encoding="utf-8"))
    lab = "CNN" if C.is_cnn(args.yolo) else "YOLO"
    # 모델 이름 → (예측 표, 판정 임계값). 임계값은 각 모델이 val 에서 정한 F1 최대 값이다
    models = {
        "베이스라인": (baseline_preds(bl["params"], paths, imgs["img"].tolist(), imgs["machine"].tolist()),
                    bl["thresholds"]["F1최대"]),
        lab: (yolo_preds(args.yolo, paths, imgs["img"].tolist(), machines=imgs["machine"].tolist()),
              yo["thresholds"]["F1최대"]),
    }

    summary = {}
    for name, (pred, thr) in models.items():
        best, fp = score_defects(pred, defects, real_boxes, thr, half)
        defects[f"{name}_score"] = best
        defects[f"{name}_hit"] = best >= thr
        summary[name] = {"thr": thr, "검출률_전체": round(float((best >= thr).mean()), 4),
                         "오검출_영상당": round(float(np.mean(list(fp.values()))), 4),
                         "오검출_합계": int(sum(fp.values()))}
    defects.to_csv(out / "defects_scored.csv", index=False, encoding="utf-8-sig")

    # 조건별 검출률 표: 행 = 명목 대비 c0, 열 = 지름 d, 값 = 그 칸 이물의 검출 비율
    tabs = {n: defects.pivot_table(index="c0", columns="d", values=f"{n}_hit", aggfunc="mean") for n in models}
    for n, t in tabs.items():
        t.round(3).to_csv(out / f"rate_{n}.csv", encoding="utf-8-sig")

    # 측정 대비 구간별 검출률 (실제 이물과 같은 방식으로 잰 대비)
    bins = [0, 0.04, 0.06, 0.08, 0.10, 0.13, 0.16, 0.20, 0.25, 0.30, 0.40, 0.70]
    defects["c_bin"] = pd.cut(defects["c_meas"], bins)
    curve = defects.groupby("c_bin", observed=True)[[f"{n}_hit" for n in models]].mean()
    curve["n"] = defects.groupby("c_bin", observed=True).size()
    curve.round(3).to_csv(out / "rate_by_measured_contrast.csv", encoding="utf-8-sig")
    for n in models:
        summary[n]["구간별_검출률"] = {str(k): round(float(v), 3) for k, v in curve[f"{n}_hit"].items()}
    # 실제 이물의 측정 대비 분포(5 · 50 · 95 백분위)를 함께 적어 합성 대비 구간과 견준다
    real = pd.read_csv(ROOT / "results/defect_stats/real_defects.csv")
    summary["실제이물_측정대비"] = {k: round(float(real["contrast"].quantile(q)), 3)
                            for k, q in [("p05", .05), ("p50", .5), ("p95", .95)]}
    json.dump(summary, open(out / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # 그림 1: 모델별 검출률 히트맵 (왼쪽 베이스라인, 오른쪽 지정 모델)
    plt.rcParams["font.family"] = "Malgun Gothic"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), dpi=150)
    for ax, (n, t) in zip(axes, tabs.items()):
        im = heatmap(ax, t, f"{n} 검출률(%)")
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.savefig(out / "heatmap.png", bbox_inches="tight")

    # 그림 2: 측정 대비 구간별 검출률 곡선. 가로 위치는 구간의 가운데 값, 노란 띠는 실제 이물 대비의 5~95% 범위
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    mid = [(b.left + b.right) / 2 for b in curve.index]
    ax.axvspan(real["contrast"].quantile(.05), real["contrast"].quantile(.95), color="#f2c14e", alpha=.25,
               label="실제 이물 대비 (5~95%)")
    for n, c in [("베이스라인", "#9aa5b1"), (lab, "#1f5fa8")]:
        ax.plot(mid, curve[f"{n}_hit"], "o-", color=c, lw=2, ms=4, label=n)
    ax.set(xlabel="측정 대비 (주변 대비 어두운 비율)", ylabel="검출률", ylim=(0, 1.03), xlim=(0, 0.6))
    ax.grid(alpha=.3)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "rate_by_contrast.png")

    print(json.dumps({n: {k: v for k, v in s.items() if k != "구간별_검출률"} for n, s in summary.items()},
                     ensure_ascii=False))
    print(curve.round(3).to_string())
    for n, t in tabs.items():
        print(n)
        print((t * 100).round(0).to_string())


if __name__ == "__main__":
    main()
