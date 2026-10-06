"""가상 테스트피스: 호기별 '이 크기·진하기 이상이면 90% 이상 잡는다' 검출 사양표.

X-ray 이물 검사기는 크기를 아는 시험편을 흘려 성능을 확인한다. 같은 발상으로, 이물이 없는 가짜 정상
사진(data/normal/test, 시험 분할 73장)에 Beer–Lambert 합성 이물(구, 지름 d · 명목 진하기 c0)을
넣어 칸마다 충분한 수를 채운다. 합성 평가셋(synth.py)은 칸마다 약 42개라 사양을 말하기엔 적다.

  - 칸 = (호기, 지름, 진하기). 호기마다 칸당 PER_CELL 개, 한 장에 PER_IMAGE 개, 서로 24px 이상 떨어뜨림
  - 검출 = 판정 임계값(각 모델이 val에서 정한 F1 최대) 이상 예측의 중심이 이물 ±(5+2)px 안
  - 사양 = 검출률의 윌슨 95% 신뢰구간 하한이 90% 이상인 가장 옅은 진하기 (그보다 진한 칸도 모두 만족)
  - 가장자리: 지름 2px · 진하기 0.30 칸에서 제품 경계까지 거리별 검출률
  - 헛경보: 합성 이물 어디에도 닿지 않은 임계값 이상 예측 (가짜 정상 위라 곧 정상 부위 헛경보)

실행: .venv\\Scripts\\python.exe src\\testpiece.py --yolo ratio3_e100 --cnn cnn_aug
결과: data/testpiece/ (영상·목록), results/testpiece/ (defects.csv, spec.csv, spec.json, 그래프)
"""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cnn as C
from defect_stats import measure
from prepare import product_mask
from synth import band_mask, insert
from synth_eval import baseline_preds, score_defects, yolo_preds

ROOT = Path(__file__).resolve().parents[1]
DIAMETERS = [1.0, 1.5, 2.0, 3.0]                               # 시험편 지름 4종 (px)
CONTRASTS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.55, 0.70]   # 명목 진하기 c0 8종 (0~1, 중심에서 줄어드는 투과율 비율)
PER_CELL = 150               # 호기마다 (지름, 진하기) 한 칸에 넣는 시험편 수
PER_IMAGE = 4                # 사진 한 장에 넣는 시험편 수
TARGET = 0.90                # 사양 기준 검출률
EDGE_CELL = (2.0, 0.30)      # 가장자리 분석에 쓰는 시험편
SEED_OFFSET = 31337          # 시험편 배치 난수의 시드에 덧붙이는 고정값
HALF = 5                     # 채점 박스 반폭 (synth.py BOX=10 과 같음)


def wilson_low(k, n, z=1.96):
    """n 개 중 k 개를 검출했을 때 검출률의 윌슨 신뢰구간 하한(0~1). z=1.96 은 양쪽 95% 구간.

    윌슨 구간은 표본이 적거나 비율이 1 에 가까워도 0~1 을 벗어나지 않는다. n 이 0 이면 0.
    """
    if n == 0:
        return 0.0
    p = k / n
    den = 1 + z * z / n
    # 하한 = (p + z²/2n - z·sqrt(p(1-p)/n + z²/4n²)) / (1 + z²/n)
    return (p + z * z / (2 * n) - z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / den


def build(data, cfg, split="test"):
    """가짜 정상 사진에 시험편을 넣어 data/testpiece[_split] 에 저장. 이미 있으면 다시 만들지 않는다.

    data: 데이터 폴더, cfg: data.yaml 내용(seed 사용), split: test 또는 val.
    반환: (defects, images). defects 열은 img, src, machine, cx, cy, d, c0, in_band, edge_dist, c_meas,
    images 열은 img, src, machine. cx, cy 는 화소 중심이 정수 + 0.5 인 픽셀 좌표.
    """
    out = data / ("testpiece" if split == "test" else f"testpiece_{split}")
    if (out / "defects.csv").exists():
        return pd.read_csv(out / "defects.csv"), pd.read_csv(out / "images.csv")
    (out / "images").mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(data / "manifest.csv").set_index("id")
    srcs = sorted(p.stem for p in (data / "normal" / split).glob("*.png"))
    cells = [(d, c) for d in DIAMETERS for c in CONTRASTS]
    rows, img_rows = [], []
    for m in sorted(man.loc[srcs, "machine"].unique()):
        ms = [s for s in srcs if man.loc[s, "machine"] == m]
        # 호기별로 난수를 따로 둔다. val 은 시드 끝에 1 을 덧붙여 test 와 다른 배치가 나오게 한다
        rng = np.random.default_rng([cfg["seed"], SEED_OFFSET, int(m)] + ([] if split == "test" else [1]))
        # 칸마다 PER_CELL 개씩 만든 목록을 섞어, 한 사진에 여러 칸의 시험편이 섞여 들어가게 한다
        todo = [c for c in cells for _ in range(PER_CELL)]
        rng.shuffle(todo)
        n_img = int(np.ceil(len(todo) / PER_IMAGE))
        cache = {}
        for k in tqdm(range(n_img), desc=f"{m}호기 시험편"):
            src = ms[k % len(ms)]            # 그 호기의 가짜 정상 사진을 돌아가며 배경으로 다시 쓴다
            if src not in cache:
                g = np.asarray(Image.open(data / "normal" / split / f"{src}.png").convert("L"))
                # pm_full: 침식하지 않은 제품 영역(Otsu). 가장자리까지 거리를 재는 데 쓴다
                # pm: 15×15 침식한 제품 영역(prepare.product_mask). 시험편을 놓을 수 있는 자리
                pm_full = cv2.threshold(cv2.GaussianBlur(g, (9, 9), 0), 0, 1,
                                        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
                pm = product_mask(g)
                ys, xs = np.nonzero(pm)
                cache[src] = (g, cv2.distanceTransform(pm_full, cv2.DIST_L2, 3), band_mask(g, pm), xs, ys)
            g, dist, band, xs, ys = cache[src]
            f = g.astype(np.float32)
            sid = f"{src}__t{k:04d}"
            placed = []
            for d, c0 in todo[k * PER_IMAGE:(k + 1) * PER_IMAGE]:
                # 제품 안 화소 하나를 뽑고 화소 안 위치(0~1)를 더해 중심을 정한다.
                # 이미 놓은 시험편과 24px 이상 떨어질 때까지 최대 200번 다시 뽑고, 끝까지 못 찾으면 마지막 자리를 쓴다
                for _ in range(200):
                    j = rng.integers(len(xs))
                    cx, cy = xs[j] + rng.random(), ys[j] + rng.random()
                    if all(np.hypot(cx - px, cy - py) >= 24 for px, py in placed):
                        break
                insert(f, cx, cy, d, c0)
                placed.append((cx, cy))
                rows.append(dict(img=sid, src=src, machine=int(m), cx=cx, cy=cy, d=d, c0=c0,
                                 in_band=bool(band[int(cy), int(cx)]), edge_dist=float(dist[int(cy), int(cx)])))
            img = np.clip(f.round(), 0, 255).astype(np.uint8)
            Image.fromarray(img).save(out / "images" / f"{sid}.png")
            # 저장한 8비트 영상에서 실제 이물과 같은 방식으로 대비를 다시 잰다 (measure 는 화소 번호 좌표라 0.5 를 뺀다)
            for (cx, cy), row in zip(placed, rows[-len(placed):]):
                row["c_meas"] = measure(img, cx - 0.5, cy - 0.5)["contrast"]
            img_rows.append(dict(img=sid, src=src, machine=int(m)))
    defects, images = pd.DataFrame(rows), pd.DataFrame(img_rows)
    defects.to_csv(out / "defects.csv", index=False, encoding="utf-8-sig")
    images.to_csv(out / "images.csv", index=False, encoding="utf-8-sig")
    return defects, images


def spec_table(defects, models):
    """호기 · 지름 · 모델별 검출 사양표를 만든다.

    defects: 채점이 끝난 시험편 표(<모델>_hit 열 포함), models: 모델 이름 목록.
    반환 열: machine, d, model, min_c0(사양 진하기, 만족하는 칸이 없으면 None), c10~c70(진하기별 검출률 0~1).
    """
    rows = []
    for (m, d), g in defects.groupby(["machine", "d"]):
        for name in models:
            rates = g.groupby("c0")[f"{name}_hit"].agg(["sum", "count"])      # 진하기별 검출 수와 시험편 수
            low = {c: wilson_low(int(r["sum"]), int(r["count"])) for c, r in rates.iterrows()}
            # 그 진하기와 그보다 진한 칸의 하한이 모두 TARGET 이상인 진하기만 남긴다.
            # 더 진한 칸이 하나라도 못 미치면 그 진하기는 사양이 되지 않는다
            ok = [c for c in CONTRASTS if all(low[x] >= TARGET for x in CONTRASTS if x >= c)]
            rows.append(dict(machine=int(m), d=d, model=name, min_c0=min(ok) if ok else None,
                             **{f"c{int(c * 100):02d}": round(r["sum"] / r["count"], 3) for c, r in rates.iterrows()}))
    return pd.DataFrame(rows)


def main():
    """시험편 세트를 만들고(이미 있으면 읽고) 모델별로 채점해 사양표 · 가장자리 · 띠 요약과 곡선 그림을 저장한다."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "data.yaml"))
    ap.add_argument("--yolo", default="ratio3_e100")
    ap.add_argument("--cnn", default="cnn_aug")
    ap.add_argument("--split", default="test", help="test = 보고용 사양, val = 판정 기준선을 정할 때 쓰는 사양 (시험 사진이 기준 결정에 새지 않게)")
    ap.add_argument("--models", default="규칙기반,CNN,YOLO")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    data = ROOT / cfg["out_dir"]
    # val 사양은 판정 기준선을 정하는 데 쓰므로 모델마다 따로 둔다 (results/testpiece_val_<모델>)
    if args.split == "val":
        out = ROOT / "results" / f"testpiece_val_{args.yolo}"
    else:   # 최종 모델 보고용은 results/testpiece, 다른 모델은 이름을 붙여 따로
        out = ROOT / "results" / ("testpiece" if args.yolo == "ratio3_e100" else f"testpiece_{args.yolo}")
    out.mkdir(parents=True, exist_ok=True)

    defects, images = build(data, cfg, args.split)
    tdir = data / ("testpiece" if args.split == "test" else f"testpiece_{args.split}")
    paths = [tdir / "images" / f"{i}.png" for i in images["img"]]
    ids, machines = images["img"].tolist(), images["machine"].tolist()
    bl = json.load(open(ROOT / "results/baseline_clean/metrics.json", encoding="utf-8"))
    use = args.models.split(",")
    # 모델 표시 이름 → (예측 표, 판정 임계값 = 각 모델의 val F1 최대)
    models = {}
    if "규칙기반" in use:
        models["규칙기반"] = (baseline_preds(bl["params"], paths, ids, machines), bl["thresholds"]["F1최대"])
    for lab, name in [("CNN", args.cnn), ("YOLO", args.yolo)]:
        if lab not in use:
            continue
        thr = json.load(open(C.metrics_file(name), encoding="utf-8"))["thresholds"]["F1최대"]
        models[lab] = (yolo_preds(name, paths, ids, machines=machines), thr)

    # 배경이 가짜 정상이라 실제 이물 박스가 없다. score_defects 에 빈 박스를 넘겨 시험편 밖 예측을 모두 헛경보로 센다
    empty = {s: np.zeros((0, 4)) for s in defects["src"].unique()}
    summary = {"n_images": len(images), "n_defects": len(defects), "per_cell": PER_CELL, "target": TARGET, "models": {}}
    for name, (pred, thr) in models.items():
        best, fp = score_defects(pred, defects, empty, thr, HALF)
        defects[f"{name}_score"] = best
        defects[f"{name}_hit"] = best >= thr
        summary["models"][name] = {"thr": thr, "검출률_전체": round(float((best >= thr).mean()), 4),
                                   "헛경보_영상당": round(float(np.mean(list(fp.values()))), 4),
                                   "헛경보_합계": int(sum(fp.values()))}
    defects.to_csv(out / "defects_scored.csv", index=False, encoding="utf-8-sig")

    spec = spec_table(defects, list(models))
    spec.to_csv(out / "spec.csv", index=False, encoding="utf-8-sig")
    summary["spec"] = {f"{r.model}/{r.machine}호기/d{r.d:g}": r.min_c0 for r in spec.itertuples()}

    # 측정 대비로 환산: 진하기 c0 칸의 실제 측정 대비 중앙값 (호기·지름별)
    conv = defects.groupby(["machine", "d", "c0"])["c_meas"].median().round(3)
    conv.to_csv(out / "c0_to_measured.csv", encoding="utf-8-sig")

    # 가장자리 거리
    # EDGE_CELL 한 칸만 골라 세기를 고정하고, 제품 경계까지 거리(px) 구간별 검출률을 본다
    e = defects[(defects["d"] == EDGE_CELL[0]) & (defects["c0"] == EDGE_CELL[1])].copy()
    e["edge_bin"] = pd.cut(e["edge_dist"], [0, 10, 20, 40, 80, 1000])
    et = e.groupby("edge_bin", observed=True)[[f"{n}_hit" for n in models]].mean()
    et["n"] = e.groupby("edge_bin", observed=True).size()
    et.round(3).to_csv(out / "edge.csv", encoding="utf-8-sig")
    summary["edge"] = {str(k): {n: round(float(v[f"{n}_hit"]), 3) for n in models} | {"n": int(v["n"])}
                       for k, v in et.iterrows()}
    # 어두운 띠 안과 밖의 검출률 (모든 칸을 합쳐서)
    bt = defects.groupby("in_band")[[f"{n}_hit" for n in models]].mean().round(3)
    summary["띠안밖"] = {("띠 안" if k else "띠 밖"): v.to_dict() for k, v in bt.iterrows()}
    json.dump(summary, open(out / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=float)

    # 그래프: 호기(열) × 지름(선), 최종 YOLO 검출률 vs 진하기
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    ms = sorted(defects["machine"].unique())
    fig, axes = plt.subplots(1, len(ms), figsize=(4.2 * len(ms), 3.6), dpi=150, sharey=True)
    cols = ["#b7c9d6", "#7fa6bf", "#3f7fa3", "#0b4f75"]
    for ax, m in zip(axes, ms):
        g = defects[defects["machine"] == m]
        for d, c in zip(DIAMETERS, cols):
            r = g[g["d"] == d].groupby("c0")["YOLO_hit"].mean()
            ax.plot(r.index, r.values, "o-", color=c, lw=1.8, ms=3.5, label=f"지름 {d:g}px")
        ax.axhline(TARGET, color="#c96f24", lw=1, ls="--")     # 사양 기준 검출률 90% 선
        ax.set(title=f"{m}호기", xlabel="명목 진하기 c0", ylim=(0, 1.02))
        ax.grid(alpha=.3)
    axes[0].set_ylabel("검출률 (최종 AI)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "curves.png")

    print(json.dumps({k: v for k, v in summary.items() if k != "spec"}, ensure_ascii=False, indent=1, default=float))
    print(spec.to_string(index=False))


if __name__ == "__main__":
    main()
