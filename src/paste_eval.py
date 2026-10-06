"""이식 시험편 평가: 진짜 이물 모양의 옅은 이물을 얼마나 찾는가.

지금까지의 저대비 평가(synth.py)는 구 모양 합성 이물이라, "구 모양으로 학습한 모델이 구 모양 시험을 잘 보는 것"일 수 있다.
여기서는 test 사진의 실제 이물 점을 잘라(augment_paste.build_bank) 같은 호기의 다른 test 사진에 세기를 낮춰 옮겨 붙인다.
모양은 진짜이고, 점도 사진도 어떤 모델의 학습에도 쓰인 적이 없다.

  - 세기 s 격자 × 사진당 PER_IMAGE 개 × VARIANTS 장. 자리는 제품 안 아무 곳 (실제 이물·서로 간 거리 유지, synth.py 와 같다).
  - 넣은 뒤 대비를 실제 이물과 같은 방식(defect_stats.measure)으로 잰다 → 구 합성 평가와 같은 축에서 비교.
  - 검출 = 각 모델이 val 에서 정한 기준선(F1 최대) 이상 예측의 중심이 이물 박스(±2px) 안 (synth_eval.py 와 같다).

실행: .venv\\Scripts\\python.exe src\\paste_eval.py --models ratio0_e100 ratio3_e100 paste3_e100 mix3_e100 cnn_aug
결과: data/paste_test/ (평가셋), results/paste_eval/ (summary.json, defects_scored.csv, rate_by_contrast.png)

  --source ref : 지운 배경 대신 '인접 프레임'을 기준으로 떼어 낸 조각을, 원래 자리와 둘레 구조가 닮은 자리에 넣는다
                 (reference_residual.py). 조각은 val·test 사진에서만 떼고(학습에 쓰인 적 없음) test 사진에 넣는다.
                 결과: data/ref_test/, results/paste_eval_ref/
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
import reference_residual as RR
from augment_paste import build_bank, paste
from defect_stats import measure
from prepare import product_mask
from synth import band_mask
from synth_eval import baseline_preds, score_defects, yolo_preds

ROOT = Path(__file__).resolve().parents[1]
STRENGTHS = [0.08, 0.12, 0.16, 0.22, 0.30, 0.40, 0.55, 0.75, 1.0]     # 세기 s 격자. 1.0 이 원래 이물의 세기다
PER_IMAGE = 4            # 사진 한 장에 넣는 이물 수
VARIANTS = 12            # 원본 사진 한 장으로 만드는 시험편 사진 수
BOX = 10                 # 채점할 때 이물 박스의 한 변 (px). 반폭 5px 를 score_defects 에 넘긴다
SEED_OFFSET = 32452843   # 설정의 시드에 더해 이 평가셋만의 난수열을 만든다
BINS = [0, 0.04, 0.06, 0.08, 0.10, 0.13, 0.16, 0.20, 0.25, 0.30, 0.40, 0.70]     # synth_eval.py 와 같은 구간


def build(data, cfg):
    """이식 시험편 평가셋을 만든다 (data/paste_test). test 사진의 실제 이물 점을 같은 호기의 다른 test 사진에 옮겨 붙인다.

    images/<원본 id>__tNN.png 와 함께 아래 파일을 쓴다.
      defects.csv : img src machine cx cy s(세기) donor(점을 떼어 온 사진) donor_depth(조각의 가장 어두운 정도)
                    in_band c_meas(넣은 뒤 잰 대비) area_meas
      images.csv  : img src machine w h,   config.json : 설정값과 장수 · 개수
    """
    out = data / "paste_test"
    (out / "images").mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(data / "manifest.csv")
    man = man[man["labeled"] & (man["split"] == "test")]
    bank = build_bank(data, man, cfg["seed"] + 1)       # 점 은행: {호기: [(사진 id, 투과율 조각)]}
    rows, img_rows = [], []
    for r in tqdm(list(man.itertuples()), desc="이식 시험편"):
        g = np.asarray(Image.open(data / "clean/images" / f"{r.id}.png"))
        h, w = g.shape
        pm = product_mask(g)
        band = band_mask(g, pm)
        # 실제 이물 박스와 그 둘레 10px 에는 넣지 않는다. 넣을 수 있는 자리 = 제품 안에서 그 밖의 화소
        forbid = np.zeros_like(pm)
        for b in np.loadtxt(data / "clean/labels" / f"{r.id}.txt", ndmin=2):
            cx, cy, bw, bh = b[1] * w, b[2] * h, b[3] * w, b[4] * h
            forbid[max(0, int(cy - bh / 2 - 10)):int(cy + bh / 2 + 10),
                   max(0, int(cx - bw / 2 - 10)):int(cx + bw / 2 + 10)] = 1
        ys, xs = np.nonzero(pm & (forbid == 0))
        donors = [d for d in bank[int(r.machine)] if d[0] != r.id]     # 같은 호기의 다른 사진에서 떼어 낸 점만 쓴다
        for v in range(VARIANTS):
            # 난수는 (시드, 목록에 적힌 사진 sha1 의 앞 8자리, 변형 번호)로 정한다. 사진 순서가 바뀌어도 같은 시험편이 나온다
            rng = np.random.default_rng([cfg["seed"] + SEED_OFFSET, int(r.sha1[:8], 16), v])
            f = g.astype(np.float32)
            sid = f"{r.id}__t{v:02d}"
            placed, new = [], []
            for _ in range(PER_IMAGE):
                for _try in range(100):                     # 먼저 넣은 이물과 24px 이상 떨어진 자리가 나올 때까지 최대 100번 다시 뽑는다
                    k = rng.integers(len(xs))
                    x, y = int(xs[k]), int(ys[k])
                    if all(np.hypot(x - px, y - py) >= 24 for px, py in placed):
                        break
                did, T = donors[rng.integers(len(donors))]
                s = STRENGTHS[rng.integers(len(STRENGTHS))]
                if not paste(f, x, y, T, s, rng):           # 조각이 영상 밖으로 나가는 자리면 이 이물은 넣지 않는다
                    continue
                placed.append((x, y))
                # 좌표는 화소 중심(+0.5)으로 적는다. donor_depth = 1 - 조각의 최소 투과율
                new.append(dict(img=sid, src=r.id, machine=r.machine, cx=x + 0.5, cy=y + 0.5, s=s, donor=did,
                                donor_depth=round(float(1 - T.min()), 3), in_band=bool(band[y, x])))
            img = np.clip(f.round(), 0, 255).astype(np.uint8)
            Image.fromarray(img).save(out / "images" / f"{sid}.png")
            for row in new:                                 # 저장한 8비트 영상에서 대비를 잰다 (실제 이물과 같은 측정 방식)
                m = measure(img, row["cx"] - 0.5, row["cy"] - 0.5)
                row.update(c_meas=m["contrast"], area_meas=m["area"])
            rows += new
            img_rows.append(dict(img=sid, src=r.id, machine=r.machine, w=w, h=h))
    pd.DataFrame(rows).to_csv(out / "defects.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(img_rows).to_csv(out / "images.csv", index=False, encoding="utf-8-sig")
    json.dump(dict(strengths=STRENGTHS, per_image=PER_IMAGE, variants=VARIANTS, box=BOX, n_images=len(img_rows),
                   n_defects=len(rows), bank={str(k): len(v) for k, v in bank.items()}),
              open(out / "config.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(len(img_rows), "장,", len(rows), "개, 은행", {k: len(v) for k, v in bank.items()})


def build_ref(data, cfg):
    """인접 프레임 기준 조각 + 구조 맞춤 자리로 만든 시험편 (data/ref_test).

    파일 구성은 build() 와 같고 사진 이름은 <원본 id>__rNN 이다. defects.csv 에 cost(자리 맞춤 비용) 열이 더 있고,
    조각을 뗄 때 쓴 기준 프레임 목록을 references.csv 로 남긴다.
    """
    out = data / "ref_test"
    (out / "images").mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(data / "manifest.csv")
    man = man[man["labeled"]]
    # 조각은 val · test 사진에서 떼고, 넣는 곳은 test 사진뿐이다
    bank, refs = RR.build_ref_bank(data, man[man["split"].isin(["val", "test"])])
    test = man[man["split"] == "test"]
    rows, img_rows = [], []
    for r in tqdm(list(test.itertuples()), desc="기준 프레임 시험편"):
        g = np.asarray(Image.open(data / "clean/images" / f"{r.id}.png"))
        h, w = g.shape
        pm = product_mask(g)
        band = band_mask(g, pm)
        real = RR.load_boxes(data, r.id, w, h)
        donors = [d for d in bank.get(int(r.machine), []) if d["id"] != r.id]
        if not donors:
            continue
        picker = RR.ContextPicker(g, pm)
        for v in range(VARIANTS):
            rng = np.random.default_rng([cfg["seed"] + SEED_OFFSET + 1, int(r.sha1[:8], 16), v])
            f = g.astype(np.float32)
            sid = f"{r.id}__r{v:02d}"
            placed, new = [], []
            for _ in range(PER_IMAGE):
                dn = donors[rng.integers(len(donors))]
                s = STRENGTHS[rng.integers(len(STRENGTHS))]
                # 피할 자리 = 실제 이물 박스 + 먼저 넣은 이물의 둘레 12px. 후보 가운데 비용이 낮은 여덟 곳에서 하나를 무작위로 고른다
                taken = [tuple(b) for b in real] + [(px - 12, py - 12, px + 12, py + 12) for px, py in placed]
                cs = picker.candidates(dn["desc"], dn["bw"], dn["bh"], taken)[:8]
                if not cs:
                    continue
                cost, x0, y0 = cs[rng.integers(len(cs))]
                if not RR.apply(f, x0, y0, dn["patch"], s):
                    continue
                cx, cy = x0 + dn["bw"] / 2, y0 + dn["bh"] / 2
                placed.append((cx, cy))
                new.append(dict(img=sid, src=r.id, machine=r.machine, cx=cx, cy=cy, s=s, donor=dn["id"], cost=round(cost, 3),
                                donor_depth=round(float(1 - dn["patch"].min()), 3), in_band=bool(band[int(cy), int(cx)])))
            if not new:
                continue
            img = np.clip(f.round(), 0, 255).astype(np.uint8)
            Image.fromarray(img).save(out / "images" / f"{sid}.png")
            for row in new:
                m = measure(img, row["cx"] - 0.5, row["cy"] - 0.5)
                row.update(c_meas=m["contrast"], area_meas=m["area"])
            rows += new
            img_rows.append(dict(img=sid, src=r.id, machine=r.machine, w=w, h=h))
    pd.DataFrame(rows).to_csv(out / "defects.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(img_rows).to_csv(out / "images.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(refs).to_csv(out / "references.csv", index=False, encoding="utf-8-sig")
    json.dump(dict(strengths=STRENGTHS, per_image=PER_IMAGE, variants=VARIANTS, box=BOX, n_images=len(img_rows),
                   n_defects=len(rows), bank={str(k): len(v) for k, v in bank.items()}, n_references=len(refs)),
              open(out / "config.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(len(img_rows), "장,", len(rows), "개, 은행", {k: len(v) for k, v in bank.items()})


def main():
    """평가셋이 없으면 만들고, 규칙 기반과 --models 의 모델을 각자의 기준선으로 채점해 요약 · 표 · 그림을 저장한다.

    defects_scored.csv 는 defects.csv 에 모델마다 <이름>_score(이물 자리의 최고 점수) · <이름>_hit(기준선 이상인지) 열을 붙인 것이다.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["ratio0_e100", "ratio3_e100", "paste3_e100", "mix3_e100", "cnn_aug"])
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--source", default="paste", choices=["paste", "ref"])
    args = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "configs" / "data.yaml", encoding="utf-8"))
    data = ROOT / cfg["out_dir"]
    syn = data / f"{args.source}_test"
    out = ROOT / "results" / ("paste_eval" if args.source == "paste" else "paste_eval_ref")
    out.mkdir(parents=True, exist_ok=True)
    if args.rebuild or not (syn / "defects.csv").exists():
        (build if args.source == "paste" else build_ref)(data, cfg)

    imgs = pd.read_csv(syn / "images.csv")
    defects = pd.read_csv(syn / "defects.csv")
    man = pd.read_csv(data / "manifest.csv").set_index("id")
    # 원본 사진의 실제 이물 박스 (YOLO 형식 비율 → xyxy 픽셀). 채점 함수에 함께 넘긴다
    real = {}
    for s in imgs["src"].unique():
        b = np.loadtxt(data / "clean/labels" / f"{s}.txt", ndmin=2)
        w, h = man.loc[s, "w"], man.loc[s, "h"]
        real[s] = np.stack([(b[:, 1] - b[:, 3] / 2) * w, (b[:, 2] - b[:, 4] / 2) * h,
                            (b[:, 1] + b[:, 3] / 2) * w, (b[:, 2] + b[:, 4] / 2) * h], 1)
    paths = [syn / "images" / f"{i}.png" for i in imgs["img"]]
    ids, machines = imgs["img"].tolist(), imgs["machine"].tolist()

    # preds = {이름: (예측 표, 그 모델의 기준선)}. 기준선은 모델마다 val 에서 정한 F1 최대 점수를 쓴다
    bl = json.load(open(ROOT / "results/baseline_clean/metrics.json", encoding="utf-8"))
    preds = {"베이스라인": (baseline_preds(bl["params"], paths, ids, machines), bl["thresholds"]["F1최대"])}
    for m in args.models:
        if not C.metrics_file(m).exists():
            print("없음, 건너뜀:", m)
            continue
        thr = json.load(open(C.metrics_file(m), encoding="utf-8"))["thresholds"]["F1최대"]
        preds[m] = (yolo_preds(m, paths, ids, machines=machines), thr)

    defects["c_bin"] = pd.cut(defects["c_meas"], BINS)
    # 실제 이물의 측정 대비 5% · 95% 지점. 이보다 옅은 시험편(c_meas < lo)과 실제 수준(lo~hi)을 나눠 본다
    real_c = pd.read_csv(ROOT / "results/defect_stats/real_defects.csv")["contrast"]
    lo, hi = float(real_c.quantile(.05)), float(real_c.quantile(.95))
    summary = {"이물수": len(defects), "사진수": len(imgs), "실제이물_측정대비(5~95%)": [round(lo, 3), round(hi, 3)]}
    for name, (pred, thr) in preds.items():
        best, fp = score_defects(pred, defects, real, thr, BOX / 2)
        defects[f"{name}_score"], defects[f"{name}_hit"] = best, best >= thr
        hit = defects[f"{name}_hit"]
        faint = defects["c_meas"] < lo
        summary[name] = {"기준선": round(float(thr), 4), "검출률_전체": round(float(hit.mean()), 4),
                         "검출률_실제보다옅음(<5%지점)": round(float(hit[faint].mean()), 4),
                         "검출률_실제수준(5~95%)": round(float(hit[(defects["c_meas"] >= lo) & (defects["c_meas"] <= hi)].mean()), 4),
                         "검출률_원본세기(s=1)": round(float(hit[defects["s"] == 1.0].mean()), 4),
                         "호기별": {int(k): round(float(v), 4) for k, v in hit.groupby(defects["machine"]).mean().items()},
                         "오검출_영상당": round(float(np.mean(list(fp.values()))), 4),
                         "구간별": {str(k): round(float(v), 3) for k, v in hit.groupby(defects["c_bin"], observed=True).mean().items()}}
        # 같은 모델의 구 합성 평가 (같은 측정 대비 구간) 과 나란히
        sf = ROOT / "results" / f"synth_eval_{name}" / "defects_scored.csv"
        if sf.exists():
            d = pd.read_csv(sf)
            col = "CNN_hit" if C.is_cnn(name) else "YOLO_hit"
            summary[name]["구합성_검출률_전체"] = round(float(d[col].mean()), 4)
            summary[name]["구합성_구간별"] = {str(k): round(float(v), 3) for k, v in
                                        d[col].groupby(pd.cut(d["c_meas"], BINS), observed=True).mean().items()}
    summary["구간별_이물수"] = {str(k): int(v) for k, v in defects.groupby("c_bin", observed=True).size().items()}
    defects.drop(columns=["c_bin"]).to_csv(out / "defects_scored.csv", index=False, encoding="utf-8-sig")
    json.dump(summary, open(out / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # 그림: 측정 대비 구간별 검출률. 가로 자리는 구간의 가운데 값, 노란 띠는 실제 이물의 대비 범위
    plt.rcParams["font.family"] = "Malgun Gothic"
    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=150)
    curve = defects.groupby("c_bin", observed=True)[[f"{n}_hit" for n in preds]].mean()
    mid = [(b.left + b.right) / 2 for b in curve.index]
    ax.axvspan(lo, hi, color="#f2c14e", alpha=.25, label="실제 이물 대비 (5~95%)")
    for n in preds:
        ax.plot(mid, curve[f"{n}_hit"], "o-", lw=1.8, ms=3.5, label=n)
    ax.set(xlabel="측정 대비 (주변 대비 어두운 비율)", ylabel="검출률", ylim=(0, 1.03), xlim=(0, 0.6))
    ax.grid(alpha=.3)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "rate_by_contrast.png")

    for n in preds:
        s = summary[n]
        print(n, {k: s[k] for k in ["기준선", "검출률_전체", "검출률_실제보다옅음(<5%지점)", "검출률_실제수준(5~95%)", "오검출_영상당"]},
              "구합성", s.get("구합성_검출률_전체"))


if __name__ == "__main__":
    main()
