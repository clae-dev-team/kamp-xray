"""상시 점검에서 '장비 문제'와 'AI 문제' 가려내기 (IAEA 진단방사선 품질관리의 물리 지표 감시를 AI 점검에 결합).

기준 정상 영상(reference_set.py)에 시험편을 넣어 점검할 때마다 두 가지를 함께 기록한다.
  - AI 검출 여부 (판정 임계값 이상 예측이 시험편 7px 안에 있나)
  - 물리 지표: 시험편 자리의 대비 대 잡음비 CNR = (주변 밝기 − 점 최저 밝기) / 주변 잡음 표준편차
    AI와 무관하게 영상만으로 잰다. 장비가 무뎌지면 CNR이 떨어지고, AI만 고장 나면 CNR은 그대로다.

모의 고장 두 가지 (모두 가정)
  A 장비 열화: 점검 START 번째부터 흐림·잡음이 선형 증가 (monitor.py 와 같은 열화)
  B AI 교체 사고: 장비는 그대로, 점검 START 번째부터 모델 파일이 합성 증강 없이 학습한 이전 모델(--bad)로 바뀜
경보: 최근 WINDOW 개 시험편 중 잡은 수가 초기 구간 기준 관리 하한 미만 (monitor.py 와 같은 규칙)
원인 판정 (IAEA 처럼 기준 영상마다 자기 기준값을 두고 물리 지표 두 가지를 본다):
  CNR_n  = 시험편 CNR / 그 기준 영상의 초기 구간 평균,   잡음_n = 영상 잡음 표준편차 / 그 영상의 초기 구간 평균
  경보 시점 최근 WINDOW 개 평균이 1에서 3 표준오차 넘게 벗어나면(CNR은 낮아질 때, 잡음은 어느 쪽이든) '장비', 아니면 'AI'
  (흐림은 점 대비와 잡음을 함께 줄여 CNR만으로는 잘 안 보여서 잡음 지표를 같이 쓴다.
   처음엔 모든 영상을 한 기준값으로 묶고 CNR만 봐서 장비 고장을 놓쳤다: 고정 자리에서도 z = -1.53)

시험편 자리: 기본(--fixed)은 IAEA 팬텀 점검처럼 기준 영상마다 같은 자리·같은 진하기로 고정한다.
  처음 설계(--random, 매번 다른 자리)는 자리마다 CNR이 크게 달라 장비 고장 때 CNR이 27% 떨어졌는데도
  3 표준오차를 못 넘어(z = -2.79) 'AI'로 잘못 판정했다 → results/diagnose_random 에 비교용으로 남긴다.

실행: .venv\\Scripts\\python.exe src\\diagnose.py --yolo ratio3_e100 --bad ratio0_e100
결과: results/diagnose/ (summary.json, checks.csv, diagnose.png)
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
from scipy.stats import binom
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from monitor import PIECE_D, degrade
from prepare import product_mask
from synth import band_mask, insert
from train_yolo import weights_path

ROOT = Path(__file__).resolve().parents[1]
N_CHECK = 400          # 모의 점검 횟수 (점검 한 번 = 기준 정상 영상 한 장에 시험편 하나)
START = 150            # 고장이 시작되는 점검 번호. 그 전까지가 기준값을 잡는 초기 구간이다
WINDOW = 20            # 경보와 원인 판정에 쓰는 최근 시험편 수 (monitor.py 와 같은 값)
FALSE_ALARM = 0.01     # 관리 하한을 정하는 확률 (monitor.py 와 같은 값)
SEED_OFFSET = 5151     # 설정의 시드와 묶어 이 실험만의 난수열을 만든다
ACQ_NOISE = 1.0        # 촬영할 때마다 새로 생기는 잡음 (회색 단계 표준편차, 가정). 같은 기준 영상도 찍을 때마다 조금씩 다르다


def cnr(g, cx, cy, r_in=5, r_out=9):
    """시험편 자리 (cx, cy) 의 대비 대 잡음비. CNR = (주변 밝기 - 점 최저 밝기) / 주변 잡음 표준편차.

    g : 회색 영상. 주변 = 중심에서 r_in~r_out px 떨어진 네모 고리. 영상 가장자리에서는 고리를 잘라서 잰다. 반환: 실수 하나.
"""
    f = g.astype(np.float32)
    h, w = f.shape
    x0, x1, y0, y1 = max(0, int(cx) - r_out), min(w, int(cx) + r_out + 1), max(0, int(cy) - r_out), min(h, int(cy) + r_out + 1)
    p = f[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.maximum(np.abs(xx - cx), np.abs(yy - cy))        # 가로 · 세로 거리 중 큰 쪽이라 고리가 네모 모양이 된다
    ring = (d >= r_in) & (d <= r_out)
    bg = float(np.median(p[ring]))                          # 주변 밝기: 고리의 중앙값
    lo = float(cv2.blur(p, (2, 2))[d <= 2].min())           # 점 최저 밝기: 2×2 평균을 낸 뒤 중심 2px 안의 최솟값
    hp = f - cv2.GaussianBlur(f, (0, 0), 2)                 # 잡음 = 흐린 영상(σ 2px)을 뺀 고역 성분
    sd = float(hp[y0:y1, x0:x1][ring].std()) + 1e-3         # 0 으로 나누지 않게 0.001 을 더한다
    return (bg - lo) / sd


def run(scenario, args, cfg, refs, spec, models, thr):
    """고장 하나를 모의한다. scenario 가 "장비" 면 START 부터 영상이 열화되고, "AI" 면 START 부터 모델이 --bad 로 바뀐다.

    refs : 기준 정상 영상 목록, spec : 호기별 시험편 진하기, models · thr : {모델 이름: 모델 · 박스 기준선}.
    반환 표의 열: scenario check(점검 번호) ref(기준 영상 번호) level(열화 수준) model hit cnr noise(제품 영역 잡음 표준편차).
"""
    rng = np.random.default_rng([cfg["seed"], SEED_OFFSET, 0 if scenario == "장비" else 1])
    rows = []
    for k in tqdm(range(N_CHECK), desc=scenario):
        ref = refs[k % len(refs)]                           # 기준 정상 영상을 순서대로 돌아가며 쓴다
        if args.random:
            ys, xs = ref["band"]
            j = rng.integers(len(xs))
            cx, cy = xs[j] + rng.random(), ys[j] + rng.random()
        else:
            cx, cy = ref["spot"]
        c0 = float(spec.get(ref["m"], 0.30))                # 그 호기가 사양표에 없으면 0.30
        f = ref["g"].astype(np.float32)
        insert(f, cx, cy, PIECE_D, c0)
        f += rng.normal(0, ACQ_NOISE, f.shape).astype(np.float32)     # 촬영 잡음. 이것이 없으면 고정 자리의 CNR 이 매번 똑같아진다
        img = np.clip(f.round(), 0, 255).astype(np.uint8)
        lv = 0.0
        if scenario == "장비" and k >= START:               # 장비 고장: START 에서 0, 마지막 점검에서 1 이 되게 선형으로 열화
            lv = (k - START) / (N_CHECK - 1 - START)
            img = degrade(img, lv, rng)
        name = args.bad if (scenario == "AI" and k >= START) else args.yolo     # AI 고장: 영상은 그대로 두고 모델만 바꾼다
        r = models[name].predict(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), imgsz=640, conf=thr[name], max_det=100,
                                 verbose=False)[0]
        b = r.boxes.xyxy.cpu().numpy()
        hit = bool(len(b) and (np.hypot((b[:, 0] + b[:, 2]) / 2 - cx, (b[:, 1] + b[:, 3]) / 2 - cy) <= 7).any())
        # 물리 지표 두 가지는 모델과 무관하게 영상만으로 잰다: 시험편 자리의 CNR, 제품 영역 전체의 잡음 표준편차
        hp = img.astype(np.float32) - cv2.GaussianBlur(img.astype(np.float32), (0, 0), 2)
        rows.append(dict(scenario=scenario, check=k, ref=k % len(refs), level=round(lv, 4), model=name, hit=hit,
                         cnr=cnr(img, cx, cy), noise=float(hp[ref["pm"]].std())))
    return pd.DataFrame(rows)


def judge(df):
    """run() 의 점검 기록에서 첫 경보를 찾고, 그 시점의 물리 지표로 원인을 '장비' 또는 'AI' 로 가른다.

    반환 사전: 경보가 없으면 {경보: None, 관리하한, 평상시검출률}.
               있으면 {경보_점검번호, 관리하한, 평상시검출률, 고장시작후_점검수, CNR_z, CNR_최근평균_기준대비,
                       잡음_z, 잡음_최근평균_기준대비, 원인판정}.
"""
    # 경보: 초기 구간 검출률 q 의 이항분포에서 정한 관리 하한보다 최근 WINDOW 개의 잡은 수가 적을 때 (monitor.py 와 같은 규칙)
    cal = df[df["check"] < START]
    q = float(cal["hit"].mean())
    lcl = int(binom.ppf(FALSE_ALARM, WINDOW, q))
    hits = df["hit"].astype(float).rolling(WINDOW).sum()
    al = df[(hits < lcl) & (df["check"] >= START)]
    if not len(al):
        return dict(경보=None, 관리하한=lcl, 평상시검출률=round(q, 3))
    k = int(al["check"].iloc[0])
    out = dict(경보_점검번호=k, 관리하한=lcl, 평상시검출률=round(q, 3), 고장시작후_점검수=k - START)
    equip = False
    # both = 양쪽 방향을 모두 볼지. CNR 은 낮아질 때만, 잡음은 늘든 줄든 장비 변화로 본다 (흐림은 잡음을 줄이고 잡음 증가는 늘린다)
    for col, name, both in [("cnr", "CNR", False), ("noise", "잡음", True)]:
        base = cal.groupby("ref")[col].mean()               # 기준 영상마다 자기 초기 평균을 기준값으로 둔다
        norm = df[col] / df["ref"].map(base)                # 기준값으로 나눈 비율. 평상시에는 1 근처다
        sd = max(float(norm[df["check"] < START].std()), 1e-6)
        recent = norm[(df["check"] > k - WINDOW) & (df["check"] <= k)]      # 경보 시점까지의 최근 WINDOW 개
        # z = (최근 평균 - 1) / 표준오차. 표준오차 = 초기 구간 표준편차 / √WINDOW
        z = (float(recent.mean()) - 1) / (sd / np.sqrt(WINDOW))
        out[f"{name}_z"] = round(z, 2)
        out[f"{name}_최근평균_기준대비"] = round(float(recent.mean()), 3)
        equip |= (abs(z) > 3) if both else (z < -3)
    out["원인판정"] = "장비" if equip else "AI"
    return out


def main():
    """두 고장(장비 · AI)을 차례로 모의하고, 점검 기록(checks.csv) · 판정 요약(summary.json) · 그림(diagnose.png)을 저장한다."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "data.yaml"))
    ap.add_argument("--yolo", default="ratio3_e100")
    ap.add_argument("--bad", default="ratio0_e100", help="AI 교체 사고 때 잘못 올라간 모델 (합성 증강 없이 학습)")
    ap.add_argument("--random", action="store_true", help="시험편 자리를 매번 무작위로 (처음 설계, 비교용)")
    args = ap.parse_args()
    from ultralytics import YOLO
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    out = ROOT / "results" / ("diagnose_random" if args.random else "diagnose")
    out.mkdir(parents=True, exist_ok=True)
    # 모델마다 자기 박스 기준선(val F1 최대)으로 채점한다
    models = {n: YOLO(str(weights_path(n))) for n in [args.yolo, args.bad]}
    thr = {n: json.load(open(ROOT / f"results/yolo_{n}/metrics.json", encoding="utf-8"))["thresholds"]["F1최대"]
           for n in models}
    # 시험편 진하기는 val 테스트피스 사양에서 정한다 (시험 사진으로 만든 사양을 설계값에 쓰지 않는다)
    spec = pd.read_csv(ROOT / f"results/testpiece_val_{args.yolo}/spec.csv")
    spec = spec[(spec["model"] == "YOLO") & (spec["d"] == PIECE_D)].set_index("machine")["min_c0"]
    # 기준 정상 영상: g = 영상, m = 호기, band = 띠 화소 (ys, xs), pm = 제품 영역, spot = 고정 시험편 자리 (x, y, 화소 중심)
    refs = []
    for r in pd.read_csv(ROOT / "results/reference/reference.csv").itertuples():
        g = np.asarray(Image.open(r.path).convert("L"))
        pm = product_mask(g)
        band = np.nonzero(band_mask(g, pm) & (pm > 0))
        rs = np.random.default_rng([cfg["seed"], SEED_OFFSET, len(refs), 9])        # 영상마다 고정 자리 하나
        j = rs.integers(len(band[1]))
        refs.append(dict(g=g, m=int(r.machine), band=band, pm=pm > 0, spot=(band[1][j] + 0.5, band[0][j] + 0.5)))

    dfs = {s: run(s, args, cfg, refs, spec, models, thr) for s in ["장비", "AI"]}
    pd.concat(dfs.values()).to_csv(out / "checks.csv", index=False, encoding="utf-8-sig")
    summary = {"설정": {"점검수": N_CHECK, "고장시작": START, "창": WINDOW, "정상모델": args.yolo, "사고모델": args.bad}}
    for s, df in dfs.items():
        summary[f"{s} 고장"] = judge(df)
        print(s, json.dumps(summary[f"{s} 고장"], ensure_ascii=False))
    json.dump(summary, open(out / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # 그림: 고장마다 한 칸. AI 검출률(최근 WINDOW 개)과 두 물리 지표(영상별 기준값 대비 비율의 이동 평균)를 같은 축에 그린다
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=150, sharey=True)
    for ax, (s, df) in zip(axes, dfs.items()):
        cal = df[df["check"] < START]
        ax.axvspan(START, N_CHECK, color="#f2e3d3", alpha=.6, label=f"{s} 고장 구간 (가정)")
        ax.plot(df["check"], df["hit"].astype(float).rolling(WINDOW).mean(), color="#0a8f86", lw=2,
                label=f"AI 검출률 (최근 {WINDOW}개)")
        for col, c, lab in [("cnr", "#8c5bb5", "물리 지표: 시험편 CNR"), ("noise", "#c96f24", "물리 지표: 영상 잡음")]:
            norm = df[col] / df["ref"].map(cal.groupby("ref")[col].mean())
            ax.plot(df["check"], norm.rolling(WINDOW).mean(), color=c, lw=1.6, ls="--", label=f"{lab} (영상별 기준 = 1)")
        j = summary[f"{s} 고장"]
        if j.get("경보_점검번호") is not None:
            ax.axvline(j["경보_점검번호"], color="#1d2733", lw=1.2)
            ax.text(j["경보_점검번호"] + 4, 0.08, f"경보 → 판정: {j['원인판정']}", fontsize=8.5, color="#1d2733")
        ax.set(title=f"{s} 고장 모의", xlabel="점검 순서", ylim=(0, 1.6))
        ax.grid(alpha=.3)
    axes[0].set_ylabel("비율")
    axes[0].legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "diagnose.png")


if __name__ == "__main__":
    main()
