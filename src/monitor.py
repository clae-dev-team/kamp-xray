"""운영 중 AI 상시 점검 모의 시연: 생산 사진 흐름에 가상 시험편을 섞어 AI 검출 감도를 계속 잰다.

공항 수하물 검사의 위험물 영상 투영(TIP)처럼, 실제 라인에서도 일정 간격으로 가상 시험편을 넣은 사진을
AI에 보여 주고 잡는지 기록하면, 장비나 AI가 조용히 무뎌지는 것을 사람이 알아채기 전에 경보로 알 수 있다.

모의 조건 (모두 가정이며 실제 장비 열화 자료가 아니다)
  - 생산 흐름: 시험 분할 가짜 정상 73장을 순서대로 되풀이해 N_FRAMES 장
  - 시험편: EVERY 장마다 1장에 지름 2px 이물 1개. 진하기 = testpiece.py 가 정한 그 호기의 사양값(90% 보장).
    자리는 실제 이물처럼 **어두운 띠 안** (공장 시험편도 가장 어려운 자리에서 점검한다).
    처음에는 제품 아무 데나 넣었는데, 실제 이물(띠 안)보다 늦게 무뎌져 경보가 실제 손실 뒤에 울렸다.
  - 열화: DEGRADE_FROM 장부터 끝까지 흐림(가우시안 σ 0→BLUR_MAX)과 잡음(표준편차 0→NOISE_MAX 회색 단계)이 선형으로 증가
    시험편은 열화 전에 넣는다 (시험편도 같은 장비를 지나가므로)
  - 경보: 최근 WINDOW 개 시험편 중 잡은 수가 관리 하한 미만. 관리 하한 = 초기 CALIB 장(열화 전)의 검출률로 본
    이항분포에서 평상시 우연히 그 아래로 떨어질 확률이 FALSE_ALARM 이하가 되는 수 (관리도 방식)
  - 대조: 가장 심한 열화를 실제 불량 시험 사진 73장에 걸었을 때 실제 이물 재현율 (점검이 없으면 모르고 지나갈 손실)

실행: .venv\\Scripts\\python.exe src\\monitor.py --yolo ratio3_e100
결과: results/monitor/ (timeline.csv, summary.json, timeline.png)
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

import metrics as M
from prepare import product_mask
from synth import band_mask, insert
from train_yolo import weights_path

ROOT = Path(__file__).resolve().parents[1]
N_FRAMES = 4000
EVERY = 10
DEGRADE_FROM = 1000
BLUR_MAX = 1.6
NOISE_MAX = 6.0
WINDOW = 20
CALIB = 1000                 # 경보선을 정하는 초기 구간 (열화 전)
FALSE_ALARM = 0.01           # 평상시 한 창에서 우연히 경보가 날 확률 상한
PIECE_D = 2.0
SEED_OFFSET = 4242


def degrade(g, level, rng):
    """level 0~1. 흐림과 잡음을 함께 키운다."""
    f = g.astype(np.float32)
    if level > 0:
        s = BLUR_MAX * level
        f = cv2.GaussianBlur(f, (0, 0), s) if s > 0.05 else f
        f = f + rng.normal(0, NOISE_MAX * level, f.shape).astype(np.float32)
    return np.clip(f.round(), 0, 255).astype(np.uint8)


def level_at(i):
    return 0.0 if i < DEGRADE_FROM else (i - DEGRADE_FROM) / (N_FRAMES - 1 - DEGRADE_FROM)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "data.yaml"))
    ap.add_argument("--yolo", default="ratio3_e100")
    args = ap.parse_args()
    from ultralytics import YOLO
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    data = ROOT / cfg["out_dir"]
    out = ROOT / "results" / "monitor"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng([cfg["seed"], SEED_OFFSET])

    model = YOLO(str(weights_path(args.yolo)))
    thr = json.load(open(ROOT / f"results/yolo_{args.yolo}/metrics.json", encoding="utf-8"))["thresholds"]["F1최대"]
    spec = pd.read_csv(ROOT / "results/testpiece/spec.csv")
    spec = spec[(spec["model"] == "YOLO") & (spec["d"] == PIECE_D)].set_index("machine")["min_c0"]
    man = pd.read_csv(data / "manifest.csv").set_index("id")
    srcs = sorted(p.stem for p in (data / "normal" / "test").glob("*.png"))
    base = {s: np.asarray(Image.open(data / "normal" / "test" / f"{s}.png").convert("L")) for s in srcs}
    pms = {}
    for s, g in base.items():
        pm = product_mask(g)
        pms[s] = np.nonzero(band_mask(g, pm) & (pm > 0))

    def detect(g):
        r = model.predict(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), imgsz=640, conf=thr, max_det=100, verbose=False)[0]
        b = r.boxes.xyxy.cpu().numpy()
        return np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], 1) if len(b) else np.zeros((0, 2))

    rows = []
    for i in tqdm(range(N_FRAMES), desc="생산 흐름"):
        s = srcs[i % len(srcs)]
        m = int(man.loc[s, "machine"])
        f = base[s].astype(np.float32)
        piece = None
        if i % EVERY == EVERY - 1:
            ys, xs = pms[s]
            k = rng.integers(len(xs))
            cx, cy = xs[k] + rng.random(), ys[k] + rng.random()
            c0 = float(spec.get(m, 0.30)) if pd.notna(spec.get(m, np.nan)) else 0.30
            insert(f, cx, cy, PIECE_D, c0)
            piece = (cx, cy, c0)
        lv = level_at(i)
        g = degrade(np.clip(f.round(), 0, 255).astype(np.uint8), lv, rng)
        pts = detect(g)
        hit = None
        n_other = len(pts)
        if piece is not None:
            near = np.hypot(pts[:, 0] - piece[0], pts[:, 1] - piece[1]) <= 7 if len(pts) else np.zeros(0, bool)
            hit = bool(near.any())
            n_other = int((~near).sum())
        rows.append(dict(frame=i, src=s, machine=m, level=round(lv, 4), piece=piece is not None,
                         c0=piece[2] if piece else None, hit=hit, other_alarms=n_other))
    tl = pd.DataFrame(rows)
    p = tl[tl["piece"]].copy()
    p["rolling"] = p["hit"].astype(float).rolling(WINDOW).mean()
    tl = tl.merge(p[["frame", "rolling"]], on="frame", how="left")
    tl.to_csv(out / "timeline.csv", index=False, encoding="utf-8-sig")
    from scipy.stats import binom
    full = p["frame"] >= WINDOW * EVERY
    base_rate = float(p[p["frame"] < CALIB]["hit"].mean())
    lcl = int(binom.ppf(FALSE_ALARM, WINDOW, base_rate))          # 이 수 미만이면 경보
    false_alarm_p = float(binom.cdf(lcl - 1, WINDOW, base_rate))
    p["count"] = p["hit"].astype(float).rolling(WINDOW).sum()
    alarm = p[(p["count"] < lcl) & full & (p["frame"] >= CALIB)]
    first = int(alarm["frame"].iloc[0]) if len(alarm) else None
    pre = p[(p["count"] < lcl) & full & (p["frame"] < CALIB)]

    # 대조: 가장 심한 열화에서 실제 불량 사진의 실제 이물 재현율
    tm = man[man["labeled"] & (man["split"] == "test")]
    sizes = {i: (r.w, r.h) for i, r in tm.iterrows()}
    gt = M.load_gt(tm.index.tolist(), data / "clean/labels", sizes)
    def real_recall(lv):
        rr = np.random.default_rng([cfg["seed"], SEED_OFFSET, 7])
        pr = []
        for i in tm.index:
            g = degrade(np.asarray(Image.open(data / "clean/images" / f"{i}.png").convert("L")), lv, rr)
            r = model.predict(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), imgsz=640, conf=0.001, max_det=100, verbose=False)[0]
            for (x0, y0, x1, y1), sc in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                pr.append(dict(id=i, x0=x0, y0=y0, x1=x1, y1=y1, score=float(sc)))
        return M.evaluate(pd.DataFrame(pr, columns=["id", "x0", "y0", "x1", "y1", "score"]), gt, thr, "center")

    curve = {round(lv, 2): real_recall(lv)["recall"] for lv in tqdm(np.linspace(0, 1, 11), desc="열화별 재현율")}
    rec = {}
    for tag, fr in [("경보 시점", first)]:
        if fr is not None:
            ev = real_recall(level_at(fr))
            rec[tag] = {"프레임": fr, "열화수준": round(level_at(fr), 3), "재현율": ev["recall"], "놓침": ev["FN"]}

    before = base_rate
    summary = {"프레임수": N_FRAMES, "시험편_간격": EVERY, "시험편수": int(len(p)), "열화_시작": DEGRADE_FROM,
               "시험편_진하기_호기별": {int(k): float(v) for k, v in spec.items()},
               "열화전_시험편_검출률": round(float(before), 3), "경보_규칙": f"최근 {WINDOW}개 중 잡은 수 < {lcl} (초기 {CALIB}장 검출률 {base_rate:.3f} 기준)",
               "초기구간_경보수": int(len(pre)),
               "평상시_헛경보확률_창당": round(false_alarm_p, 4),
               "첫_경보_프레임": first, "경보까지_지연_프레임": (first - DEGRADE_FROM) if first else None,
               "경보_시점_열화수준": round(level_at(first), 3) if first else None,
               "열화수준별_실제재현율": curve,
               "열화전_정상부위_헛경보_영상당": round(float(tl[tl["frame"] < DEGRADE_FROM]["other_alarms"].mean()), 4),
               "경보시점_실제불량": rec}
    json.dump(summary, open(out / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=float)
    print(json.dumps(summary, ensure_ascii=False, indent=1, default=float))

    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=150)
    ax.axvspan(DEGRADE_FROM, N_FRAMES, color="#f2e3d3", alpha=.6, label="장비 열화 구간 (가정)")
    ax.plot(p["frame"], p["rolling"], color="#0a8f86", lw=2, label=f"시험편 검출률 (최근 {WINDOW}개)")
    fr = np.arange(DEGRADE_FROM, N_FRAMES)
    lv = np.array(list(curve)), np.array(list(curve.values()))
    ax.plot(fr, np.interp([level_at(i) for i in fr], *lv), color="#c96f24", lw=1.6, ls=":",
            label="같은 열화에서 실제 이물 재현율")
    ax.plot([0, DEGRADE_FROM], [curve[0.0]] * 2, color="#c96f24", lw=1.6, ls=":")
    ax.axhline(lcl / WINDOW, color="#1d2733", lw=0.8, ls="--", label=f"경보선 ({WINDOW}개 중 {lcl}개)")
    if first:
        ax.axvline(first, color="#1d2733", lw=1.2)
        ax.text(first + 8, 0.06, f"경보: {first}번째 사진", fontsize=8, color="#1d2733")
    ax.set(xlabel="생산 사진 순서", ylabel="검출률", ylim=(0, 1.05), xlim=(0, N_FRAMES))
    ax.grid(alpha=.3)
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "timeline.png")


if __name__ == "__main__":
    main()
