"""상시 점검 경보 규칙 비교: 기존 '최근 20개 창' vs 베르누이 CUSUM (Zamzmi 2024 의 관리도 감시를 시험편에 적용).

monitor.py 가 남긴 시험편 결과(timeline.csv 의 hit)만 다시 읽어 계산한다 (AI를 다시 돌리지 않음).

  창 규칙  : 최근 WINDOW 개 시험편 중 잡은 수가 관리 하한 미만 (monitor.py 와 같은 규칙)
  CUSUM    : 놓침 여부 y(1=놓침)로 로그우도비를 누적. 평상시 놓침률 q0(초기 구간) → 나빠진 놓침률 q1 = q0 + SHIFT
             S_t = max(0, S_{t-1} + y·ln(q1/q0) + (1-y)·ln((1-q1)/(1-q0))), S_t > h 이면 경보
  공정 비교: 두 규칙의 '평상시 평균 오경보 간격(ARL0, 시험편 수)'을 같게 맞춘다.
             창 규칙의 ARL0 를 모의(q0 고정 이항)로 잰 뒤, CUSUM 의 h 를 같은 ARL0 가 되게 고른다.

실행: .venv\\Scripts\\python.exe src\\monitor_cusum.py
결과: results/monitor_cusum/ (summary.json, cusum.png)
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import binom

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
WINDOW = 20            # 창 규칙의 창 크기 (monitor.py 와 같은 값)
FALSE_ALARM = 0.01     # 창 규칙의 관리 하한을 정하는 확률 (monitor.py 와 같은 값)
CALIB = 1000           # 이 사진 번호 전까지가 초기 구간 (monitor.py 와 같은 값)
SHIFT = 0.25           # 잡아내고 싶은 변화: 놓침률이 25%p 늘어남
N_SIM = 4000           # ARL0 를 잴 때 되풀이하는 모의 횟수
HORIZON = 3000         # 모의 길이 (시험편 수)


def window_alarm(miss, lcl):
    """창 규칙의 첫 경보 위치. miss 는 시험편 순서의 0/1 배열(1 = 놓침), lcl 은 관리 하한(개).

    최근 WINDOW 개 중 잡은 수가 lcl 미만이 된 첫 시험편의 순번(0부터)을 돌려준다. 경보가 없으면 None.
    """
    hits = 1 - miss
    c = np.convolve(hits, np.ones(WINDOW), "valid")     # c[j] = j 번째부터 WINDOW 개 중 잡은 수
    idx = np.nonzero(c < lcl)[0]
    return idx[0] + WINDOW - 1 if len(idx) else None    # 창의 마지막 시험편 순번으로 바꾼다


def cusum(miss, q0, q1):
    """베르누이 CUSUM 누적값. miss 는 0/1 배열(1 = 놓침), q0 는 평상시 놓침률, q1 은 나빠진 놓침률.

    놓치면 ln(q1/q0) 만큼 오르고 잡으면 ln((1-q1)/(1-q0)) 만큼 내려간다. 0 아래로는 내려가지 않는다.
    반환: miss 와 같은 길이의 누적값 배열 S_t.
    """
    a, b = np.log(q1 / q0), np.log((1 - q1) / (1 - q0))     # a > 0 (놓쳤을 때의 증가분), b < 0 (잡았을 때의 감소분)
    s, out = 0.0, np.empty(len(miss))
    for k, y in enumerate(miss):
        s = max(0.0, s + (a if y else b))
        out[k] = s
    return out


def arl0(rule, q0, rng):
    """평상시 평균 오경보 간격(ARL0, 시험편 수)을 모의로 잰다.

    놓침률 q0 인 베르누이 열을 HORIZON 개 만들어 rule(놓침 배열 → 첫 경보 순번 또는 None)에 넣기를 N_SIM 번 되풀이한다.
    HORIZON 안에 경보가 없으면 HORIZON 으로 센다.
    """
    runs = []
    for _ in range(N_SIM):
        t = rule(rng.random(HORIZON) < q0)
        runs.append(HORIZON if t is None else t + 1)    # 순번은 0부터이므로 1을 더해 시험편 수로 바꾼다
    return float(np.mean(runs))


def main():
    """두 가지 시험편 흐름(생산 사진 · 기준 정상 영상)마다 창 규칙과 CUSUM 의 경보 시점을 비교해 저장한다.

    summary.json: {흐름 이름: {평상시_놓침률_q0, 창_관리하한, 평상시_평균오경보간격_시험편, CUSUM_h,
                   창 규칙 · CUSUM: {경보_프레임, 열화수준, 그때_실제재현율(곡선보간)}}}
    """
    out = ROOT / "results" / "monitor_cusum"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    summary = {"설정": {"창": WINDOW, "변화폭": SHIFT, "모의횟수": N_SIM}}
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.6), dpi=150, sharex=True)
    for ax, (tag, d) in zip(axes, [("생산 사진 시험편", "monitor"), ("기준 정상 영상 시험편", "monitor_reference")]):
        tl = pd.read_csv(ROOT / "results" / d / "timeline.csv")
        sm = json.load(open(ROOT / "results" / d / "summary.json", encoding="utf-8"))
        p = tl[tl["piece"]].reset_index(drop=True)          # 시험편을 넣은 사진만. 행 번호가 시험편 순번이 된다
        miss = (~p["hit"].astype(bool)).to_numpy().astype(int)
        cal = p["frame"] < CALIB
        q0 = max(float(miss[cal].mean()), 1e-3)             # 초기 구간에 놓침이 0건이면 로그를 계산할 수 없어 0.001 로 받친다
        q1 = min(q0 + SHIFT, 0.95)
        lcl = int(binom.ppf(FALSE_ALARM, WINDOW, 1 - q0))   # 창 규칙의 관리 하한 (검출률 1 - q0 기준)
        arl_w = arl0(lambda m: window_alarm(m.astype(int), lcl), q0, rng)
        # h 를 이분 탐색으로 창 규칙과 같은 ARL0 가 되게
        # h 가 클수록 경보가 드물어 ARL0 가 길어진다. CUSUM 쪽이 짧으면 h 를 올리고, 길면 내린다 (0.5~20 사이를 18번 반씩 좁힘)
        lo, hi = 0.5, 20.0
        for _ in range(18):
            h = (lo + hi) / 2
            a = arl0(lambda m: (lambda s: (np.nonzero(s > h)[0][0] if (s > h).any() else None))(cusum(m, q0, q1)), q0, rng)
            lo, hi = (h, hi) if a < arl_w else (lo, h)
        h = (lo + hi) / 2
        post = (p["frame"] >= CALIB).to_numpy()
        s = np.zeros(len(miss))
        s[post] = cusum(miss[post], q0, q1)          # 초기 구간이 끝난 뒤 0에서 시작 (ARL0 모의와 같은 조건)
        k_c = next((k for k in range(len(s)) if post[k] and s[k] > h), None)    # CUSUM 첫 경보의 시험편 순번
        k_w = window_alarm(np.where(post, miss, 0), lcl)                         # 창 규칙: 초기 구간의 놓침은 0 으로 두고 그 뒤의 놓침만 센다
        # 경보 프레임 → 열화 수준(monitor.py 의 level_at 과 같은 식) → 그 수준의 실제 이물 재현율(monitor.py 가 남긴 곡선을 선형 보간)
        curve = {float(k): v for k, v in sm["열화수준별_실제재현율"].items()}
        lv = lambda fr: 0.0 if fr < sm["열화_시작"] else (fr - sm["열화_시작"]) / (sm["프레임수"] - 1 - sm["열화_시작"])
        rec = lambda x: float(np.interp(x, list(curve), list(curve.values())))
        res = {"평상시_놓침률_q0": round(q0, 3), "창_관리하한": lcl, "평상시_평균오경보간격_시험편": round(arl_w, 1),
               "CUSUM_h": round(h, 3)}
        for name, k in [("창 규칙", k_w), ("CUSUM", k_c)]:
            fr = int(p.loc[k, "frame"]) if k is not None else None
            res[name] = {"경보_프레임": fr, "열화수준": round(lv(fr), 3) if fr else None,
                         "그때_실제재현율(곡선보간)": round(rec(lv(fr)), 3) if fr else None}
        summary[tag] = res
        print(tag, json.dumps(res, ensure_ascii=False))
        # 그림: CUSUM 누적값과 경보선 h, 두 규칙의 경보 시점(세로선). 바탕색은 열화 구간
        ax.axvspan(sm["열화_시작"], sm["프레임수"], color="#f2e3d3", alpha=.6)
        ax.plot(p["frame"], s, color="#0a8f86", lw=1.6, label="CUSUM 누적값")
        ax.axhline(h, color="#1d2733", lw=0.9, ls="--", label=f"CUSUM 경보선 h={h:.2f}")
        for name, c, ls in [("CUSUM", "#0a8f86", "-"), ("창 규칙", "#c96f24", ":")]:
            if res[name]["경보_프레임"]:
                ax.axvline(res[name]["경보_프레임"], color=c, lw=1.4, ls=ls, label=f"{name} 경보 {res[name]['경보_프레임']}")
        ax.set(title=tag, ylabel="누적값", ylim=(0, max(h * 3, 1)))
        ax.grid(alpha=.3)
        ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    axes[-1].set_xlabel("생산 사진 순서 (열화 구간은 가정)")
    fig.tight_layout()
    fig.savefig(out / "cusum.png")
    json.dump(summary, open(out / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
