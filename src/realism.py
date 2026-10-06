"""시험편 현실성 검증: 합성 이물이 실제 이물과 구분되는가 (TIP 영상 현실성 연구, Hättenschwiler 2022 의 문제의식).

실제 이물 자리마다 (location_test.py 와 같은 방법)
  R 실제   : 정제본의 실제 이물 조각
  S 합성   : 같은 사진에서 실제 이물 점만 지우고, 같은 자리에 Beer–Lambert 합성 이물을 넣은 조각.
             진하기 c0 · 지름 d 는 격자에서 '측정 대비 + 반치폭 넓이'가 실제와 가장 가까운 값으로 맞춘다.
  S0 고정  : 맞추지 않고 c0 = 0.70, d = 2 로 고정한 합성 (대조군: 맞추기가 왜 필요한지)
  ST 이식   : 같은 호기 다른 자리의 진짜 점(실제 − 지운 배경 차이)을 이 자리의 지운 배경에 옮겨 붙인 것.
             모양은 진짜, 배경은 합성과 같은 '지운 배경'이다. 실제 vs ST 가 0.5 근처면 높은 AUC 는 지운 흔적이
             아니라 점 모양 때문이라는 뜻이다 (대조군).
  SB 네모  : 픽셀 칸에 맞춘 네모 덩어리(1×1~3×3)에 같은 투과율을 곱한 합성. 실제 이물이 칸에 맞춘 2~4칸짜리
             고른 덩어리라서(구 합성은 십자 모양으로 번져 거의 완벽히 구분됐다: AUC 0.998) 모양을 실제에 맞춘 판
배경이 같은 사진·같은 자리라 차이는 점 자체에서만 난다.

구분력: 조각(11×11, 주변 중앙값을 빼고 점 깊이로 나눠 모양만 남김)으로 로지스틱 회귀를 학습해
사진 단위로 묶은 5겹 교차검증 AUC 를 잰다. 0.5 = 구분 불가(현실적), 1.0 = 완전히 구분됨.
손으로 만든 특징(대비·반치폭 넓이·가장자리 기울기·좌우상하 대칭)으로도 따로 잰다.

실행: .venv\\Scripts\\python.exe src\\realism.py
결과: results/realism/ (summary.json, pairs.png)
"""
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from defect_stats import measure
from location_test import dot_mask
from normal_set import dot_centers
from prepare import restore
from synth import insert

ROOT = Path(__file__).resolve().parents[1]
C0_GRID = np.round(np.arange(0.20, 0.96, 0.05), 2)     # 맞춰 볼 진하기 c0 후보: 0.20~0.95, 0.05 간격
D_GRID = [1.5, 2.0, 2.5, 3.0]                          # 맞춰 볼 구 지름 후보 (px)
HALF = 5                                               # 비교 조각 반폭: 조각 한 변 = 2 × 5 + 1 = 11px
BLOCKS = [(1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2), (3, 3)]     # (세로, 가로) 픽셀 칸


def insert_box(f, cx, cy, bh, bw, c0):
    """(cx, cy) 를 포함하는 칸 정렬 bh×bw 덩어리에 투과율 (1 - c0) 를 곱한다 (두께가 고른 작은 조각).

    f: float32 영상(제자리 수정), cx, cy: 화소 중심이 정수 + 0.5 인 좌표, bh · bw: 세로 · 가로 칸 수, c0: 0~1.
    """
    y0, x0 = int(round(cy - bh / 2)), int(round(cx - bw / 2))
    f[max(0, y0):y0 + bh, max(0, x0):x0 + bw] *= (1 - c0)


def patch(g, cx, cy):
    """(cx, cy) 가 든 화소를 가운데에 둔 11×11 조각(float32). 영상 밖은 가장자리 값으로 채운다."""
    f = g.astype(np.float32)
    x, y = int(round(cx - 0.5)), int(round(cy - 0.5))      # 화소 번호로 바꾼다
    # HALF + 4 = 9px 씩 덧댔으므로 덧댄 좌표의 y + 4 는 원래 좌표의 y - 5 다
    p = np.pad(f, HALF + 4, mode="edge")[y + 4:y + 4 + 2 * HALF + 1, x + 4:x + 4 + 2 * HALF + 1]
    return p


def shape_vec(p):
    """조각에서 밝기와 대비를 빼고 모양만 남긴 벡터. 반환: 길이 121 (11×11 을 편 것), 주변은 0 근처 · 가장 어두운 곳은 -1."""
    # 주변 밝기 = 조각 바깥 2px 테두리의 중앙값
    ring = np.ones_like(p, bool)
    ring[2:-2, 2:-2] = False
    bg = np.median(p[ring])
    depth = max(bg - p.min(), 1.0)       # 점 깊이. 0 으로 나누지 않게 최소 1
    return ((p - bg) / depth).ravel()


def hand_feats(g, cx, cy):
    """손으로 만든 특징 5개: 측정 대비, 반치폭 넓이, 가장자리 기울기, 상하 비대칭, 좌우 비대칭."""
    m = measure(g, cx - 0.5, cy - 0.5)
    p = patch(g, cx, cy)
    gy, gx = np.gradient(p)
    gm = np.hypot(gx, gy)                # 화소별 기울기 크기
    s = (p - np.median(p)) / max(np.median(p) - p.min(), 1)
    # 기울기: 가운데 5×5 의 평균 기울기를 조각 표준편차로 나눈 값. 대칭: 조각을 뒤집은 것과의 평균 절대 차
    return [m["contrast"], m["area"], float(gm[HALF - 2:HALF + 3, HALF - 2:HALF + 3].mean() / max(p.std(), 1e-3)),
            float(np.abs(s - s[::-1, :]).mean()), float(np.abs(s - s[:, ::-1]).mean())]


def cv_auc(X, y, groups):
    """표준화 + 로지스틱 회귀의 5겹 교차검증 AUC (소수 셋째 자리).

    X: (조각 수, 특징 수), y: 0 = 실제 · 1 = 비교 대상, groups: 조각이 나온 사진 이름.
    같은 사진의 조각이 학습과 검증에 나뉘어 들어가지 않게 사진 단위로 겹을 나눈다.
    """
    pred = np.zeros(len(y))
    # 겹마다 검증 쪽 예측을 채워 넣고, 모든 조각의 예측을 모아 AUC 를 한 번 계산한다
    for tr, te in GroupKFold(5).split(X, y, groups):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))
        clf.fit(X[tr], y[tr])
        pred[te] = clf.predict_proba(X[te])[:, 1]
    return round(float(roc_auc_score(y, pred)), 3)


def main():
    """라벨 있는 사진의 이물 자리마다 실제 · 합성 조각을 만들고, 종류별 구분력 AUC 와 예시 그림을 저장한다."""
    cfg = yaml.safe_load(open(ROOT / "configs" / "data.yaml", encoding="utf-8"))
    data = ROOT / cfg["out_dir"]
    out = ROOT / "results" / "realism"
    out.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(data / "manifest.csv")
    man = man[man["labeled"]]
    rows, examples = [], []
    bank = {}                     # 호기별 진짜 점 차이 조각 (이식 대조군용)
    for r in tqdm(list(man.itertuples()), desc="자리별 실제·합성 조각"):
        g = np.asarray(Image.open(data / "clean/images" / f"{r.id}.png").convert("L"))
        h, w = g.shape
        # 점 중심은 화소 중심이 정수 + 0.5 인 좌표. dot_mask · measure 는 화소 번호 좌표라 0.5 를 빼서 넘긴다
        cs = dot_centers(g, np.loadtxt(data / "clean/labels" / f"{r.id}.txt", ndmin=2), w, h)
        mask = np.zeros_like(g)
        for cx, cy in cs:
            mask |= dot_mask(g, cx - 0.5, cy - 0.5)
        rng = np.random.default_rng([cfg["seed"], int(r.sha1[:8], 16), 77])
        er = restore(g, mask, rng).astype(np.float32)      # 사진 안 이물 점을 모두 지운 배경
        for k, (cx, cy) in enumerate(cs):
            mr = measure(g, cx - 0.5, cy - 0.5)            # 실제 이물의 측정 대비와 반치폭 넓이
            # S 맞춘 합성(구): 지름 × 진하기 격자를 모두 넣어 보고 실제와 가장 가까운 것을 고른다.
            # 거리 = 대비 차이(0.05 를 1 단위로) + 넓이 차이(실제 넓이에 대한 비율)
            best, bd = None, np.inf
            for d in D_GRID:
                for c0 in C0_GRID:
                    t = er.copy()
                    insert(t, cx, cy, d, c0)
                    ti = np.clip(t.round(), 0, 255).astype(np.uint8)
                    mt = measure(ti, cx - 0.5, cy - 0.5)
                    dist = abs(mt["contrast"] - mr["contrast"]) / 0.05 + abs(mt["area"] - mr["area"]) / max(mr["area"], 1)
                    if dist < bd:
                        bd, best = dist, (d, c0, ti)
            # SB 맞춘 합성(칸 정렬 네모): 덩어리 크기 × 진하기 격자에서 같은 거리로 고른다
            bb, bdb = None, np.inf
            for bh, bw in BLOCKS:
                for c0 in C0_GRID:
                    t = er.copy()
                    insert_box(t, cx, cy, bh, bw, c0)
                    ti = np.clip(t.round(), 0, 255).astype(np.uint8)
                    mt = measure(ti, cx - 0.5, cy - 0.5)
                    dist = abs(mt["contrast"] - mr["contrast"]) / 0.05 + abs(mt["area"] - mr["area"]) / max(mr["area"], 1)
                    if dist < bdb:
                        bdb, bb = dist, ((bh, bw), c0, ti)
            # S0 고정 합성: 맞추지 않고 지름 2px · 진하기 0.70 으로 넣는다
            t0 = er.copy()
            insert(t0, cx, cy, 2.0, 0.70)
            s0 = np.clip(t0.round(), 0, 255).astype(np.uint8)
            # 진짜 점 차이 조각을 저장하고, 같은 호기에 앞서 저장된 다른 사진의 점을 이 자리에 옮겨 붙인다
            x, y = int(round(cx - 0.5)), int(round(cy - 0.5))
            # 진짜 점 = 실제 영상 - 지운 배경, 중심 둘레 7×7. 영상 가장자리에 걸려 7×7 이 안 되는 자리는 쓰지 않는다
            diff = g.astype(np.float32)[y - 3:y + 4, x - 3:x + 4] - er[y - 3:y + 4, x - 3:x + 4]
            pool = bank.setdefault(r.machine, [])
            st = None
            # 가장 최근에 저장된 것부터 거슬러 올라가 다른 사진에서 나온 점을 찾는다. 없으면 ST 는 만들지 않는다
            donor = next((d for d in reversed(pool) if d[0] != r.id), None)
            if donor is not None and diff.shape == (7, 7):
                t = er.copy()
                t[y - 3:y + 4, x - 3:x + 4] += donor[1]
                st = np.clip(t.round(), 0, 255).astype(np.uint8)
            if diff.shape == (7, 7):
                pool.append((r.id, diff))
            # 종류별로 같은 자리의 조각을 잘라 모양 벡터와 손 특징을 남긴다. d · c0 는 맞춘 합성(S)에만 적는다
            kinds = [("R", g), ("S", best[2]), ("S0", s0), ("SB", bb[2])] + ([("ST", st)] if st is not None else [])
            for kind, img in kinds:
                rows.append(dict(id=r.id, k=k, kind=kind, d=best[0] if kind == "S" else None, c0=best[1] if kind == "S" else None,
                                 shape=shape_vec(patch(img, cx, cy)), feats=hand_feats(img, cx, cy)))
            # 예시 그림용: 표의 색인이 60 의 배수인 사진의 첫 이물만, 최대 6곳 (실제 | 맞춘 구 | 맞춘 네모)
            if len(examples) < 6 and k == 0 and r.Index % 60 == 0:
                examples.append([patch(g, cx, cy), patch(best[2], cx, cy), patch(bb[2], cx, cy)])
    df = pd.DataFrame(rows)
    summary = {"자리수": int((df["kind"] == "R").sum())}
    # 실제(R)와 비교 대상 한 종류씩 짝지어, 둘을 가르는 분류기의 AUC 를 잰다 (y: 실제 0, 비교 대상 1)
    for a, b, tag in [("R", "S", "실제 vs 맞춘 합성(구)"), ("R", "S0", "실제 vs 고정 합성(구, c0 0.7, d 2)"),
                      ("R", "SB", "실제 vs 맞춘 합성(칸 정렬 네모)"),
                      ("R", "ST", "대조: 실제 vs 진짜 점 이식(지운 배경 위)")]:
        sub = df[df["kind"].isin([a, b])]
        y = (sub["kind"] == b).to_numpy().astype(int)
        grp = sub["id"].to_numpy()
        summary[tag] = {"AUC_모양(11x11)": cv_auc(np.stack(sub["shape"]), y, grp),
                        "AUC_손특징": cv_auc(np.array(sub["feats"].tolist()), y, grp)}
        print(tag, summary[tag])
    # 대조: 실제끼리 무작위로 두 무리로 나눴을 때 (0.5 근처가 나와야 정상)
    rr = df[df["kind"] == "R"].copy()
    ids = rr["id"].unique()
    half = set(np.random.default_rng(0).permutation(ids)[: len(ids) // 2])
    y = rr["id"].isin(half).to_numpy().astype(int)
    summary["대조: 실제끼리 무작위 두 무리"] = {"AUC_모양(11x11)": cv_auc(np.stack(rr["shape"]), y, rr["id"].to_numpy())}
    print(summary["대조: 실제끼리 무작위 두 무리"])
    s = df[df["kind"] == "S"]
    summary["맞춘_합성_조건"] = {"지름_분포": s["d"].value_counts().sort_index().to_dict(),
                            "진하기_중앙": float(s["c0"].median())}
    json.dump(summary, open(out / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=float)

    # 예시 그림: 행 = 자리, 열 = 실제 / 맞춘 합성(구) / 맞춘 합성(네모). 11px 조각을 12배(132px)로 키운다
    tiles = []
    for ex in examples:
        row = []
        for p in ex:
            v = np.clip(p, 0, 255).astype(np.uint8)
            row += [cv2.resize(v, (132, 132), interpolation=cv2.INTER_NEAREST), np.full((132, 6), 255, np.uint8)]
        tiles.append(np.hstack(row))
        tiles.append(np.full((6, tiles[-1].shape[1]), 255, np.uint8))
    Image.fromarray(np.vstack(tiles)).save(out / "pairs.png")


if __name__ == "__main__":
    main()
