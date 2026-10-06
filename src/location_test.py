"""위치 의존 검증: 모델이 '어두운 점'을 보는가, '이물이 늘 있던 자리(띠 왼쪽 끝)'를 보는가.

test 영상의 실제 이물 자리에서 세 조건을 비교한다.
  R 실제      : 정제본 그대로
  E 이물 지움  : 이물 점(반치폭 영역+1px)만 지워 주변 결로 메움 → 자리만 남음
  S 자리 교체  : E 위에 같은 자리로 합성 이물(실제와 비슷한 d=2, c0=0.7)을 넣음
E에서도 반응하면 자리만 보고 판단하는 것, S에서 못 찾으면 합성 이물 모양이 실제와 다른 것이다.
다른 자리에 넣은 합성 이물 결과(synth_eval)와 함께 읽는다.

실행: .venv\\Scripts\\python.exe src\\location_test.py --yolo y26s_640
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from PIL import Image

import baseline as B
from prepare import restore
from synth import insert
import cnn as C
from train_yolo import weights_path

ROOT = Path(__file__).resolve().parents[1]
SWAP = dict(d=2.0, c0=0.70)      # S 조건에 넣는 합성 이물: 지름 2px, 명목 대비 0.70


def dot_mask(g, cx, cy):
    """(cx, cy) 근처 이물 점의 반치폭 영역 + 1px.

    g: 회색 영상(uint8), cx, cy: 점 중심의 화소 번호 좌표(픽셀).
    반환: 영상과 같은 크기의 uint8 마스크(지울 자리 1). defect_stats.measure 의 반치폭 면적과 같은 방식으로 잡는다.
    """
    f = g.astype(np.float32)
    h, w = f.shape
    # 중심 둘레 19×19 조각만 본다 (영상 밖은 잘림)
    x0, x1, y0, y1 = max(0, int(cx) - 9), min(w, int(cx) + 10), max(0, int(cy) - 9), min(h, int(cy) + 10)
    p = f[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d = np.maximum(np.abs(xx - cx), np.abs(yy - cy))      # 중심에서의 체비쇼프 거리 (정사각형 고리)
    bg = np.median(p[d >= 5])                             # 주변 밝기: 5px 이상 떨어진 고리의 중앙값
    lo = cv2.blur(p, (2, 2))[d <= 3].min()                # 점의 가장 어두운 값: 중심 3px 안 2×2 평균의 최솟값
    # 주변과 최저점의 중간보다 어두운 화소 = 반치폭 영역 후보
    half = (p < bg - (bg - lo) / 2).astype(np.uint8)
    n, lab = cv2.connectedComponents(half, connectivity=8)
    # 중심 3px 안에 걸친 덩어리만 남긴다 (멀리 있는 다른 어두운 자리는 지우지 않음). 라벨 0 은 배경
    keep = np.unique(lab[(d <= 3) & (half > 0)])
    m = np.zeros(f.shape, np.uint8)
    m[y0:y1, x0:x1] = np.isin(lab, keep[keep > 0])
    return cv2.dilate(m, np.ones((3, 3), np.uint8))       # 3×3 팽창으로 둘레 1px 을 더 지운다


def site_score(pred, cx, cy, half=5, margin=2):
    """자리 (cx, cy) 에서 가로·세로 half + margin px 안에 중심이 든 예측의 최고 점수. 없으면 0.

    pred: 한 영상의 예측 표(열 px, py, score). 기본값 5 + 2 = 7px 은 합성 평가의 채점 범위와 같다.
    """
    q = pred[(np.abs(pred["px"] - cx) <= half + margin) & (np.abs(pred["py"] - cy) <= half + margin)]
    return float(q["score"].max()) if len(q) else 0.0


def main():
    """test 영상으로 R · E · S 세 조건 영상을 만들고, 이물 자리마다 모델과 베이스라인의 반응을 비교해 저장한다."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "data.yaml"))
    ap.add_argument("--yolo", default="y26s_640")
    args = ap.parse_args()
    from ultralytics import YOLO
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    data = ROOT / cfg["out_dir"]
    # 기본 모델 결과는 location_test/, 다른 모델은 location_test_<이름>/ 에 따로 둔다
    out = ROOT / "results" / ("location_test" if args.yolo == "y26s_640" else f"location_test_{args.yolo}")
    out.mkdir(parents=True, exist_ok=True)

    man = pd.read_csv(data / "manifest.csv")
    man = man[man["labeled"] & (man["split"] == "test")]
    real = pd.read_csv(ROOT / "results/defect_stats/real_defects.csv")
    real = real[real["split"] == "test"]

    # 조건별 영상 만들기 (영상 안 이물을 한꺼번에 지우거나 바꾼다)
    imgs = {"R_실제": {}, "E_이물지움": {}, "S_자리교체": {}}
    sites = []
    for r in man.itertuples():
        g = np.asarray(Image.open(data / "clean/images" / f"{r.id}.png"))
        # 영상마다 고정된 난수: 설정 시드 + 파일 해시 앞 8자리. 다시 돌려도 같은 결로 메워진다
        rng = np.random.default_rng([cfg["seed"], int(r.sha1[:8], 16)])
        mask = np.zeros_like(g)
        centers = []
        for b in np.loadtxt(data / "clean/labels" / f"{r.id}.txt", ndmin=2):
            # 라벨은 class cx cy w h (0~1 비율). 픽셀 박스로 바꿔 그 안에서 가장 어두운 곳을 찾는다
            bw, bh = b[3] * r.w, b[4] * r.h
            x0, y0 = int(max(0, b[1] * r.w - bw / 2)), int(max(0, b[2] * r.h - bh / 2))
            sub = cv2.blur(g[y0:int(b[2] * r.h + bh / 2) + 1, x0:int(b[1] * r.w + bw / 2) + 1].astype(np.float32), (2, 2))
            iy, ix = np.unravel_index(np.argmin(sub), sub.shape)
            cx, cy = x0 + ix + 0.5, y0 + iy + 0.5          # 2×2 평균 최솟값 → 점 중심
            # centers 는 화소 중심이 정수 + 0.5 인 좌표(insert 가 쓰는 방식), dot_mask 는 화소 번호 좌표라 0.5 를 뺀다
            mask |= dot_mask(g, cx - 0.5, cy - 0.5)
            centers.append((cx, cy))
        erased = restore(g, mask, rng)                     # E: 점 자리만 주변 결로 메움
        swapped = erased.astype(np.float32)
        for cx, cy in centers:
            insert(swapped, cx, cy, **SWAP)                # S: 지운 영상 위 같은 자리에 합성 이물 (제자리 수정)
        imgs["R_실제"][r.id] = g
        imgs["E_이물지움"][r.id] = erased
        imgs["S_자리교체"][r.id] = np.clip(swapped.round(), 0, 255).astype(np.uint8)
        sites += [dict(id=r.id, machine=r.machine, cx=cx, cy=cy) for cx, cy in centers]
    sites = pd.DataFrame(sites)

    # 세 조건 영상을 지정 모델(YOLO 또는 CNN)과 베이스라인으로 예측한다
    bl = json.load(open(ROOT / "results/baseline_clean/metrics.json", encoding="utf-8"))
    yo = json.load(open(C.metrics_file(args.yolo), encoding="utf-8"))
    boxes = {int(k): v for k, v in bl["params"]["box_by_machine"].items()}
    use_cnn = C.is_cnn(args.yolo)
    lab = "CNN" if use_cnn else "YOLO"
    if use_cnn:
        model, cboxes, dev = C.load(args.yolo)
    else:
        model = YOLO(str(weights_path(args.yolo)))
    summary = {}
    for cond, dct in imgs.items():
        rows = []
        for r in man.itertuples():
            g = dct[r.id]
            if use_cnn:
                for q in C.detect(model, g, cboxes[int(r.machine)], dev).itertuples():
                    rows.append((lab, r.id, (q.x0 + q.x1) / 2, (q.y0 + q.y1) / 2, float(q.score)))
            else:
                # 파일이 아니라 배열을 바로 넣으므로 회색을 3채널(BGR)로 바꿔 준다
                res = model.predict(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), imgsz=640, conf=0.001, max_det=100,
                                    verbose=False)[0]
                for (x0, y0, x1, y1), s in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
                    rows.append((lab, r.id, (x0 + x1) / 2, (y0 + y1) / 2, float(s)))
            d = B.detect(g, bl["params"]["se"], bl["params"]["sigma"], bl["params"]["score"], boxes[int(r.machine)])
            for q in d.itertuples():
                rows.append(("베이스라인", r.id, (q.x0 + q.x1) / 2, (q.y0 + q.y1) / 2, float(q.score)))
        pred = pd.DataFrame(rows, columns=["model", "id", "px", "py", "score"])
        # 자리별 최고 점수를 구해, 각 모델의 val F1 최대 임계값 이상이면 '반응'으로 센다
        for name, thr in [(lab, yo["thresholds"]["F1최대"]), ("베이스라인", bl["thresholds"]["F1최대"])]:
            p = pred[pred["model"] == name]
            sc = np.array([site_score(p[p["id"] == s.id], s.cx, s.cy) for s in sites.itertuples()])
            sites[f"{cond}/{name}"] = sc
            summary[f"{cond}/{name}"] = dict(자리수=len(sc), 반응=int((sc >= thr).sum()),
                                             반응률=round(float((sc >= thr).mean()), 3),
                                             점수중앙=round(float(np.median(sc)), 3), 임계값=round(thr, 3))
            print(cond, name, summary[f"{cond}/{name}"])
    sites.to_csv(out / "sites.csv", index=False, encoding="utf-8-sig")
    json.dump(summary, open(out / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # 확인용 확대 그림: 행 = 자리, 열 = R / E / S
    tiles = []
    for s in sites.iloc[::max(1, len(sites) // 6)].head(6).itertuples():      # 자리 목록에서 고른 간격으로 6곳
        row = []
        for cond in imgs:
            # 16px 씩 덧대면 [cy, cy+32) 구간이 원래 좌표의 cy±16 이 된다. 32px 조각을 5배(160px)로 키운다
            p = np.pad(imgs[cond][s.id], 16, mode="edge")
            c = p[int(s.cy):int(s.cy) + 32, int(s.cx):int(s.cx) + 32]
            row.append(np.pad(cv2.resize(c, (160, 160), interpolation=cv2.INTER_NEAREST), 3, constant_values=255))
        tiles.append(np.hstack(row))
    Image.fromarray(np.vstack(tiles)).save(out / "examples.png")


if __name__ == "__main__":
    main()
