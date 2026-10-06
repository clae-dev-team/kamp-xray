"""박스 단위 평가. 모든 모델(베이스라인·CNN·YOLO)이 같은 기준으로 채점되도록 한 곳에 둔다.

맞춤 기준 (둘 다 계산)
  center : 예측 박스 중심이 정답 박스(가장자리 margin px 확장) 안에 있으면 맞음.
           이물이 약 10px로 작아 박스 몇 px 차이로 IoU가 크게 흔들리므로 주지표로 쓴다.
  iou50  : IoU ≥ 0.5 (YOLO 관례, mAP50 비교용)
정답 하나에는 예측 하나만 맞출 수 있고, 점수가 높은 예측부터 배정한다.
정상 이미지가 없어 이미지 단위 오경보율은 여기서 계산하지 않는다.
"""
import numpy as np
import pandas as pd


def load_gt(ids, label_dir, sizes):
    """{id: (N,4) xyxy 픽셀}. sizes = {id: (w, h)}.

    ids: 영상 id 목록, label_dir: YOLO 형식 라벨 폴더(<id>.txt).
    라벨 한 줄은 class cx cy w h 이고 값은 영상 크기에 대한 0~1 비율이다.
    """
    gt = {}
    for i in ids:
        b = np.loadtxt(f"{label_dir}/{i}.txt", ndmin=2)   # 박스가 하나뿐이어도 (1, 5) 로 읽는다
        w, h = sizes[i]
        # 0열은 class 라 쓰지 않는다. 비율에 영상 크기를 곱해 픽셀로 바꾼다
        cx, cy, bw, bh = b[:, 1] * w, b[:, 2] * h, b[:, 3] * w, b[:, 4] * h
        gt[i] = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
    return gt


def _iou(a, b):
    """박스 하나 a (4,) 와 박스 여러 개 b (N, 4) 의 IoU. 모두 xyxy 픽셀. 반환: (N,) 0~1."""
    x0, y0 = np.maximum(a[0], b[:, 0]), np.maximum(a[1], b[:, 1])
    x1, y1 = np.minimum(a[2], b[:, 2]), np.minimum(a[3], b[:, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)   # 겹치지 않으면 0
    area = lambda r: (r[..., 2] - r[..., 0]) * (r[..., 3] - r[..., 1])
    return inter / (area(a) + area(b) - inter + 1e-9)   # 1e-9: 넓이가 0일 때 0으로 나누는 것을 막는다


def match(pred: pd.DataFrame, gt: dict, rule="center", margin=2.0):
    """pred 열: id, x0, y0, x1, y1, score. 반환: pred에 tp 열(0/1)과 gt_idx 열을 붙인 표, 정답별 검출 여부.

    gt: {id: (N, 4) xyxy 픽셀}. rule: "center"(중심 일치) 또는 그 밖의 값(IoU 0.5 이상).
    margin: center 기준에서 정답 박스를 사방으로 넓히는 폭(px).
    반환하는 표는 점수 내림차순으로 다시 정렬되어 있고, gt_idx 는 맞은 정답의 번호(못 맞으면 -1)이다.
    정답별 검출 여부는 {id: (N,) bool} 이다.
    """
    # 점수가 높은 예측부터 정답을 차지한다. 동점은 원래 순서를 지킨다 (안정 정렬)
    pred = pred.sort_values("score", ascending=False, kind="stable").reset_index(drop=True)
    tp = np.zeros(len(pred), int)
    gi = np.full(len(pred), -1)
    hit = {i: np.zeros(len(g), bool) for i, g in gt.items()}
    for i, grp in pred.groupby("id", sort=False):
        g = gt.get(i, np.zeros((0, 4)))
        # used 는 hit[i] 와 같은 배열이다. 여기서 표시하면 반환하는 hit 에도 그대로 남는다
        used = hit.setdefault(i, np.zeros(len(g), bool))
        for idx, r in zip(grp.index, grp[["x0", "y0", "x1", "y1"]].to_numpy()):
            if len(g) == 0:   # 정답이 없는 영상의 예측은 모두 오검출(tp 0)로 남는다
                continue
            if rule == "center":
                cx, cy = (r[0] + r[2]) / 2, (r[1] + r[3]) / 2
                ok = (g[:, 0] - margin <= cx) & (cx <= g[:, 2] + margin) & \
                     (g[:, 1] - margin <= cy) & (cy <= g[:, 3] + margin)
                # 여러 정답에 걸치면 중심이 가장 가까운 것
                d = np.hypot((g[:, 0] + g[:, 2]) / 2 - cx, (g[:, 1] + g[:, 3]) / 2 - cy)
                cand = np.where(ok & ~used)[0]   # 이미 다른 예측이 차지한 정답은 뺀다
                j = cand[np.argmin(d[cand])] if len(cand) else -1
            else:
                iou = _iou(r, g)
                iou[used] = -1   # 이미 차지된 정답은 고르지 못하게 한다
                j = int(np.argmax(iou)) if iou.max() >= 0.5 else -1
            if j >= 0:
                used[j] = True
                tp[idx], gi[idx] = 1, j
    pred["tp"], pred["gt_idx"] = tp, gi
    return pred, hit


def pr_curve(pred_matched: pd.DataFrame, n_gt: int):
    """점수 내림차순 누적 정밀도·재현율과 AP (VOC식 모든 점 보간).

    pred_matched: match 가 돌려준 표(score, tp 열). n_gt: 정답 박스 수.
    반환: (점수, 정밀도, 재현율, AP). 앞의 셋은 예측 수 길이의 배열이고,
    k번째 값은 점수 상위 k+1개까지 받았을 때의 값이다.
    """
    p = pred_matched.sort_values("score", ascending=False)
    tp = p["tp"].to_numpy()
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)   # 누적 맞음 수, 누적 오검출 수
    rec = ctp / max(n_gt, 1)
    prec = ctp / np.maximum(ctp + cfp, 1)
    # AP: 양 끝에 (재현율 0, 정밀도 1) 과 (재현율 1, 정밀도 0) 을 덧붙인다
    mrec = np.r_[0, rec, 1]
    mpre = np.r_[1, prec, 0]
    # 정밀도를 오른쪽부터의 최댓값으로 바꿔 단조 감소로 만든 뒤, 재현율이 늘어난 폭만큼 곱해 더한다
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    ap = float(np.sum((mrec[1:] - mrec[:-1]) * mpre[1:]))
    return p["score"].to_numpy(), prec, rec, ap


def at_threshold(pred_matched: pd.DataFrame, n_gt: int, thr: float) -> dict:
    """점수가 thr 이상인 예측만 받았을 때의 지표.

    반환: thr, TP, FP, FN, precision, recall, F1 (비율은 소수 넷째 자리까지).
    FN 은 정답 수에서 TP 를 뺀 값이다.
    """
    p = pred_matched[pred_matched["score"] >= thr]
    tp = int(p["tp"].sum())
    fp = len(p) - tp
    fn = n_gt - tp
    # 분모가 0이 되는 경우(예측 없음, 정답 없음)는 max 로 막아 0으로 나오게 한다
    prec = tp / max(tp + fp, 1)
    rec = tp / max(n_gt, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return dict(thr=float(thr), TP=tp, FP=fp, FN=fn, precision=round(prec, 4),
                recall=round(rec, 4), F1=round(f1, 4))


def best_f1_threshold(pred_matched: pd.DataFrame, n_gt: int) -> float:
    """F1 이 가장 높아지는 임계값(그 지점 예측의 점수). 예측이 없으면 0.0.

    F1 이 같은 지점이 여럿이면 점수가 가장 높은 쪽을 고른다.
    """
    scores, prec, rec, _ = pr_curve(pred_matched, n_gt)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    return float(scores[int(np.argmax(f1))]) if len(scores) else 0.0


def recall_threshold(pred_matched: pd.DataFrame, n_gt: int, target: float) -> float:
    """재현율이 target 이상이 되는 가장 높은 임계값 (재현율 우선 판정용).

    target: 목표 재현율(0~1). 후보를 다 받아도 목표에 못 미치면 가장 낮은 점수를, 예측이 없으면 0.0 을 돌려준다.
    """
    scores, _, rec, _ = pr_curve(pred_matched, n_gt)
    ok = np.nonzero(rec >= target)[0]   # 재현율은 뒤로 갈수록 커지므로 첫 지점의 점수가 가장 높다
    return float(scores[ok[0]]) if len(ok) else float(scores[-1]) if len(scores) else 0.0


def evaluate(pred: pd.DataFrame, gt: dict, thr: float, rule="center") -> dict:
    """한 분할을 한 임계값 · 한 맞춤 기준으로 채점한다.

    pred: 열 id, x0, y0, x1, y1, score. gt: {id: (N, 4) xyxy 픽셀}. thr: 점수 임계값.
    반환: rule, n_images, n_gt, AP 와 at_threshold 의 지표. AP 는 임계값과 상관없이 예측 전체로 계산한다.
    """
    n_gt = sum(len(g) for g in gt.values())
    # gt 에 없는 영상의 예측은 채점에서 뺀다
    pm, _ = match(pred[pred["id"].isin(gt.keys())], gt, rule)
    _, _, _, ap = pr_curve(pm, n_gt)
    return dict(rule=rule, n_images=len(gt), n_gt=n_gt, AP=round(ap, 4), **at_threshold(pm, n_gt, thr))
