"""인접 프레임 기준 이식과 구조 맞춤 자리 고르기.

팀 내 다른 분석(위치 강건성 분석: 같은 세션의 인접 프레임을 깨끗한 기준으로 삼아 실제 이물의 차이를 떼어 내고,
주변 구조가 비슷한 자리에 옮기는 방법)의 핵심 절차를 이 저장소의 분할 · 전처리 · 채점 구조에 맞춰 다시 작성한 모듈이다.

augment_paste.py 의 이식은 "같은 사진에서 이물을 지운 배경"을 기준으로 삼는다. 여기서는 지운 배경 대신
**같은 호기에서 몇 초 차이로 찍힌 다른 사진의 같은 자리**(그 자리에 이물이 없는 것)를 기준으로 쓴다. 지운 흔적에 기대지 않는다.

1. 기준 프레임 찾기 (find_references)
   같은 호기 · 같은 날, 촬영 시각 차 MAX_GAP 초 이하, 같은 크기, 이물 자리에 상대 사진의 정답 박스가 겹치지 않음(IoU ≤ 0.01).
   이물 둘레 고리(박스의 3배 영역에서 박스 + 2px 를 뺀 곳)의 평균 밝기 차 RING_MAX 이하(배경이 비슷함),
   중심 차 − 고리 차 CENTER_MIN 이상(이물이 실제로 떼어짐). 여러 장이면 고리 차가 가장 작은 것.
2. 떼어 내기 (extract): 박스 영역에서 mul = 실제 / 기준 (투과율, X선 감쇠와 같은 곱셈) 또는 add = 실제 − 기준.
   경계 20% 는 smoothstep 으로 부드럽게 줄인다.
3. 자리 고르기 (ContextPicker): 이물이 있던 자리의 둘레 구조와 닮은 자리를 고른다.
   비용 = 밝기 구조 차 + 0.35 × 기울기 크기 차 + 0.50 × 기울기 방향 비 차 (둘레를 표준화해 절대 밝기는 무시, 중앙은 제외).
   제품 안, 가장자리 여유, 기존 이물과 거리, 최소 이동량을 지킨 후보 가운데 비용이 낮은 것.
   모델의 예측은 어디에도 쓰지 않는다.
"""
import re
from datetime import datetime

import cv2
import numpy as np
from PIL import Image

MAX_GAP = 30.0            # 초
CONTEXT_SCALE = 3.0
MAX_REF_IOU = 0.01
RING_MAX = 15.0
CENTER_MIN = 10.0
FEATHER = 0.20
MIN_SHIFT = 0.15          # 영상 크기 대비 최소 이동량
EDGE_MULT = 2.5           # 제품 가장자리 여유 (이물 크기의 배수)
GT_MULT = 1.5             # 기존 이물과의 여유
MIN_STD = 3.0             # 밋밋한 자리는 후보에서 뺀다
GRID = (48, 40)           # 후보 격자 (가로, 세로)
MAX_MISMATCH = 0.20       # 치우침 보정 뒤에도 8% 넘게 다른 화소가 조각의 이 비율을 넘으면 버린다 (배경 구조가 안 맞는 기준 프레임)


def shot_time(image_id):
    m = re.search(r"_(\d{8})_(\d{6})", image_id)
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")


def load_gray(data, image_id):
    return np.asarray(Image.open(data / "clean/images" / f"{image_id}.png").convert("L"))


def load_boxes(data, image_id, w, h):
    """정답 박스 (N, 4) xyxy 화소."""
    b = np.loadtxt(data / "clean/labels" / f"{image_id}.txt", ndmin=2)
    cx, cy, bw, bh = b[:, 1] * w, b[:, 2] * h, b[:, 3] * w, b[:, 4] * h
    return np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)


def _iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def _ring(box, w, h):
    """박스의 3배 영역(context)과, 그 안에서 박스 + 2px 를 뺀 고리 마스크."""
    bw, bh = box[2] - box[0], box[3] - box[1]
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    x0, y0 = max(0, int(np.floor(cx - bw * CONTEXT_SCALE / 2))), max(0, int(np.floor(cy - bh * CONTEXT_SCALE / 2)))
    x1, y1 = min(w, int(np.ceil(cx + bw * CONTEXT_SCALE / 2))), min(h, int(np.ceil(cy + bh * CONTEXT_SCALE / 2)))
    ring = np.ones((y1 - y0, x1 - x0), bool)
    gx0, gy0 = max(x0, int(np.floor(box[0])) - 2), max(y0, int(np.floor(box[1])) - 2)
    gx1, gy1 = min(x1, int(np.ceil(box[2])) + 2), min(y1, int(np.ceil(box[3])) + 2)
    ring[gy0 - y0:gy1 - y0, gx0 - x0:gx1 - x0] = False
    return (x0, y0, x1, y1), ring


def find_references(data, rows):
    """rows(한 분할의 manifest 행) 안에서 이물마다 기준 프레임을 찾는다. 반환: dict 목록."""
    rows = rows.reset_index(drop=True)
    imgs = {r.id: load_gray(data, r.id) for r in rows.itertuples()}
    boxes = {r.id: load_boxes(data, r.id, r.w, r.h) for r in rows.itertuples()}
    times = {r.id: shot_time(r.id) for r in rows.itertuples()}
    out = []
    for r in rows.itertuples():
        g = imgs[r.id].astype(np.float32)
        h, w = g.shape
        cands = [c for c in rows.itertuples() if c.machine == r.machine and c.id != r.id
                 and imgs[c.id].shape == g.shape and abs((times[c.id] - times[r.id]).total_seconds()) <= MAX_GAP]
        for k, box in enumerate(boxes[r.id]):
            (x0, y0, x1, y1), ring = _ring(box, w, h)
            if ring.sum() < 10:
                continue
            best = None
            for c in cands:
                if any(_iou(box, b) > MAX_REF_IOU for b in boxes[c.id]):
                    continue
                d = np.abs(g[y0:y1, x0:x1] - imgs[c.id][y0:y1, x0:x1].astype(np.float32))
                ring_mae, center_mae = float(d[ring].mean()), float(d[~ring].mean())
                if ring_mae <= RING_MAX and center_mae - ring_mae >= CENTER_MIN and (best is None or ring_mae < best["ring_mae"]):
                    best = dict(id=r.id, k=k, machine=int(r.machine), ref=c.id, ring_mae=ring_mae,
                                center_minus_ring=center_mae - ring_mae,
                                gap=abs((times[c.id] - times[r.id]).total_seconds()),
                                box=(int(np.floor(box[0])), int(np.floor(box[1])), int(np.ceil(box[2])), int(np.ceil(box[3]))))
            if best:
                out.append(best)
    return out


def feather_mask(h, w, frac=FEATHER):
    yy, xx = np.mgrid[0:h, 0:w]
    edge = np.minimum(np.minimum(xx, w - 1 - xx), np.minimum(yy, h - 1 - yy)).astype(np.float32)
    a = np.clip(edge / max(1.0, min(h, w) * frac), 0, 1)
    return a * a * (3 - 2 * a)


def extract(source, ref, box, mode="mul", debias=True):
    """박스 영역의 이물 조각. mul: 투과율(곱해서 넣는다), add: 밝기 차(더해서 넣는다). 경계는 부드럽게 1(또는 0)로 줄인다.

    두 사진은 같은 자리라도 배경 밝기가 조금 다르다(고리 차 중앙값 약 12단계). 그대로 떼면 조각 전체가 한쪽으로 치우쳐,
    옮긴 자리에 박스 모양의 어둡거나 밝은 네모가 생긴다 (이물이 아니라 네모를 배울 수 있는 단서).
    debias=True 면 조각 테두리 2px 의 중앙값을 배경 수준으로 보고 빼(나눠) 준다.
    """
    x0, y0, x1, y1 = box
    s, r = source[y0:y1, x0:x1].astype(np.float32), ref[y0:y1, x0:x1].astype(np.float32)
    a = feather_mask(*s.shape)
    border = np.ones(s.shape, bool)
    border[2:-2, 2:-2] = False
    if mode == "add":
        d = s - r
        return (d - (np.median(d[border]) if debias else 0)) * a
    t = (s + 1) / (r + 1)
    if debias:
        t = t / np.median(t[border])
    return 1 + (np.clip(t, 0.02, 1.3) - 1) * a


def border_offset(source, ref, box):
    """치우침 보정 전 조각 테두리의 배경 비율 (1 이면 치우침 없음). 진단용."""
    x0, y0, x1, y1 = box
    t = (source[y0:y1, x0:x1].astype(np.float32) + 1) / (ref[y0:y1, x0:x1].astype(np.float32) + 1)
    border = np.ones(t.shape, bool)
    border[2:-2, 2:-2] = False
    return float(np.median(t[border]))


def apply(f, x0, y0, patch, s=1.0, mode="mul"):
    """f(float32)의 (x0, y0) 에서 시작하는 자리에 조각을 세기 s 로 넣는다. 영상 밖이면 False."""
    ph, pw = patch.shape
    if x0 < 0 or y0 < 0 or x0 + pw > f.shape[1] or y0 + ph > f.shape[0]:
        return False
    if mode == "add":
        f[y0:y0 + ph, x0:x0 + pw] += patch * s
    else:
        f[y0:y0 + ph, x0:x0 + pw] *= patch ** s
    return True


def build_ref_bank(data, rows, mode="mul", debias=True):
    """{호기: [dict(id, patch, bw, bh, desc)]} 와 찾은 기준 프레임 목록. desc 는 자리 고르기에 쓰는 원래 자리의 둘레 구조."""
    refs = find_references(data, rows)
    bank = {}
    cache = {}
    for r in refs:
        g = cache.setdefault(r["id"], load_gray(data, r["id"]))
        ref = cache.setdefault(r["ref"], load_gray(data, r["ref"]))
        x0, y0, x1, y1 = r["box"]
        if x1 - x0 < 3 or y1 - y0 < 3:
            continue
        r["border_offset"] = border_offset(g, ref, r["box"])
        patch = extract(g, ref, r["box"], mode, debias)
        r["mismatch_area"] = float((np.abs(patch - (1 if mode == "mul" else 0)) > (0.08 if mode == "mul" else 8)).mean())
        r["used"] = bool(not debias or r["mismatch_area"] <= MAX_MISMATCH)
        if not r["used"]:
            continue
        cp = ContextPicker(g, None)
        desc = cp.describe((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0)
        h, w = g.shape
        bank.setdefault(r["machine"], []).append(dict(id=r["id"], patch=patch, bw=x1 - x0, bh=y1 - y0, desc=desc,
                                                      src=((x0 + x1) / 2 / w, (y0 + y1) / 2 / h)))
    return bank, refs


class ContextPicker:
    """한 영상에서 '원래 자리와 둘레 구조가 닮은 자리'를 고른다."""

    def __init__(self, gray, product_mask):
        self.g = gray
        self.h, self.w = gray.shape
        self.blur = cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32)
        self.gx = cv2.Sobel(self.blur, cv2.CV_32F, 1, 0, ksize=3)
        self.gy = cv2.Sobel(self.blur, cv2.CV_32F, 0, 1, ksize=3)
        self.pm = product_mask
        self.dist = cv2.distanceTransform((product_mask > 0).astype(np.uint8), cv2.DIST_L2, 5) if product_mask is not None else None

    @staticmethod
    def _shape(bw, bh):
        pw, ph = max(bw + 8, int(round(4.0 * bw))) | 1, max(bh + 8, int(round(4.0 * bh))) | 1
        valid = np.ones((ph, pw), bool)
        ew, eh = max(bw + 4, int(round(1.5 * bw))), max(bh + 4, int(round(1.5 * bh)))
        valid[max(0, ph // 2 - eh // 2):ph // 2 + (eh + 1) // 2, max(0, pw // 2 - ew // 2):pw // 2 + (ew + 1) // 2] = False
        return pw, ph, valid

    def describe(self, cx, cy, bw, bh):
        """(표준화한 둘레 밝기, 기울기 크기, 가로 기울기 비) 또는 영상 밖이면 None."""
        pw, ph, valid = self._shape(bw, bh)
        x0, y0 = int(round(cx)) - pw // 2, int(round(cy)) - ph // 2
        if x0 < 0 or y0 < 0 or x0 + pw > self.w or y0 + ph > self.h or valid.sum() < 20:
            return None
        sl = (slice(y0, y0 + ph), slice(x0, x0 + pw))
        v = self.blur[sl][valid]
        sd = max(float(v.std()), 1.0)
        ax, ay = np.abs(self.gx[sl][valid]), np.abs(self.gy[sl][valid])
        return dict(norm=(v - v.mean()) / sd, mag=np.hypot(ax, ay) / sd, orient=float(ax.sum() / (ax.sum() + ay.sum() + 1e-6)),
                    std=float(v.std()), inside=float(self.pm[sl].mean()) if self.pm is not None else 1.0)

    def candidates(self, desc, bw, bh, boxes, src_norm=None):
        """안전 조건을 지킨 후보 자리와 비용. [(비용, 박스 왼쪽 위 x, y)] 를 비용 오름차순으로."""
        if desc is None:
            return []
        scale = max(bw, bh)
        edge_need, gt_need = max(6.0, EDGE_MULT * scale), GT_MULT * scale
        out = []
        for cy in np.linspace(bh, self.h - bh, GRID[1]).round().astype(int):
            for cx in np.linspace(bw, self.w - bw, GRID[0]).round().astype(int):
                x0, y0 = cx - bw // 2, cy - bh // 2
                if x0 < 0 or y0 < 0 or x0 + bw > self.w or y0 + bh > self.h:
                    continue
                if self.pm[y0:y0 + bh, x0:x0 + bw].min() == 0 or self.dist[y0:y0 + bh, x0:x0 + bw].min() < edge_need:
                    continue
                if any(not (x0 + bw + gt_need <= b[0] or b[2] + gt_need <= x0 or y0 + bh + gt_need <= b[1] or b[3] + gt_need <= y0)
                       for b in boxes):
                    continue
                if src_norm is not None and np.hypot(cx / self.w - src_norm[0], cy / self.h - src_norm[1]) < MIN_SHIFT:
                    continue
                d = self.describe(cx, cy, bw, bh)
                if d is None or d["inside"] < 0.90 or d["std"] < MIN_STD or len(d["norm"]) != len(desc["norm"]):
                    continue
                cost = float(np.abs(d["norm"] - desc["norm"]).mean() + 0.35 * np.abs(d["mag"] - desc["mag"]).mean()
                             + 0.50 * abs(d["orient"] - desc["orient"]))
                out.append((cost, int(x0), int(y0)))
        out.sort()
        return out
