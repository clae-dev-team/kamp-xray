"""인접 프레임 기준 이식과 구조 맞춤 자리 고르기.

팀 내 다른 분석(위치 강건성 분석: 같은 세션의 인접 프레임을 깨끗한 기준으로 삼아 실제 이물의 차이를 떼어 내고,
주변 구조가 비슷한 자리에 옮기는 방법)의 핵심 절차를 이 저장소의 분할 · 전처리 · 채점 구조에 맞춰 다시 작성한 모듈이다.

augment_paste.py 의 이식은 "같은 사진에서 이물을 지운 배경"을 기준으로 삼는다. 여기서는 지운 배경 대신
**같은 호기에서 몇 초 차이로 찍힌 다른 사진의 같은 자리**(그 자리에 이물이 없는 것)를 기준으로 쓴다. 지운 흔적에 기대지 않는다.

1. 기준 프레임 찾기 (find_references)
   같은 호기, 촬영 시각 차 MAX_GAP 초 이하(따라서 사실상 같은 날), 같은 크기, 이물 자리에 상대 사진의 정답 박스가 겹치지 않음(IoU ≤ 0.01).
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
CONTEXT_SCALE = 3.0       # 둘레를 보는 영역의 크기 (박스의 배수)
MAX_REF_IOU = 0.01        # 기준 후보 사진의 정답 박스와 이 값보다 많이 겹치면 그 사진은 쓰지 않는다
RING_MAX = 15.0           # 고리의 평균 밝기 차 상한 (밝기 단계, 0~255 눈금)
CENTER_MIN = 10.0         # 중심 차 - 고리 차의 하한 (밝기 단계)
FEATHER = 0.20            # 조각 경계에서 부드럽게 줄이는 폭 (조각 짧은 변 대비 비율)
MIN_SHIFT = 0.15          # 영상 크기 대비 최소 이동량
EDGE_MULT = 2.5           # 제품 가장자리 여유 (이물 크기의 배수)
GT_MULT = 1.5             # 기존 이물과의 여유
MIN_STD = 3.0             # 밋밋한 자리는 후보에서 뺀다
GRID = (48, 40)           # 후보 격자 (가로, 세로)
MAX_MISMATCH = 0.20       # 치우침 보정 뒤에도 8% 넘게 다른 화소가 조각의 이 비율을 넘으면 버린다 (배경 구조가 안 맞는 기준 프레임)


def shot_time(image_id):
    """영상 id 에 들어 있는 _YYYYMMDD_HHMMSS 를 촬영 시각(datetime)으로 읽는다."""
    m = re.search(r"_(\d{8})_(\d{6})", image_id)
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")


def load_gray(data, image_id):
    """정제 영상(data/clean/images/<id>.png)을 (H, W) uint8 흑백 배열로 읽는다. data = 전처리 결과 폴더."""
    return np.asarray(Image.open(data / "clean/images" / f"{image_id}.png").convert("L"))


def load_boxes(data, image_id, w, h):
    """정답 박스 (N, 4) xyxy 화소.

    라벨 파일의 YOLO 형식(0~1 비율)을 영상 크기 w, h 로 곱해 (x0, y0, x1, y1) 실수 좌표로 바꾼다.
    """
    b = np.loadtxt(data / "clean/labels" / f"{image_id}.txt", ndmin=2)
    cx, cy, bw, bh = b[:, 1] * w, b[:, 2] * h, b[:, 3] * w, b[:, 4] * h
    return np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)


def _iou(a, b):
    """두 박스 (x0, y0, x1, y1) 의 겹친 넓이 / 합친 넓이 (0~1). 안 겹치면 0."""
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def _ring(box, w, h):
    """박스의 3배 영역(context)과, 그 안에서 박스 + 2px 를 뺀 고리 마스크.

    box: (x0, y0, x1, y1) 화소, w · h: 영상 크기.
    반환: ((x0, y0, x1, y1) 영상 안으로 자른 3배 영역, 그 영역 크기의 bool 마스크). 마스크는 고리 True · 가운데(박스 + 2px) False.
    """
    bw, bh = box[2] - box[0], box[3] - box[1]
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    x0, y0 = max(0, int(np.floor(cx - bw * CONTEXT_SCALE / 2))), max(0, int(np.floor(cy - bh * CONTEXT_SCALE / 2)))
    x1, y1 = min(w, int(np.ceil(cx + bw * CONTEXT_SCALE / 2))), min(h, int(np.ceil(cy + bh * CONTEXT_SCALE / 2)))
    ring = np.ones((y1 - y0, x1 - x0), bool)
    # 가운데 구멍 = 박스를 바깥으로 올림·내림한 뒤 2px 넓힌 범위. 마스크는 3배 영역의 왼쪽 위(x0, y0)가 원점이다
    gx0, gy0 = max(x0, int(np.floor(box[0])) - 2), max(y0, int(np.floor(box[1])) - 2)
    gx1, gy1 = min(x1, int(np.ceil(box[2])) + 2), min(y1, int(np.ceil(box[3])) + 2)
    ring[gy0 - y0:gy1 - y0, gx0 - x0:gx1 - x0] = False
    return (x0, y0, x1, y1), ring


def find_references(data, rows):
    """rows(한 분할의 manifest 행) 안에서 이물마다 기준 프레임을 찾는다. 반환: dict 목록.

    기준 프레임을 찾은 이물만 들어간다. 한 dict 는
      id(이물이 있는 사진), k(그 사진 안 박스 번호), machine, ref(기준 사진 id),
      ring_mae(고리 평균 밝기 차), center_minus_ring(중심 차 - 고리 차), gap(촬영 시각 차, 초),
      box(박스를 바깥으로 올림·내림한 정수 (x0, y0, x1, y1)).
    """
    rows = rows.reset_index(drop=True)
    # 분할 안의 영상·박스·촬영 시각을 한 번에 읽어 둔다 (사진끼리 여러 번 맞대어 보므로)
    imgs = {r.id: load_gray(data, r.id) for r in rows.itertuples()}
    boxes = {r.id: load_boxes(data, r.id, r.w, r.h) for r in rows.itertuples()}
    times = {r.id: shot_time(r.id) for r in rows.itertuples()}
    out = []
    for r in rows.itertuples():
        g = imgs[r.id].astype(np.float32)
        h, w = g.shape
        # 기준 후보 = 같은 호기, 다른 사진, 같은 크기, 촬영 시각 차 MAX_GAP 초 이하 (날짜는 시각 차 조건으로 걸러진다)
        cands = [c for c in rows.itertuples() if c.machine == r.machine and c.id != r.id
                 and imgs[c.id].shape == g.shape and abs((times[c.id] - times[r.id]).total_seconds()) <= MAX_GAP]
        for k, box in enumerate(boxes[r.id]):
            (x0, y0, x1, y1), ring = _ring(box, w, h)
            # 고리 픽셀이 10개도 안 되면 배경 차를 믿을 수 없어 건너뛴다
            if ring.sum() < 10:
                continue
            best = None
            for c in cands:
                # 상대 사진의 같은 자리에도 이물이 있으면 기준으로 쓸 수 없다
                if any(_iou(box, b) > MAX_REF_IOU for b in boxes[c.id]):
                    continue
                # 두 사진의 같은 자리 밝기 차(절댓값)를 고리와 가운데로 나눠 평균한다
                # 고리 차가 작고(배경이 비슷함) 가운데 차가 그보다 충분히 커야(이물이 한쪽에만 있음) 기준으로 삼는다
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
    """조각 경계에서 0, 안쪽에서 1 인 가중치 (h, w) float32. 경계 근처는 부드럽게 이어진다.

    frac: 줄이는 띠의 폭 (짧은 변 대비 비율, 최소 1px).
    """
    yy, xx = np.mgrid[0:h, 0:w]
    # edge = 가장 가까운 변까지의 거리(px). 맨 바깥 줄이 0 이다
    edge = np.minimum(np.minimum(xx, w - 1 - xx), np.minimum(yy, h - 1 - yy)).astype(np.float32)
    a = np.clip(edge / max(1.0, min(h, w) * frac), 0, 1)
    # smoothstep 3a² - 2a³: 양 끝(0, 1)에서 기울기가 0 이라 경계가 꺾이지 않는다
    return a * a * (3 - 2 * a)


def extract(source, ref, box, mode="mul", debias=True):
    """박스 영역의 이물 조각. mul: 투과율(곱해서 넣는다), add: 밝기 차(더해서 넣는다). 경계는 부드럽게 1(또는 0)로 줄인다.

    두 사진은 같은 자리라도 배경 밝기가 조금 다르다(고리 차 중앙값 약 12단계). 그대로 떼면 조각 전체가 한쪽으로 치우쳐,
    옮긴 자리에 박스 모양의 어둡거나 밝은 네모가 생긴다 (이물이 아니라 네모를 배울 수 있는 단서).
    debias=True 면 조각 테두리 2px 의 중앙값을 배경 수준으로 보고 빼(나눠) 준다.

    source: 이물이 있는 사진, ref: 기준 사진 (둘 다 (H, W) 흑백), box: 정수 (x0, y0, x1, y1), x1 · y1 은 끝 다음 칸.
    반환: 박스 크기 (y1-y0, x1-x0) float32. mul 은 이물 없는 곳이 1 인 투과율, add 는 이물 없는 곳이 0 인 밝기 차.
    """
    x0, y0, x1, y1 = box
    s, r = source[y0:y1, x0:x1].astype(np.float32), ref[y0:y1, x0:x1].astype(np.float32)
    a = feather_mask(*s.shape)
    # border = 조각 바깥 2px 테두리 (배경 수준을 재는 곳). 조각이 4px 이하면 전체가 테두리가 된다
    border = np.ones(s.shape, bool)
    border[2:-2, 2:-2] = False
    if mode == "add":
        d = s - r
        return (d - (np.median(d[border]) if debias else 0)) * a
    # 기준 픽셀이 0 이어도 나눌 수 있게 양쪽에 1 을 더한 비율
    t = (s + 1) / (r + 1)
    if debias:
        t = t / np.median(t[border])
    # 투과율을 0.02~1.3 으로 자르고, 경계로 갈수록 1(변화 없음)에 가까워지게 가중치 a 로 섞는다
    return 1 + (np.clip(t, 0.02, 1.3) - 1) * a


def border_offset(source, ref, box):
    """치우침 보정 전 조각 테두리의 배경 비율 (1 이면 치우침 없음). 진단용."""
    x0, y0, x1, y1 = box
    t = (source[y0:y1, x0:x1].astype(np.float32) + 1) / (ref[y0:y1, x0:x1].astype(np.float32) + 1)
    border = np.ones(t.shape, bool)
    border[2:-2, 2:-2] = False
    return float(np.median(t[border]))


def apply(f, x0, y0, patch, s=1.0, mode="mul"):
    """f(float32)의 (x0, y0) 에서 시작하는 자리에 조각을 세기 s 로 넣는다. 영상 밖이면 False.

    (x0, y0) 는 조각의 왼쪽 위 픽셀. mul 은 patch^s 를 곱하고(두께 s 배), add 는 patch·s 를 더한다. f 를 제자리에서 고친다.
    """
    ph, pw = patch.shape
    if x0 < 0 or y0 < 0 or x0 + pw > f.shape[1] or y0 + ph > f.shape[0]:
        return False
    if mode == "add":
        f[y0:y0 + ph, x0:x0 + pw] += patch * s
    else:
        f[y0:y0 + ph, x0:x0 + pw] *= patch ** s
    return True


def build_ref_bank(data, rows, mode="mul", debias=True):
    """{호기: [dict(id, patch, bw, bh, desc)]} 와 찾은 기준 프레임 목록. desc 는 자리 고르기에 쓰는 원래 자리의 둘레 구조.

    patch: extract 결과, bw · bh: 조각 가로·세로(px), src: 원래 자리 중심을 영상 크기로 나눈 (x, y) 0~1.
    기준 프레임 목록(refs)의 각 dict 에는 border_offset, mismatch_area, used(은행에 넣었는지)를 덧붙인다.
    """
    refs = find_references(data, rows)
    bank = {}
    cache = {}
    for r in refs:
        g = cache.setdefault(r["id"], load_gray(data, r["id"]))
        ref = cache.setdefault(r["ref"], load_gray(data, r["ref"]))
        x0, y0, x1, y1 = r["box"]
        # 한 변이 3px 보다 작은 박스는 쓰지 않는다
        if x1 - x0 < 3 or y1 - y0 < 3:
            continue
        r["border_offset"] = border_offset(g, ref, r["box"])
        patch = extract(g, ref, r["box"], mode, debias)
        # 변화 없음(mul 은 1, add 는 0)에서 크게 벗어난 픽셀의 비율. 기준은 mul 8%, add 밝기 8단계
        # 이물은 조각의 일부만 차지하므로, 이 비율이 MAX_MISMATCH 를 넘으면 배경 구조가 안 맞는 것으로 보고 버린다
        r["mismatch_area"] = float((np.abs(patch - (1 if mode == "mul" else 0)) > (0.08 if mode == "mul" else 8)).mean())
        r["used"] = bool(not debias or r["mismatch_area"] <= MAX_MISMATCH)
        if not r["used"]:
            continue
        # 원래 자리의 둘레 구조를 적어 둔다. 여기서는 자리를 고르지 않으므로 제품 마스크 없이 만든다
        cp = ContextPicker(g, None)
        desc = cp.describe((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0)
        h, w = g.shape
        bank.setdefault(r["machine"], []).append(dict(id=r["id"], patch=patch, bw=x1 - x0, bh=y1 - y0, desc=desc,
                                                      src=((x0 + x1) / 2 / w, (y0 + y1) / 2 / h)))
    return bank, refs


class ContextPicker:
    """한 영상에서 '원래 자리와 둘레 구조가 닮은 자리'를 고른다.

    describe 로 한 자리의 둘레 구조를 적고, candidates 로 다른 사진에서 적어 온 구조와 닮은 자리를 비용순으로 뽑는다.
    """

    def __init__(self, gray, product_mask):
        """gray: (H, W) uint8 영상, product_mask: 제품 마스크. None 이면 describe 만 쓸 수 있다 (candidates 는 마스크가 필요하다)."""
        self.g = gray
        self.h, self.w = gray.shape
        # 5×5 가우시안으로 잡음을 누른 밝기와, 그 가로(gx)·세로(gy) 방향 기울기
        self.blur = cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32)
        self.gx = cv2.Sobel(self.blur, cv2.CV_32F, 1, 0, ksize=3)
        self.gy = cv2.Sobel(self.blur, cv2.CV_32F, 0, 1, ksize=3)
        self.pm = product_mask
        # dist = 제품 마스크 안에서 마스크 가장자리까지의 거리(px)
        self.dist = cv2.distanceTransform((product_mask > 0).astype(np.uint8), cv2.DIST_L2, 5) if product_mask is not None else None

    @staticmethod
    def _shape(bw, bh):
        """둘레 구조를 보는 창의 크기와 쓸 픽셀 마스크. 반환: (창 가로 pw, 창 세로 ph, (ph, pw) bool 마스크).

        창은 박스의 4배(적어도 박스 + 8px)이고 홀수로 맞춘다. 가운데의 박스 1.5배(적어도 박스 + 4px)는 False 로 빼서
        이물이 놓일 자리 자체는 비교하지 않는다.
        """
        pw, ph = max(bw + 8, int(round(4.0 * bw))) | 1, max(bh + 8, int(round(4.0 * bh))) | 1
        valid = np.ones((ph, pw), bool)
        ew, eh = max(bw + 4, int(round(1.5 * bw))), max(bh + 4, int(round(1.5 * bh)))
        valid[max(0, ph // 2 - eh // 2):ph // 2 + (eh + 1) // 2, max(0, pw // 2 - ew // 2):pw // 2 + (ew + 1) // 2] = False
        return pw, ph, valid

    def describe(self, cx, cy, bw, bh):
        """(표준화한 둘레 밝기, 기울기 크기, 가로 기울기 비) 또는 영상 밖이면 None.

        (cx, cy): 자리 중심(px), bw · bh: 박스 크기(px). 반환 dict:
          norm   둘레 픽셀 밝기를 평균 0 · 표준편차 1 로 맞춘 1차원 배열 (절대 밝기와 대비 크기를 지운다)
          mag    같은 픽셀의 기울기 크기를 둘레 표준편차로 나눈 것
          orient 가로 기울기 합 / (가로 + 세로 기울기 합), 0~1 (세로 줄무늬일수록 1 에 가깝다)
          std    둘레 밝기의 표준편차 (밋밋한 자리를 거르는 데 쓴다)
          inside 창 안에서 제품이 차지하는 비율 0~1 (마스크가 없으면 1)
        """
        pw, ph, valid = self._shape(bw, bh)
        x0, y0 = int(round(cx)) - pw // 2, int(round(cy)) - ph // 2
        # 창이 영상 밖으로 나가거나 쓸 픽셀이 20개 미만이면 구조를 적지 않는다
        if x0 < 0 or y0 < 0 or x0 + pw > self.w or y0 + ph > self.h or valid.sum() < 20:
            return None
        sl = (slice(y0, y0 + ph), slice(x0, x0 + pw))
        v = self.blur[sl][valid]
        # 거의 평평한 자리에서 잡음이 크게 부풀지 않게 나누는 값은 1 이상으로 둔다
        sd = max(float(v.std()), 1.0)
        ax, ay = np.abs(self.gx[sl][valid]), np.abs(self.gy[sl][valid])
        return dict(norm=(v - v.mean()) / sd, mag=np.hypot(ax, ay) / sd, orient=float(ax.sum() / (ax.sum() + ay.sum() + 1e-6)),
                    std=float(v.std()), inside=float(self.pm[sl].mean()) if self.pm is not None else 1.0)

    def candidates(self, desc, bw, bh, boxes, src_norm=None):
        """안전 조건을 지킨 후보 자리와 비용. [(비용, 박스 왼쪽 위 x, y)] 를 비용 오름차순으로.

        desc: 옮겨 올 이물의 원래 자리 구조(describe 결과), bw · bh: 조각 크기(px),
        boxes: 피해야 할 박스 [(x0, y0, x1, y1)] (실제 이물, 먼저 넣은 이물),
        src_norm: 원래 자리 중심 (x/w, y/h). 주면 그 자리에서 MIN_SHIFT 보다 가까운 후보를 뺀다.
        """
        if desc is None:
            return []
        scale = max(bw, bh)
        # edge_need = 제품 가장자리에서 떨어져야 하는 거리(px, 최소 6), gt_need = 피할 박스와 띄울 간격(px)
        edge_need, gt_need = max(6.0, EDGE_MULT * scale), GT_MULT * scale
        out = []
        # 영상 전체에 가로 GRID[0] × 세로 GRID[1] 개의 후보 중심을 고르게 깔고 하나씩 검사한다
        for cy in np.linspace(bh, self.h - bh, GRID[1]).round().astype(int):
            for cx in np.linspace(bw, self.w - bw, GRID[0]).round().astype(int):
                x0, y0 = cx - bw // 2, cy - bh // 2
                if x0 < 0 or y0 < 0 or x0 + bw > self.w or y0 + bh > self.h:
                    continue
                # 박스 전체가 제품 안이고, 박스 안 어느 픽셀도 제품 가장자리에서 edge_need 보다 가깝지 않아야 한다
                if self.pm[y0:y0 + bh, x0:x0 + bw].min() == 0 or self.dist[y0:y0 + bh, x0:x0 + bw].min() < edge_need:
                    continue
                # 피할 박스와는 가로나 세로 어느 한쪽으로 gt_need 이상 떨어져 있어야 한다
                if any(not (x0 + bw + gt_need <= b[0] or b[2] + gt_need <= x0 or y0 + bh + gt_need <= b[1] or b[3] + gt_need <= y0)
                       for b in boxes):
                    continue
                if src_norm is not None and np.hypot(cx / self.w - src_norm[0], cy / self.h - src_norm[1]) < MIN_SHIFT:
                    continue
                d = self.describe(cx, cy, bw, bh)
                # 창의 90% 이상이 제품 안이고, 둘레가 밋밋하지 않은(표준편차 MIN_STD 이상) 자리만 남긴다
                if d is None or d["inside"] < 0.90 or d["std"] < MIN_STD or len(d["norm"]) != len(desc["norm"]):
                    continue
                # 비용 = 밝기 구조 차 + 0.35 × 기울기 크기 차 + 0.50 × 기울기 방향 비 차 (앞의 둘은 픽셀별 절댓값 차의 평균)
                cost = float(np.abs(d["norm"] - desc["norm"]).mean() + 0.35 * np.abs(d["mag"] - desc["mag"]).mean()
                             + 0.50 * abs(d["orient"] - desc["orient"]))
                out.append((cost, int(x0), int(y0)))
        out.sort()
        return out
