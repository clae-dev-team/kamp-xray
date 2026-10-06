"""조각 분류 CNN: '32×32 조각 한가운데에 이물이 있는가'를 가려내는 작은 합성곱 신경망.

베이스라인(규칙)과 YOLO(탐지 전용 대형 모델) 사이의 비교 모델이다.
  - 학습: 이물 중심 조각(양성) vs 이물 없는 조각(음성)을 뽑아 이진 분류로 배운다.
      음성 = 제품 안 무작위 자리 + 베이스라인이 헷갈린 자리(띠 모서리 등) + 학습 중 CNN이 틀린 자리(어려운 음성 재수집)
  - 추론: 패딩 없는 합성곱만 써서 영상 전체에 한 번에 적용하면 4px 간격으로 조각을 다 훑은 것과 같다.
      2px씩 밀어 4번 돌려 2px 간격 확률 지도를 만들고, 국소 최댓값에 호기별 중앙 크기 박스를 씌운다.
  - 채점: metrics.py 공용 기준. 판정 임계값은 val, test는 마지막에 한 번만.
학습 데이터는 최종 YOLO(ratio3_e100)와 같은 data/aug/train.txt (실제 350장 + 합성 1,050장)를 기본으로 쓴다.

실행 예:
  .venv\\Scripts\\python.exe src\\cnn.py --name cnn_aug
  .venv\\Scripts\\python.exe src\\cnn.py --name cnn_real --train-list data/clean/train.txt   # 합성 없이
  .venv\\Scripts\\python.exe src\\cnn.py --name cnn_aug --skip-train                         # 저장된 가중치로 채점만
결과: runs/<name>/cnn.pt, results/cnn_<name>/ (pred_*.csv, metrics.json, pr_test.png, history.csv)
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import baseline as B
import metrics as M
from prepare import product_mask

ROOT = Path(__file__).resolve().parents[1]   # 저장소 최상위 폴더
# 아래 거리 · 크기 상수의 단위는 모두 원본 영상의 픽셀이다
PATCH = 32                   # 조각 크기 = 신경망 수용영역
HALF = PATCH // 2            # 조각 반쪽. 영상 둘레에 이만큼 반사 패딩해 가장자리에서도 조각을 뗀다
STRIDE = 4                   # 신경망 출력 간격 (풀링 2번)
POS_JITTER = 2               # 양성 조각 중심을 이물 중심에서 이만큼까지 흔든다
NEG_MIN_DIST = 6             # 음성 조각 중심은 모든 이물 중심에서 이만큼 떨어뜨린다
RING = (6, 9)                # 이물이 조각 안에 있지만 가운데가 아닌 거리 → 음성 ('가운데에 있는가'를 날카롭게)
LOGIT_MIN = -4.6             # 후보로 남길 최소 로짓 (확률 0.01, PR 곡선용)
NMS_R = 6                    # 이 거리(px) 안의 더 낮은 봉우리는 버린다
RECALL_TARGET = 0.95         # 재현율 우선 임계값의 목표 재현율 (베이스라인 · YOLO 와 같은 값)


class PatchNet(nn.Module):
    """패딩 없는 합성곱: 32×32 입력 → 1×1 출력. 큰 영상에 넣으면 4px 간격 확률 지도가 나온다.

    입력: (묶음, 1, H, W) 흑백(to_tensor 로 정규화한 값). 출력: (묶음, 1, H', W') 로짓(시그모이드 전 값).
    w: 첫 단의 채널 수. 단이 깊어지며 2배, 4배로 늘어난다.
    """

    def __init__(self, w=32):
        super().__init__()

        def block(i, o):
            """3×3 합성곱(패딩 없음) + 배치 정규화 + ReLU. 지날 때마다 가로 · 세로가 2씩 준다."""
            return [nn.Conv2d(i, o, 3), nn.BatchNorm2d(o), nn.ReLU(inplace=True)]

        # 옆의 숫자는 32×32 조각을 넣었을 때 한 변의 길이 변화다
        self.body = nn.Sequential(
            *block(1, w), *block(w, w), nn.MaxPool2d(2),               # 32→28→14
            *block(w, 2 * w), *block(2 * w, 2 * w), nn.MaxPool2d(2),   # 14→10→5
            *block(2 * w, 4 * w), *block(4 * w, 4 * w),                # 5→1
            nn.Conv2d(4 * w, 1, 1))                                    # 1×1 합성곱으로 로짓 하나

    def forward(self, x):
        """x: (묶음, 1, H, W). 반환: (묶음, 1, H', W') 로짓. H = W = 32 이면 H' = W' = 1."""
        return self.body(x)


def to_tensor(gray):
    """0~255 화소값을 신경망 입력 범위로 바꾼다: (값/255 - 0.5) / 0.25, 곧 -2~2. 배열 모양은 그대로다."""
    return (np.asarray(gray, np.float32) / 255.0 - 0.5) / 0.25


# ---------------------------------------------------------------- 데이터
def label_path(img_path: Path) -> Path:
    """영상 경로에 짝이 되는 라벨 경로. .../images/<이름>.png → .../labels/<이름>.txt"""
    return img_path.parent.parent / "labels" / (img_path.stem + ".txt")


def centers(img_path, w, h):
    """영상의 이물 중심 좌표. w, h: 영상 크기(px). 반환: (N, 2) 배열, 열은 (x, y) 픽셀.

    라벨 파일이 없거나 비어 있으면 (0, 2) 를 돌려준다.
    """
    lp = label_path(img_path)
    if not lp.exists() or lp.stat().st_size == 0:
        return np.zeros((0, 2))
    b = np.loadtxt(lp, ndmin=2)
    return np.stack([b[:, 1] * w, b[:, 2] * h], 1)   # YOLO 라벨의 1 · 2열 = 중심 x · y (0~1 비율)


def crop(padded, cx, cy):
    """padded = 원본을 HALF 만큼 반사 패딩한 영상. (cx, cy) 픽셀을 중심으로 32×32.

    cx, cy 는 원본 좌표다. 패딩 때문에 원본의 x 는 padded 의 x + HALF 에 있으므로,
    padded 의 x 에서 PATCH 만큼 자르면 원본의 x - HALF ~ x + HALF 구간이 된다.
    """
    x, y = int(round(cx)), int(round(cy))
    return padded[y:y + PATCH, x:x + PATCH]


def far_from(pts, c, dist):
    """각 점이 모든 이물 중심에서 dist(px) 이상 떨어져 있는가.

    pts: (N, 2), c: (M, 2), 둘 다 열은 (x, y) 픽셀. 반환: (N,) bool. 이물이 없으면 모두 True.
    """
    if len(c) == 0:
        return np.ones(len(pts), bool)
    # (N, M) 거리표를 만들어 점마다 가장 가까운 이물까지의 거리를 본다
    d = np.hypot(pts[:, None, 0] - c[None, :, 0], pts[:, None, 1] - c[None, :, 1]).min(1)
    return d >= dist


class Pool:
    """영상과 음성 후보 자리를 메모리에 올려 두고 에폭마다 조각을 새로 뽑는다.

    items 의 원소(영상 한 장): pad = 반사 패딩한 영상, c = 이물 중심 (N, 2), prod = 제품 영역 화소 좌표,
    hard = 베이스라인 후보 자리, mined = CNN 이 틀린 자리(mine 이 채움), shape = 원본 (높이, 너비).
    좌표 배열은 모두 (x, y) 픽셀 순서다.
    """

    def __init__(self, paths, bl_prm, box_by_machine, machine_of, rng):
        """paths: 학습 영상 경로 목록, bl_prm: 베이스라인이 고른 값(se, sigma, score),
        box_by_machine: {호기: (너비, 높이) px}, machine_of: 경로 → 호기 함수, rng: numpy 난수 생성기."""
        self.rng = rng
        self.items = []
        boxes = {int(k): v for k, v in box_by_machine.items()}
        for p in tqdm(paths, desc="학습 영상 읽기"):
            g = np.asarray(Image.open(p).convert("L"))
            h, w = g.shape
            c = centers(p, w, h)
            pm = product_mask(g) > 0   # prepare.py 의 제품 영역(경계를 15×15 로 깎은 것)
            ys, xs = np.nonzero(pm)
            # 베이스라인이 이물로 착각할 만한 자리 = 어려운 음성 후보
            d = B.detect(g, bl_prm["se"], bl_prm["sigma"], bl_prm["score"], boxes[machine_of(p)])
            hard = np.stack([(d.x0 + d.x1).to_numpy() / 2, (d.y0 + d.y1).to_numpy() / 2], 1) if len(d) else np.zeros((0, 2))
            # 베이스라인이 맞게 찾은 진짜 이물 자리는 음성 후보에서 뺀다
            hard = hard[far_from(hard, c, NEG_MIN_DIST)] if len(hard) else hard
            self.items.append(dict(pad=np.pad(g, HALF, mode="reflect"), c=c, prod=np.stack([xs, ys], 1),
                                   hard=hard, mined=np.zeros((0, 2)), shape=(h, w)))

    def sample(self, pos_rep=4, n_ring=2, n_rand=16, n_hard=8, n_mined=8):
        """한 에폭 분량의 조각을 뽑는다.

        pos_rep: 이물 하나를 양성으로 뽑는 횟수(매번 중심을 다르게 흔든다).
        n_ring: 한 번 돌 때 이물 옆(RING 거리)에서 뽑는 음성 수.
        n_rand, n_hard, n_mined: 영상당 무작위 · 베이스라인 후보 · 재수집 음성의 최대 수.
        반환: X (조각 수, 32, 32) float32 화소값 0~255, Y (조각 수,) float32 양성 1 · 음성 0.
        """
        X, Y = [], []
        r = self.rng
        for it in self.items:
            c = it["c"]
            for _ in range(pos_rep):
                # 양성: 이물 중심에서 가로 · 세로 ±POS_JITTER px 안으로 흔든 자리
                for cx, cy in c:
                    j = r.integers(-POS_JITTER, POS_JITTER + 1, 2)
                    X.append(crop(it["pad"], cx + j[0], cy + j[1]))
                    Y.append(1)
                # 고리 음성: 이물이 조각 안에 보이지만 가운데는 아닌 자리.
                # 영상마다 이물 하나(목록의 마지막 이물) 둘레에서 뽑고, 다른 이물과 가까우면 버린다
                for _ in range(n_ring):
                    a, rr = r.uniform(0, 2 * np.pi), r.uniform(*RING)   # 방향(라디안), 거리(px)
                    q = np.array([[cx + rr * np.cos(a), cy + rr * np.sin(a)]])
                    if far_from(q, c, NEG_MIN_DIST)[0]:
                        X.append(crop(it["pad"], *q[0]))
                        Y.append(0)
            # 무작위 음성: 이물 근처를 걸러 낼 것을 감안해 3배로 뽑은 뒤 앞에서 n_rand 개만 쓴다
            cand = it["prod"][r.integers(0, len(it["prod"]), n_rand * 3)]
            cand = cand[far_from(cand.astype(float), c, NEG_MIN_DIST)][:n_rand]
            for src, n in [(cand, n_rand), (it["hard"], n_hard), (it["mined"], n_mined)]:
                if len(src) == 0:
                    continue
                # 복원 추출이라 같은 자리가 두 번 뽑힐 수 있다
                for cx, cy in src[r.integers(0, len(src), min(n, len(src)))]:
                    X.append(crop(it["pad"], cx, cy))
                    Y.append(0)
        X = np.stack(X).astype(np.float32)
        return X, np.array(Y, np.float32)


def augment(x, rng):
    """좌우·상하 뒤집기(YOLO 학습과 같음)와 밝기 ±, 대비 ±10%.

    x: (조각 수, 32, 32) float32 화소값 0~255. 뒤집기는 x 자체를 바꾼다.
    반환: 같은 모양의 새 배열(0~255 로 자른 값).
    """
    n = len(x)
    f = rng.random(n) < 0.5      # 조각마다 절반 확률
    x[f] = x[f, :, ::-1]         # 축 2(가로)를 뒤집는다 = 좌우
    f = rng.random(n) < 0.5
    x[f] = x[f, ::-1, :]         # 축 1(세로)을 뒤집는다 = 상하
    gain = rng.uniform(0.9, 1.1, (n, 1, 1)).astype(np.float32)   # 대비 배율
    off = rng.uniform(-10, 10, (n, 1, 1)).astype(np.float32)     # 밝기 이동 (회색 단계)
    # 조각 평균을 기준으로 대비를 늘리거나 줄인 뒤 밝기를 옮긴다
    return np.clip((x - x.mean((1, 2), keepdims=True)) * gain + x.mean((1, 2), keepdims=True) + off, 0, 255)


# ---------------------------------------------------------------- 추론
@torch.no_grad()
def prob_map(model, gray, device):
    """2px 간격 로짓 지도. map[i, j] ↔ 조각 중심 연속 좌표 (2j, 2i).
    확률(sigmoid)은 이물 근처에서 1.0으로 포화돼 봉우리가 평평해지므로 로짓 그대로 돌려준다.

    gray: (h, w) 흑백 영상(0~255). 반환: ((h+1)//2, (w+1)//2) float32 로짓.
    신경망이 값을 내지 못한 칸은 -30 으로 남는다.
    """
    h, w = gray.shape
    pad = np.pad(gray, HALF, mode="reflect").astype(np.float32)
    H, W = (h + 1) // 2, (w + 1) // 2
    out = np.full((H, W), -30.0, np.float32)   # -30: 확률로 치면 사실상 0
    # 신경망 출력은 4px 간격이다. 입력을 세로 · 가로로 0 또는 2px 밀어 4번 돌리고 엇갈려 끼우면 2px 간격이 된다
    for dy in (0, 2):
        for dx in (0, 2):
            # [None, None]: (H, W) → (1, 1, H, W) 로 묶음 · 채널 축을 붙인다
            t = torch.from_numpy(to_tensor(pad[dy:, dx:]))[None, None].to(device)
            o = model(t)[0, 0].float().cpu().numpy()
            # o[a, b] 는 원본 좌표 (dx + 4b, dy + 4a) 가 중심인 조각의 로짓이고, 지도에서는 [dy/2 + 2a, dx/2 + 2b] 칸이다
            ny = min(len(range(dy // 2, H, 2)), o.shape[0])
            nx = min(len(range(dx // 2, W, 2)), o.shape[1])
            out[dy // 2:dy // 2 + 2 * ny:2, dx // 2:dx // 2 + 2 * nx:2] = o[:ny, :nx]
    return out


def detect(model, gray, box_wh, device, return_map=False):
    """영상 한 장에서 이물 후보를 찾는다.

    gray: (h, w) 흑백 영상(0~255). box_wh: 후보에 씌울 박스의 (너비, 높이) px.
    반환: 열 x0, y0, x1, y1(원본 픽셀), score(0~1 확률) 인 표.
    return_map=True 이면 (표, 2px 간격 로짓 지도) 를 함께 돌려준다.
    """
    lm = prob_map(model, gray, device)
    H, W = lm.shape
    # 제품 영역을 9×9 로 넓힌 뒤 지도와 같은 2px 간격으로 솎는다
    prod = cv2.dilate(product_mask(gray), np.ones((9, 9), np.uint8))[::2, ::2][:H, :W] > 0
    # 봉우리: 지도의 3×3칸(원본 기준 ±2px) 안에서 최댓값이고, 제품 영역 안이며, 최소 로짓 이상인 칸
    peak = (lm == cv2.dilate(lm, np.ones((3, 3), np.uint8))) & prod & (lm >= LOGIT_MIN)
    ys, xs = np.nonzero(peak)
    order = np.argsort(-lm[ys, xs], kind="stable")   # 로짓이 높은 봉우리부터
    keep = []
    for k in order:                      # 거리 기준 NMS (같은 이물에 봉우리 여러 개 방지)
        # xs · ys 는 2px 간격 칸 번호라, NMS_R(px) 을 2로 나눠 칸 단위로 비교한다
        if all((xs[k] - xs[q]) ** 2 + (ys[k] - ys[q]) ** 2 > (NMS_R / 2) ** 2 for q in keep):
            keep.append(k)
    ys, xs = ys[keep], xs[keep]
    # 최댓값 칸만 쓰면 2px 격자만큼 흔들리므로, 주변 5×5칸에서 exp(로짓 차) 가중 무게중심으로 위치를 다듬는다
    cx, cy = np.empty(len(xs)), np.empty(len(xs))
    for k, (y, x) in enumerate(zip(ys, xs)):
        y0, y1, x0, x1 = max(y - 2, 0), min(y + 3, H), max(x - 2, 0), min(x + 3, W)   # 지도 밖으로 나가지 않게 자른다
        wt = np.exp(np.clip(lm[y0:y1, x0:x1] - lm[y, x], -20, 0))   # 봉우리 칸의 가중치가 1
        wt[wt < np.exp(-2)] = 0                                     # 봉우리보다 로짓이 2 넘게 낮은 칸은 뺀다
        gy, gx = np.mgrid[y0:y1, x0:x1]
        # 칸 번호 무게중심에 2를 곱해 원본 픽셀 좌표로 되돌린다
        cx[k], cy[k] = (gx * wt).sum() / wt.sum() * 2.0, (gy * wt).sum() / wt.sum() * 2.0
    bw, bh = box_wh
    score = 1 / (1 + np.exp(-lm[ys, xs].astype(np.float64)))   # float64라 0.9999999 대에서도 순서가 남는다
    d = pd.DataFrame({"x0": cx - bw / 2, "y0": cy - bh / 2, "x1": cx + bw / 2, "y1": cy + bh / 2, "score": score})
    return (d, lm) if return_map else d


def load(name, device=None):
    """다른 스크립트(judge·synth_eval·location_test)에서 쓰는 입구. 반환: (model, box_by_machine, device).

    name: runs/<name>/cnn.pt 의 모델 이름. model 은 평가 상태(eval)이고,
    box_by_machine 은 {호기: (너비, 높이) px} 이다. device 를 안 주면 GPU 가 있을 때 GPU 를 쓴다.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # 가중치와 함께 설정값(dict)을 저장했으므로 weights_only=False 로 읽는다
    ck = torch.load(ROOT / "runs" / name / "cnn.pt", map_location=device, weights_only=False)
    model = PatchNet(ck["width"]).to(device).eval()
    model.load_state_dict(ck["state"])
    return model, {int(k): tuple(v) for k, v in ck["box_by_machine"].items()}, device


def is_cnn(name):
    """모델 이름이 이 파일로 학습한 CNN인지 (runs/<name>/cnn.pt 유무). 아니면 YOLO로 본다."""
    return (ROOT / "runs" / name / "cnn.pt").exists()


def metrics_file(name):
    """모델의 채점 결과 파일 경로. CNN 이면 results/cnn_<name>/, 아니면 results/yolo_<name>/ 의 metrics.json."""
    return ROOT / "results" / (f"cnn_{name}" if is_cnn(name) else f"yolo_{name}") / "metrics.json"


def predict_paths(name, paths, ids, machines):
    """영상 경로 여러 개를 예측해 한 표로 모은다.

    paths · ids · machines 는 같은 길이의 목록(경로, 표에 적을 id, 호기)이다.
    반환: 열 id, x0, y0, x1, y1, score 인 표. 후보가 없는 영상은 행이 없다.
    """
    model, boxes, dev = load(name)
    out = []
    for p, i, m in tqdm(list(zip(paths, ids, machines)), desc=f"CNN {name}"):
        d = detect(model, np.asarray(Image.open(p).convert("L")), boxes[int(m)], dev)
        d.insert(0, "id", i)
        out.append(d)
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------- 학습
def mine(model, pool, device, per_img=8):
    """학습 영상에서 CNN이 높게 본 자리 중 이물이 아닌 곳을 다음 에폭 음성으로 쓴다.

    pool: Pool. 영상마다 로짓이 높은 순으로 per_img 개까지 it["mined"] 에 (x, y) 픽셀로 넣는다(이전 것은 덮어쓴다).
    반환: 전체 영상에서 모은 자리 수.
    """
    model.eval()
    n = 0
    for it in pool.items:
        h, w = it["shape"]
        g = it["pad"][HALF:HALF + h, HALF:HALF + w]   # 패딩을 걷어 내 원본 영상으로 되돌린다
        pm2 = prob_map(model, g, device)
        peak = (pm2 == cv2.dilate(pm2, np.ones((3, 3), np.uint8))) & (pm2 >= -1.4)   # 확률 0.2 이상
        ys, xs = np.nonzero(peak)
        pts = np.stack([xs * 2.0, ys * 2.0], 1)       # 2px 간격 칸 번호 → 원본 픽셀 (x, y)
        # 진짜 이물 근처의 봉우리는 정답이므로 음성에서 뺀다
        keep = far_from(pts, it["c"], NEG_MIN_DIST) if len(pts) else np.zeros(0, bool)
        pts, sc = pts[keep], pm2[ys, xs][keep]
        it["mined"] = pts[np.argsort(-sc)[:per_img]]
        n += len(it["mined"])
    model.train()
    return n


def train(args, paths, bl_prm, box_by_machine, machine_of, device):
    """조각 분류 CNN 을 학습한다.

    args: 명령줄 인자(seed, width, lr, epochs, batch, mine_from, mine_every 를 쓴다).
    나머지 인자는 Pool 에 그대로 넘긴다.
    반환: (학습된 model, 에폭별 기록 표). 표의 열은 epoch, loss, n_patch, n_pos, mined, sec 이다.
    """
    # 같은 시드면 같은 조각 · 같은 초깃값이 나오도록 난수와 cuDNN 을 고정한다
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    pool = Pool(paths, bl_prm, box_by_machine, machine_of, rng)
    model = PatchNet(args.width).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # 학습률 일정은 에폭 단위다(total_steps = 에폭 수, 에폭 끝에 한 번 step). 앞 10% 동안 올렸다가 내린다
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs, pct_start=0.1)
    hist = []
    for ep in range(args.epochs):
        t0 = time.time()
        # 어려운 음성 재수집: mine_from 번째 에폭부터 mine_every 에폭마다. 재수집하지 않은 에폭은 None 으로 적는다
        n_mined = mine(model, pool, device) if ep >= args.mine_from and (ep - args.mine_from) % args.mine_every == 0 else None
        X, Y = pool.sample()   # 에폭마다 조각을 새로 뽑는다
        X = augment(X, rng)
        order = rng.permutation(len(X))
        # 양성 손실에 (음성 수 / 양성 수) 를 곱해 두 쪽의 비중을 맞춘다
        pos_w = torch.tensor((Y == 0).sum() / max((Y == 1).sum(), 1), device=device)
        model.train()
        tot, k = 0.0, 0   # 손실 합(조각 수 가중), 본 조각 수
        for s in range(0, len(order), args.batch):
            b = order[s:s + args.batch]
            x = torch.from_numpy(to_tensor(X[b]))[:, None].to(device)   # (묶음, 32, 32) → (묶음, 1, 32, 32)
            y = torch.from_numpy(Y[b]).to(device)
            # 32×32 입력이라 출력이 (묶음, 1, 1, 1) 이다. 펴서 조각당 로짓 하나로 만든다
            loss = F.binary_cross_entropy_with_logits(model(x).flatten(), y, pos_weight=pos_w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
            k += len(b)
        sched.step()
        hist.append(dict(epoch=ep + 1, loss=round(tot / k, 5), n_patch=k, n_pos=int(Y.sum()),
                         mined=n_mined, sec=round(time.time() - t0, 1)))
        print(hist[-1], flush=True)
    return model, pd.DataFrame(hist)


def main():
    """학습(--skip-train 이면 생략) → 전 분할 예측 → 임계값 결정(val) → 채점 → 결과 저장.

    runs/<name>/cnn.pt 와 results/cnn_<name>/ (history.csv, pred_<분할>.csv, metrics.json, pr_test.png) 를 남긴다.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "data.yaml"))
    ap.add_argument("--train-list", default="data/aug/train.txt", help="학습 영상 목록 (YOLO 형식 txt)")
    ap.add_argument("--variant", default="clean", help="채점 영상")
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--width", type=int, default=32)         # PatchNet 첫 단의 채널 수
    ap.add_argument("--mine-from", type=int, default=5)      # 어려운 음성 재수집을 시작하는 에폭(0부터 셈)
    ap.add_argument("--mine-every", type=int, default=5)     # 재수집 간격(에폭)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    data = ROOT / cfg["out_dir"]
    out = ROOT / "results" / f"cnn_{args.name}"
    run_dir = ROOT / "runs" / args.name
    out.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    # 채점은 정답(라벨)이 있는 실제 영상으로만 한다
    man = pd.read_csv(data / "manifest.csv")
    man = man[man["labeled"]].set_index("id")
    sizes = {i: (r.w, r.h) for i, r in man.iterrows()}                 # {id: (너비, 높이) px}
    machine = {i: int(m) for i, m in man["machine"].items()}           # {id: 호기}
    split = {s: man.index[man["split"] == s].tolist() for s in ["train", "val", "test"]}
    img_dir, label_dir = data / args.variant / "images", data / args.variant / "labels"
    gt = {s: M.load_gt(ids, label_dir, sizes) for s, ids in split.items()}
    # 박스 크기 = 실제 train 정답의 호기별 중앙값 (베이스라인과 같음)
    # 베이스라인 결과 파일을 읽으므로 baseline.py 를 먼저 돌려야 한다
    bl = json.load(open(ROOT / "results" / "baseline_clean" / "metrics.json", encoding="utf-8"))
    box_by_machine = {int(k): tuple(v) for k, v in bl["params"]["box_by_machine"].items()}

    if not args.skip_train:
        lst = ROOT / args.train_list
        paths = [Path(l.strip()) for l in open(lst, encoding="utf-8") if l.strip()]
        paths = [p if p.is_absolute() else lst.parent / p for p in paths]   # 상대 경로는 목록 파일이 있는 폴더 기준
        # 파일 이름에서 "__" 앞부분이 원본 영상 id 다 (합성 영상은 <id>__a0 처럼 뒤에 꼬리가 붙는다)
        machine_of = lambda p: machine[p.stem.split("__")[0]]
        t0 = time.time()
        model, hist = train(args, paths, bl["params"], box_by_machine, machine_of, device)
        hist.to_csv(out / "history.csv", index=False, encoding="utf-8-sig")
        # 가중치와 함께 추론에 필요한 값(채널 수, 박스 크기)을 한 파일에 담는다
        torch.save({"state": model.state_dict(), "width": args.width, "box_by_machine": box_by_machine,
                    "args": vars(args), "train_sec": round(time.time() - t0, 1)}, run_dir / "cnn.pt")

    # 방금 학습했더라도 저장한 파일에서 다시 읽어 채점한다 (--skip-train 과 같은 길)
    model, boxes, device = load(args.name, device)
    preds = {}
    for s, ids in split.items():
        rows = []
        for i in tqdm(ids, desc=f"{s} 예측"):
            d = detect(model, np.asarray(Image.open(img_dir / f"{i}.png").convert("L")), boxes[machine[i]], device)
            d.insert(0, "id", i)
            rows.append(d)
        preds[s] = pd.concat(rows, ignore_index=True)

    # val 에서 임계값 두 가지: F1 최대, 재현율 우선(F1 기준보다 높아지지 않게 둘 중 작은 값)
    n_val = sum(len(g) for g in gt["val"].values())
    pm_val, _ = M.match(preds["val"], gt["val"])
    thr = {"F1최대": M.best_f1_threshold(pm_val, n_val)}
    thr[f"재현율{int(RECALL_TARGET * 100)}"] = min(thr["F1최대"], M.recall_threshold(pm_val, n_val, RECALL_TARGET))

    # 모든 분할 채점. metrics 의 키는 "<분할>/<임계값 이름>/<맞춤 기준>" 이다
    res = {"params": vars(args), "thresholds": thr, "metrics": {}}
    for s in ["train", "val", "test"]:
        for name, t in thr.items():
            for rule in ["center", "iou50"]:
                res["metrics"][f"{s}/{name}/{rule}"] = M.evaluate(preds[s], gt[s], t, rule)
    # 후보 전체를 임계값으로 자르지 않고 저장한다. tp · gt_idx 열은 중심 일치 기준이다
    for s, p in preds.items():
        pm, _ = M.match(p, gt[s])
        pm.to_csv(out / f"pred_{s}.csv", index=False, encoding="utf-8-sig")
    json.dump(res, open(out / "metrics.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # test PR 곡선 (중심 일치 · IoU 0.5 두 기준)
    plt.rcParams["font.family"] = "Malgun Gothic"
    fig, ax = plt.subplots(figsize=(5, 4.2), dpi=150)
    n_test = sum(len(g) for g in gt["test"].values())
    for rule, c in [("center", "#1f5fa8"), ("iou50", "#9aa5b1")]:
        pm, _ = M.match(preds["test"], gt["test"], rule)
        _, prec, rec, apv = M.pr_curve(pm, n_test)
        ax.plot(rec, prec, color=c, lw=1.8, label=f"{rule}  AP {apv:.3f}")
    ax.set(xlabel="재현율", ylabel="정밀도", xlim=(0, 1), ylim=(0, 1.02), title=f"조각 분류 CNN {args.name} · test")
    ax.grid(alpha=.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "pr_test.png")

    for k, v in res["metrics"].items():
        if k.startswith(("val", "test")):
            print(k, {x: v[x] for x in ["AP", "thr", "TP", "FP", "FN", "precision", "recall", "F1"]})


if __name__ == "__main__":
    main()
