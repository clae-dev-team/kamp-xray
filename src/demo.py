"""검사 화면 시연: 사진 한 장을 넣으면 3단 판정 · 박스 · 놓치기 쉬운 구역을 보여 주는 작은 웹 화면.

현장에서 이 모델이 어떻게 쓰일지를 보이기 위한 것이다. 판정에 쓰는 모델 · 기준선 · 위험 지도는 모두 제출 결과와 같다.
  - 모델      : configs/pipeline.yaml 의 최종 모델 가중치
  - 기준선    : results/risk_threshold/summary.json 의 보장 기준선 (합격선 · 불합격선)
  - 위험 지도 : miss_risk.py 의 놓침 모형 (val 시험편으로 맞춘 것)
추가 패키지 없이 표준 라이브러리 http.server 로 돈다. 화면은 demo/index.html 한 파일.

보기 사진 세 묶음 (모두 학습에 쓰지 않은 test 분할)
  실제 불량 : 정제본 그대로 / 정상 : 이물 점을 지운 가짜 정상 / 옅은 이물 : 진짜 점을 옅게 옮겨 붙인 이식 시험편
직접 올린 사진에 색 표시가 있으면 prepare.py 와 같은 방법으로 지우고 판정한다 (표시를 단서로 쓰지 않는다).

판정 뒤의 일 (모델 · 기준선은 건드리지 않는다)
  - 작업자 확정   : 사람이 본 결과(이물 맞음 / 아님)를 demo/log/confirm.csv 에 쌓는다. 다음 학습 자료와 기준선 점검의 근거가 된다
  - 폴더 감시     : --watch 폴더에 사진이 들어오면 자동으로 판정해 대기 목록에 올린다 (검사기가 저장하는 폴더에 물리는 형태)
  - 자가 점검     : 정상 사진 20장에 사양 진하기의 가상 시험편을 넣어 지금도 잡는지 본다 (monitor.py 와 같은 시험편 · 같은 경보 규칙)
  - 교대 보고서   : 이번에 검사한 묶음의 판정 · 확정 · 신호 자리 · 자가 점검을 한 장으로 모은다

실행: .venv\\Scripts\\python.exe src\\demo.py [--watch demo\\inbox]  →  http://127.0.0.1:8765
"""
import argparse
import base64
import csv
import io
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import numpy as np
import pandas as pd
import yaml
from PIL import Image

import metrics as M
from conditions import Img
from miss_risk import fit_risk, ring_texture, risk_map
from prepare import color_mask, restore
from synth import insert
from train_yolo import weights_path

ROOT = Path(__file__).resolve().parents[1]        # 저장소 맨 위 폴더
# 사진 묶음의 내부 이름 → 화면에 보이는 이름. watch 는 감시 폴더로 들어온 사진이다
KINDS = {"ng": "실제 불량", "ok": "정상", "faint": "옅은 이물", "watch": "들어온 사진"}
RISK_RGB = (232, 148, 58)     # 놓치기 쉬운 구역을 칠하는 색 (R, G, B)
WEAK = 0.3                    # 2순위(약한 신호) 하한. operating_point.py · zone_rules.py 와 같은 값
LOG = ROOT / "demo" / "log"   # 작업자 확정 기록(confirm.csv)을 쌓는 폴더
DECISIONS = ("이물", "정상")  # 작업자가 고를 수 있는 확정 값 (취소는 따로 처리한다)
CHECK_N, CHECK_D, CHECK_NEAR = 20, 2.0, 7     # 자가 점검: monitor.py 의 WINDOW · PIECE_D · 맞춤 거리와 같은 값
BLUR_MAX, NOISE_MAX = 1.6, 6.0                # 모의 열화 (monitor.py 와 같은 값)
IMG_EXT = {".png", ".bmp", ".jpg", ".jpeg"}       # 감시 폴더에서 사진으로 보는 확장자


class State:
    """서버 전체가 함께 쓰는 상태. 인스턴스를 만들지 않고 클래스 속성으로 쓴다.

    setup() 이 채우는 것: 모델 · 기준선(t_low, t_high) · 목록(man) · 위험 모형(risk, high) · 보기 사진(samples, paths, truth).
    실행 중에 쌓이는 것: session(판정 기록), confirm(작업자 확정), checks(자가 점검 결과), seq(판정 순번).
    """
    lock = threading.Lock()                      # session · confirm · seq 를 여러 요청 스레드가 함께 고치므로 잠근다
    gpu = ThreadPoolExecutor(max_workers=1)      # 추론 전용 스레드 하나. 요청마다 새 스레드에서 추론하면 GPU 준비가 매번 붙어 한 장에 60ms 가 더 든다
    session, confirm, checks, seq = {}, {}, [], 0    # session · confirm 의 열쇠는 "묶음/사진" 문자열 (remember 참고)
    watch = None                                 # 감시 폴더 경로. 감시를 켜지 않으면 None


def png_url(arr):
    """배열(회색 또는 RGBA, uint8)을 PNG 로 바꿔 data URL 문자열로 돌려준다. 화면이 파일 없이 바로 그릴 수 있다."""
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def setup(name, watch=None):
    """서버를 열기 전에 한 번 부른다. 모델 · 기준선 · 위험 모형 · 보기 사진 목록을 State 에 채운다.

    name  : 모델 이름 (runs/<name> 의 가중치와 results/yolo_<name> 의 기준선을 쓴다)
    watch : 감시할 폴더 경로. 주면 감시 스레드를 띄운다
    """
    from ultralytics import YOLO

    S = State
    S.name = name
    # 기준선과 보장 문구는 제출에 쓴 보장 기준선(risk_threshold 의 채택값)을 그대로 읽는다
    S.data = ROOT / yaml.safe_load(open(ROOT / "configs" / "data.yaml", encoding="utf-8"))["out_dir"]
    th = json.load(open(ROOT / "results/risk_threshold/summary.json", encoding="utf-8"))["채택"]
    S.t_low, S.t_high, S.guarantee = th["합격선"], th["불합격선"], th["보장"]
    S.man = pd.read_csv(S.data / "manifest.csv").set_index("id")
    S.model = YOLO(str(weights_path(name)))
    # 놓침 모형과 고위험 기준값(val 사진 제품 화소 위험의 상위 20%). 주변 결 등급의 경계는 miss_risk 결과에서 읽는다
    S.risk, _, S.high, _ = fit_risk(S.data, S.man, S.t_low, name)
    S.tex_q = json.load(open(ROOT / "results/miss_risk/summary.json", encoding="utf-8"))["주변결_등급기준(고리, val 3분위)"]

    # 보기 사진: 정답이 있는 test 분할만 쓴다 (학습에 쓰지 않은 사진)
    #   samples[묶음] = [{id, machine}], paths[(묶음, id)] = 파일 경로, truth[(묶음, id)] = 정답 박스 [x0, y0, x1, y1] 목록
    test = S.man[S.man["labeled"] & (S.man["split"] == "test")]
    S.samples = {"ng": [], "ok": [], "faint": [], "watch": []}
    S.paths, S.truth = {}, {}
    for i, r in test.iterrows():                         # 같은 id 로 실제 불량(정제본)과 정상(이물 점을 지운 것) 두 묶음을 만든다
        S.samples["ng"].append(dict(id=i, machine=int(r.machine)))
        S.samples["ok"].append(dict(id=i, machine=int(r.machine)))
        S.paths[("ng", i)] = S.data / "clean/images" / f"{i}.png"
        S.paths[("ok", i)] = S.data / "normal/test" / f"{i}.png"
        S.truth[("ng", i)] = M.load_gt([i], S.data / "clean/labels", {i: (r.w, r.h)})[i].tolist()
        S.truth[("ok", i)] = []
    # 옅은 이물 묶음: 이식 시험편(paste_eval.py) 가운데 사진마다 첫 변형(__t00)만 보인다. 평가셋이 없으면 묶음이 빈다
    # 정답 = 원본 사진의 실제 이물 박스 + 옮겨 붙인 자리(중심 ±5px)
    pf = S.data / "paste_test/defects.csv"
    if pf.exists():
        d = pd.read_csv(pf)
        for img, g in d[d["img"].str.endswith("__t00")].groupby("img", sort=False):
            src = g["src"].iloc[0]
            S.samples["faint"].append(dict(id=img, machine=int(g["machine"].iloc[0])))
            S.paths[("faint", img)] = S.data / "paste_test/images" / f"{img}.png"
            S.truth[("faint", img)] = S.truth[("ng", src)] + [[r.cx - 5, r.cy - 5, r.cx + 5, r.cy + 5] for r in g.itertuples()]
    for k in ("ng", "ok"):                               # 첫 호출 준비 시간을 미리 치른다 (호기별 사진 크기마다)
        for mc in (1, 2, 3):
            i = next(s_["id"] for s_ in S.samples[k] if s_["machine"] == mc)
            inspect(np.asarray(Image.open(S.paths[(k, i)]).convert("L")), mc)

    # 자가 점검 설정: 시험편 진하기 = val 테스트피스 사양, 채점 기준선 · 경보 하한 = 상시 점검(monitor.py)과 같은 값
    S.check = None
    try:
        spec = pd.read_csv(ROOT / f"results/testpiece_val_{name}/spec.csv")
        spec = spec[(spec["model"] == "YOLO") & (spec["d"] == CHECK_D)].set_index("machine")["min_c0"]
        # c0 = 호기별 시험편 진하기, thr = 박스 기준선(val F1 최대), lcl = 경보 하한(개), base = 평상시 검출률
        # 경보 하한은 monitor.py 가 요약에 적어 둔 규칙 문장("... 잡은 수 < N ...")에서 숫자만 꺼낸다
        mon = json.load(open(ROOT / "results/monitor/summary.json", encoding="utf-8"))
        thr = json.load(open(ROOT / f"results/yolo_{name}/metrics.json", encoding="utf-8"))["thresholds"]["F1최대"]
        S.check = dict(c0={int(m): float(v) for m, v in spec.items() if pd.notna(v)}, thr=float(thr),
                       lcl=int(re.search(r"< (\d+)", mon["경보_규칙"]).group(1)), base=float(mon["열화전_시험편_검출률"]))
    except Exception as e:                                # 사양표 · 상시 점검 결과가 없으면 자가 점검만 끈다
        print("자가 점검 끔:", e, flush=True)
    # 교대 보고서에 나란히 적을 호기별 기준값 (process_signal.py 의 기준 구간). 결과 파일이 없으면 비워 둔다
    #   per = 이물이 검출된 사진 한 장당 이물 수, off = 띠 밖 이물 비율
    base = json.load(open(ROOT / "results/process_signal/summary.json", encoding="utf-8"))["호기별"] \
        if (ROOT / "results/process_signal/summary.json").exists() else {}
    S.base = {int(m): dict(per=v["기준_이물_사진당"], off=v["기준_띠밖_비율"]) for m, v in base.items()}
    S.started = datetime.now()                          # 교대 보고서의 시작 시각
    if watch:                                           # 감시는 서버가 꺼질 때 같이 끝나도록 데몬 스레드로 돌린다
        S.watch = Path(watch).resolve()
        S.watch.mkdir(parents=True, exist_ok=True)
        threading.Thread(target=watch_loop, daemon=True).start()
    print("준비 완료:", {KINDS[k]: len(v) for k, v in S.samples.items() if v}, "합격선", S.t_low, "불합격선", S.t_high,
          "| 폴더 감시:", S.watch or "끔", flush=True)


def guess_machine(w, h, name=""):
    """호기 번호(1~3)를 짐작한다. 파일 이름이 m1_ · m2_ · m3_ 으로 시작하면 그 번호, 아니면 사진 너비(px)로 가른다.

    너비 500 초과 → 3호기, 330 초과 → 1호기, 그 밖 → 2호기. h 는 쓰지 않는다.
    """
    m = re.match(r"m([123])_", name)
    return int(m.group(1)) if m else 3 if w > 500 else 1 if w > 330 else 2


def load_gray(src):
    """사진을 읽어 회색으로. 색 표시가 있으면 지운다 (표시를 단서로 쓰지 않는다). → (회색, 지웠는지)"""
    rgb = np.asarray(Image.open(src).convert("RGB"))
    gray = np.asarray(Image.fromarray(rgb).convert("L"))
    mask = color_mask(rgb)
    if mask.any():                                      # 복원에 쓰는 난수를 고정해 같은 사진은 늘 같은 결과가 나오게 한다
        gray = restore(gray, mask, np.random.default_rng(0))
    return gray, bool(mask.any())


def predict(gray, conf):
    """회색 영상 한 장을 추론 전용 스레드에서 추론한다 (한 번에 한 장).

    conf 는 이 점수 미만의 박스를 버리는 하한이다. ultralytics 결과 객체 하나를 돌려준다 (boxes.xyxy, boxes.conf).
    """
    return State.gpu.submit(lambda: State.model.predict(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), imgsz=640, conf=conf, max_det=100,
                                                        verbose=False)[0]).result()


def inspect(gray, machine, truth=None, cleaned=False):
    """사진 한 장의 판정과 화면에 보일 설명을 만든다.

    gray : 회색 영상(uint8), machine : 호기 번호, truth : 정답 박스 목록(없으면 None), cleaned : 색 표시를 지웠는지.
    반환 사전의 키
      w, h, machine, top(최고 점수), t_low, t_high, verdict(합격 · 재검사 · 불합격), truth, cleaned
      boxes : 합격선 이상 신호, 점수 높은 순. x0 y0 x1 y1 score verdict band(띠 안인지) edge(가장자리까지 px)
              texture(주변 결 등급) u v(제품 외곽 상자 기준 0~1 자리)
      weak  : 약한 신호(WEAK 이상 합격선 미만) 가운데 점수 높은 다섯 개. x0 y0 x1 y1 score
      image, risk : 사진과 고위험 구역 덧칠의 PNG data URL,  risk_area : 제품 화소 가운데 고위험 구역 비율
      ms : 걸린 시간(ms). infer = 추론, explain = 자리 설명과 위험 지도
    """
    S = State
    t0 = time.perf_counter()
    res = predict(gray, 0.001)                            # 점수가 낮은 박스까지 모두 받아 두고 아래에서 기준선으로 가른다
    t1 = time.perf_counter()
    xyxy, conf = res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()
    top = float(conf.max()) if len(conf) else 0.0
    I = Img.from_array(gray)
    h, w = gray.shape
    ys, xs = np.nonzero(I["pm"])                          # 제품 외곽 상자: 신호 자리를 제품 기준 비율로 적는다 (교대 보고서의 누적 지도)
    px0, px1, py0, py1 = (xs.min(), xs.max() + 1, ys.min(), ys.max() + 1) if len(xs) else (0, w, 0, h)
    boxes = []
    # 합격선 이상 박스마다 박스 중심 화소에서 자리의 성질(띠 · 가장자리 거리 · 주변 결)을 읽는다
    for (x0, y0, x1, y1), s in sorted(zip(xyxy[conf >= S.t_low], conf[conf >= S.t_low]), key=lambda b: -b[1]):
        cx, cy = int(np.clip((x0 + x1) / 2, 0, w - 1)), int(np.clip((y0 + y1) / 2, 0, h - 1))
        t = ring_texture(I, cx, cy)
        boxes.append(dict(x0=float(x0), y0=float(y0), x1=float(x1), y1=float(y1), score=round(float(s), 3),
                          verdict="불합격" if s >= S.t_high else "재검사", band=bool(I["band"][cy, cx]),
                          edge=round(float(I["dist"][cy, cx]), 1),
                          texture="매끈" if t < S.tex_q[0] else "중간" if t < S.tex_q[1] else "거침",
                          u=round(float((cx - px0) / max(1, px1 - px0)), 3), v=round(float((cy - py0) / max(1, py1 - py0)), 3)))
    # 약한 신호: 합격선에는 못 미치지만 WEAK 이상인 박스. 합격 제품의 2순위 확인 표시에 쓴다
    wk = (conf >= WEAK) & (conf < S.t_low)
    weak = [dict(x0=float(x0), y0=float(y0), x1=float(x1), y1=float(y1), score=round(float(s), 3))
            for (x0, y0, x1, y1), s in sorted(zip(xyxy[wk], conf[wk]), key=lambda b: -b[1])[:5]]
    # 놓침 위험 지도: 제품 안 화소마다 기준 이물의 놓침 확률 (제품 밖은 NaN). 고위험 기준값 이상인 화소만 칠한다
    r = risk_map(S.risk, I, machine)
    ok = ~np.isnan(r)
    hi = np.nan_to_num(r) >= S.high
    over = np.zeros((h, w, 4), np.uint8)                  # 투명 바탕의 RGBA 덧칠. 화면이 사진 위에 겹쳐 그린다
    over[hi] = (*RISK_RGB, 120)                           # 불투명도 120/255
    t2 = time.perf_counter()
    return dict(w=w, h=h, machine=machine, top=round(top, 3), t_low=S.t_low, t_high=S.t_high,
                verdict="불합격" if top >= S.t_high else "재검사" if top >= S.t_low else "합격",
                boxes=boxes, weak=weak, truth=truth, cleaned=cleaned, image=png_url(gray), risk=png_url(over),
                risk_area=round(float(hi[ok].mean()) if ok.any() else 0.0, 3),
                ms=dict(infer=round((t1 - t0) * 1000, 1), explain=round((t2 - t1) * 1000, 1)))


def remember(kind, pid, res):
    """이번 실행에서 판정한 제품을 적어 둔다 (집계 · 대기 목록 · 교대 보고서의 근거). 같은 사진은 한 번만.

    kind : 묶음 이름, pid : 사진 id, res : inspect() 의 반환값. 기록의 열쇠 "묶음/사진" 을 돌려준다.
    기록 한 건: seq(판정 순번) kind id machine verdict top n(신호 수) weak(약한 신호 수) infer(ms) boxes(자리 정보만) time.
    """
    S = State
    key = f"{kind}/{pid}"
    with S.lock:
        if key not in S.session:
            S.seq += 1
            S.session[key] = dict(seq=S.seq, kind=kind, id=pid, machine=res["machine"], verdict=res["verdict"], top=res["top"],
                                  n=len(res["boxes"]), weak=len(res["weak"]), infer=res["ms"]["infer"],
                                  boxes=[{k: b[k] for k in ("score", "band", "edge", "texture", "u", "v")} for b in res["boxes"]],
                                  time=datetime.now().strftime("%H:%M:%S"))
    return key


def watch_loop():
    """감시 폴더에 새로 들어온 사진을 판정해 '들어온 사진' 묶음에 올린다. 쓰는 중인 파일은 크기가 멈춘 뒤에 읽는다."""
    S = State
    seen, size = set(), {}                                # seen = 이미 판정한 파일 이름, size = 지난번에 본 파일 크기
    while True:
        for p in sorted(S.watch.iterdir(), key=lambda f: f.stat().st_mtime):
            if p.suffix.lower() not in IMG_EXT or p.name in seen:
                continue
            n = p.stat().st_size
            if size.get(p.name) != n:                     # 크기가 지난번과 다르면 아직 쓰는 중으로 보고 다음 바퀴(1초 뒤)에 다시 본다
                size[p.name] = n
                continue
            try:
                gray, cleaned = load_gray(p)
            except Exception:                             # 읽지 못한 파일은 건너뛴다. seen 에 넣지 않으므로 다음 바퀴에 다시 시도한다
                continue
            seen.add(p.name)
            h, w = gray.shape
            machine = guess_machine(w, h, p.name)
            S.paths[("watch", p.name)] = p
            S.truth[("watch", p.name)] = None             # 들어온 사진은 정답이 없다
            res = inspect(gray, machine, None, cleaned)
            S.samples["watch"].append(dict(id=p.name, machine=machine))
            remember("watch", p.name, res)
        time.sleep(1.0)


def self_check(level=0.0):
    """정상 사진 CHECK_N 장의 어두운 띠 안에 사양 진하기의 가상 시험편을 하나씩 넣고 잡는지 본다.

    시험편 · 채점 기준선 · 경보 하한은 상시 점검(monitor.py)과 같다. level > 0 이면 흐림 · 잡음을 건 모의 열화 조건이다.
    level 은 0~1 이고 1 이 monitor.py 의 가장 심한 열화다.
    반환 사전: time n hits(잡은 수) lcl(경보 하한) base(평상시 검출률) level ok(hits >= lcl) sec(걸린 초)
               pieces = 시험편별 [{machine, c0, hit}]. pieces 를 뺀 나머지는 State.checks 에도 쌓는다.
    """
    S, C = State, State.check
    rng = np.random.default_rng()                         # 시드를 주지 않는다. 점검할 때마다 사진과 자리가 달라진다
    pick = rng.choice(len(S.samples["ok"]), CHECK_N, replace=False)
    pieces, t0 = [], time.perf_counter()
    for k in pick:
        s = S.samples["ok"][int(k)]
        g = np.asarray(Image.open(S.paths[("ok", s["id"])]).convert("L"))
        I = Img.from_array(g)
        ys, xs = np.nonzero(I["band"] & (I["pm"] > 0))    # 제품 안 어두운 띠의 화소 가운데 한 곳을 고른다
        j = rng.integers(len(xs))
        # 화소 안에서 소수점 자리까지 무작위로 둔다. 그 호기의 사양값이 없으면 진하기 0.30 을 쓴다
        cx, cy, c0 = xs[j] + rng.random(), ys[j] + rng.random(), C["c0"].get(s["machine"], 0.30)
        f = g.astype(np.float32)
        insert(f, cx, cy, CHECK_D, c0)
        if level > 0:                                     # 시험편도 같은 장비를 지나가므로 넣은 뒤에 열화를 건다
            f = cv2.GaussianBlur(f, (0, 0), BLUR_MAX * level) + rng.normal(0, NOISE_MAX * level, f.shape).astype(np.float32)
        g2 = np.clip(f.round(), 0, 255).astype(np.uint8)
        r = predict(g2, C["thr"])
        b = r.boxes.xyxy.cpu().numpy()
        # 검출 = 기준선 이상 박스의 중심이 시험편 중심에서 CHECK_NEAR px 안에 하나라도 있음
        hit = bool(len(b) and (np.hypot((b[:, 0] + b[:, 2]) / 2 - cx, (b[:, 1] + b[:, 3]) / 2 - cy) <= CHECK_NEAR).any())
        pieces.append(dict(machine=s["machine"], c0=c0, hit=hit))
    hits = sum(p["hit"] for p in pieces)
    out = dict(time=datetime.now().strftime("%H:%M:%S"), n=CHECK_N, hits=hits, lcl=C["lcl"], base=C["base"], level=level,
               ok=hits >= C["lcl"], pieces=pieces, sec=round(time.perf_counter() - t0, 1))
    S.checks.append({k: v for k, v in out.items() if k != "pieces"})
    return out


def report():
    """이번 실행에서 판정한 묶음의 요약. 화면의 교대 보고서가 이것을 그린다.

    반환 사전의 키
      started, now, model, t_low, t_high, watch(감시 폴더 이름 또는 None)
      n ok hold ng : 검사 수와 판정별 수,  weak : 합격 가운데 약한 신호가 있는 제품 수,  infer : 추론 시간 중앙값(ms)
      machines : 호기별 [{machine, n, hold(사람이 볼 제품 수), signals(신호 수), per(신호 있는 제품당 신호 수),
                 off(띠 밖 신호 비율), base(process_signal 기준값)}]
      signals band(띠 안 비율) edge(가장자리 거리 중앙값 px) texture(결 등급별 수) points([호기, u, v] 목록)
      confirm : {target(사람이 볼 제품 수), done(그중 확정한 수), yes(이물), no(정상)},  checks : 최근 자가 점검 다섯 건
    """
    S = State
    with S.lock:                                          # 집계하는 동안 기록이 바뀌지 않도록 사본을 떠서 쓴다
        rec, conf = list(S.session.values()), dict(S.confirm)
    n = len(rec)
    by = lambda v: sum(r["verdict"] == v for r in rec)    # 판정이 v 인 제품 수
    boxes = [dict(b, machine=r["machine"]) for r in rec for b in r["boxes"]]   # 모든 제품의 신호를 한 줄로 펴고 호기를 붙인다
    machines = []
    for m in sorted({r["machine"] for r in rec}):
        rm = [r for r in rec if r["machine"] == m]
        sig = [r for r in rm if r["n"]]                   # 신호가 하나라도 있는 제품. per 의 분모 (process_signal 의 이물_사진당과 같은 정의)
        bm = [b for b in boxes if b["machine"] == m]
        machines.append(dict(machine=m, n=len(rm), hold=sum(r["verdict"] != "합격" for r in rm), signals=len(bm),
                             per=round(len(bm) / len(sig), 2) if sig else None,
                             off=round(sum(not b["band"] for b in bm) / len(bm), 3) if bm else None,
                             base=S.base.get(m)))
    # 사람이 볼 제품 = 합격이 아니거나, 합격이어도 약한 신호가 있는 것
    target = [f'{r["kind"]}/{r["id"]}' for r in rec if r["verdict"] != "합격" or r["weak"]]
    done = {k: v for k, v in conf.items() if k in S.session}
    tex = {t: sum(b["texture"] == t for b in boxes) for t in ("매끈", "중간", "거침")}
    return dict(started=S.started.strftime("%Y-%m-%d %H:%M"), now=datetime.now().strftime("%Y-%m-%d %H:%M"), model=S.name,
                t_low=S.t_low, t_high=S.t_high, n=n, ok=by("합격"), hold=by("재검사"), ng=by("불합격"),
                weak=sum(r["verdict"] == "합격" and r["weak"] > 0 for r in rec),
                infer=round(float(np.median([r["infer"] for r in rec])), 1) if rec else None,
                machines=machines, signals=len(boxes),
                band=round(sum(b["band"] for b in boxes) / len(boxes), 3) if boxes else None,
                edge=round(float(np.median([b["edge"] for b in boxes])), 1) if boxes else None, texture=tex,
                points=[[b["machine"], b["u"], b["v"]] for b in boxes],
                confirm=dict(target=len(target), done=sum(k in done for k in target),
                             yes=sum(v == "이물" for v in done.values()), no=sum(v == "정상" for v in done.values())),
                checks=S.checks[-5:], watch=str(S.watch.name) if S.watch else None)


class Handler(BaseHTTPRequestHandler):
    """화면(demo/index.html)과 /api/... 요청을 처리한다. 요청마다 스레드가 따로 돈다 (ThreadingHTTPServer)."""

    def log_message(self, *a):
        """요청마다 찍히는 접속 기록을 끈다."""
        pass

    def send(self, body, ctype="application/json; charset=utf-8", code=200):
        """응답을 보낸다. body 가 bytes 가 아니면 JSON 으로 바꾼다. 늘 새 결과를 받도록 캐시를 끈다."""
        if not isinstance(body, bytes):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        """GET 요청.

        /              화면 파일
        /api/info      모델 · 기준선 · 묶음별 사진 목록 (화면이 처음 한 번 읽는다)
        /api/session   판정 기록과 확정 내용 (since 보다 뒤의 것만)
        /api/report    교대 보고서 자료
        /api/selfcheck 자가 점검 실행 (level 0~1)
        /api/thumb     목록에 보일 사진 파일
        /api/inspect   사진 한 장 판정 (kind, id. light 를 주면 그림 없이)
        """
        S = State
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}   # 같은 이름이 여러 번 와도 첫 값만 쓴다
        if u.path == "/":
            return self.send((ROOT / "demo" / "index.html").read_bytes(), "text/html; charset=utf-8")
        if u.path == "/api/info":
            return self.send(dict(model=S.name, t_low=S.t_low, t_high=S.t_high, guarantee=S.guarantee, weak=WEAK, kinds=KINDS,
                                  samples=S.samples, watch=S.watch.name if S.watch else None, check=S.check is not None))
        if u.path == "/api/session":                    # 이번 실행에서 판정한 것 (seq 보다 뒤의 것만) + 확정 내용
            since = int(q.get("since", 0))
            with S.lock:
                rec = [{k: v for k, v in r.items() if k != "boxes"} for r in S.session.values() if r["seq"] > since]
                return self.send(dict(seq=S.seq, records=rec, confirm=S.confirm))
        if u.path == "/api/report":
            return self.send(report())
        if u.path == "/api/selfcheck":
            if S.check is None:
                return self.send({"error": "사양표 또는 상시 점검 결과가 없어 자가 점검을 할 수 없습니다"}, code=400)
            return self.send(self_check(min(1.0, max(0.0, float(q.get("level", 0))))))
        if u.path == "/api/thumb":
            p = S.paths.get((q.get("kind"), unquote(q.get("id", ""))))
            return self.send(p.read_bytes(), "image/png") if p else self.send({"error": "없는 사진"}, code=404)
        if u.path == "/api/inspect":
            key = (q.get("kind"), unquote(q.get("id", "")))
            if key not in S.paths:
                return self.send({"error": "없는 사진"}, code=404)
            machine = next(s["machine"] for s in S.samples[key[0]] if s["id"] == key[1])
            if key[0] == "watch":                   # 들어온 사진은 색 표시가 있을 수 있어 지우고 읽는다. 보기 사진은 이미 정제본이다
                gray, cleaned = load_gray(S.paths[key])
            else:
                gray, cleaned = np.asarray(Image.open(S.paths[key]).convert("L")), False
            res = dict(id=key[1], kind=key[0], **inspect(gray, machine, S.truth[key], cleaned))
            remember(key[0], key[1], res)
            if q.get("light"):                      # 묶음 전체 검사: 그림은 빼고 판정만 보낸다
                res.pop("image"), res.pop("risk")
            return self.send(res)
        self.send({"error": "없는 주소"}, code=404)

    def do_POST(self):
        """POST 요청.

        /api/confirm  작업자 확정을 기록한다 (kind, id, decision)
        /api/upload   본문으로 받은 사진 한 장을 판정한다 (name, machine). 판정 기록에는 남기지 않는다
        """
        S = State
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if u.path == "/api/confirm":                    # 작업자 확정: decision = 이물 | 정상 | 취소
            key, dec = f'{q.get("kind")}/{unquote(q.get("id", ""))}', q.get("decision", "")
            r = S.session.get(key)
            if r is None or dec not in (*DECISIONS, "취소"):
                return self.send({"error": "판정한 적 없는 사진이거나 잘못된 확정입니다"}, code=400)
            with S.lock:
                if dec == "취소":
                    S.confirm.pop(key, None)
                else:
                    S.confirm[key] = dec
                # 확정 기록은 지우지 않고 뒤에 덧붙인다. 취소도 한 줄로 남는다
                LOG.mkdir(parents=True, exist_ok=True)
                f = LOG / "confirm.csv"
                new = not f.exists()                    # 새 파일이면 첫 줄에 열 이름을 쓴다
                with open(f, "a", newline="", encoding="utf-8-sig") as fp:
                    wr = csv.writer(fp)
                    if new:
                        wr.writerow(["시각", "묶음", "사진", "호기", "AI판정", "최고점수", "신호수", "작업자확정"])
                    wr.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), KINDS.get(r["kind"], r["kind"]), r["id"], r["machine"],
                                 r["verdict"], r["top"], r["n"], dec])
            return self.send(dict(ok=True, key=key, decision=None if dec == "취소" else dec))
        if u.path != "/api/upload":
            return self.send({"error": "없는 주소"}, code=404)
        try:
            gray, cleaned = load_gray(io.BytesIO(raw))
        except Exception:
            return self.send({"error": "사진을 읽을 수 없습니다"}, code=400)
        h, w = gray.shape
        name = unquote(q.get("name", "올린 사진"))
        # 호기는 화면에서 고른 값을 쓰고, 고르지 않았으면 파일 이름과 너비로 짐작한다
        machine = int(q["machine"]) if q.get("machine") in ("1", "2", "3") else guess_machine(w, h, name)
        self.send(dict(id=name, kind="upload", **inspect(gray, machine, None, cleaned)))


def main():
    """인자를 읽고 준비(setup)를 마친 뒤 서버를 연다. 이 PC 에서만 접속할 수 있다 (127.0.0.1)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo", default=None, help="모델 이름 (기본: configs/pipeline.yaml 의 최종 모델)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--watch", default=None, help="감시할 폴더. 새로 들어온 사진을 자동으로 판정한다")
    args = ap.parse_args()
    name = args.yolo or yaml.safe_load(open(ROOT / "configs" / "pipeline.yaml", encoding="utf-8"))["final"]["name"]
    setup(name, args.watch)
    print(f"http://127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
