"""검사 화면 시연: 사진 한 장을 넣으면 3단 판정 · 박스 · 놓치기 쉬운 구역을 보여 주는 작은 웹 화면.

현장에서 이 모델이 어떻게 쓰일지를 보이기 위한 것이다. 판정에 쓰는 모델 · 기준선 · 위험 지도는 모두 제출 결과와 같다.
  - 모델      : configs/pipeline.yaml 의 최종 모델 가중치
  - 기준선    : results/risk_threshold/summary.json 의 보장 기준선 (합격선 · 불합격선)
  - 위험 지도 : miss_risk.py 의 놓침 모형 (val 시험편으로 맞춘 것)
추가 패키지 없이 표준 라이브러리 http.server 로 돈다. 화면은 demo/index.html 한 파일.

보기 사진 세 묶음 (모두 학습에 쓰지 않은 test 분할)
  실제 불량 : 정제본 그대로 / 정상 : 이물 점을 지운 가짜 정상 / 옅은 이물 : 진짜 점을 옅게 옮겨 붙인 이식 시험편
직접 올린 사진에 색 표시가 있으면 prepare.py 와 같은 방법으로 지우고 판정한다 (표시를 단서로 쓰지 않는다).

실행: .venv\\Scripts\\python.exe src\\demo.py  →  http://127.0.0.1:8765
"""
import argparse
import base64
import io
import json
import threading
import time
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
from train_yolo import weights_path

ROOT = Path(__file__).resolve().parents[1]
KINDS = {"ng": "실제 불량", "ok": "정상", "faint": "옅은 이물"}
RISK_RGB = (232, 148, 58)
WEAK = 0.3                    # 2순위(약한 신호) 하한. operating_point.py · zone_rules.py 와 같은 값


class State:
    lock = threading.Lock()


def png_url(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def setup(name):
    from ultralytics import YOLO

    S = State
    S.name = name
    S.data = ROOT / yaml.safe_load(open(ROOT / "configs" / "data.yaml", encoding="utf-8"))["out_dir"]
    th = json.load(open(ROOT / "results/risk_threshold/summary.json", encoding="utf-8"))["채택"]
    S.t_low, S.t_high, S.guarantee = th["합격선"], th["불합격선"], th["보장"]
    S.man = pd.read_csv(S.data / "manifest.csv").set_index("id")
    S.model = YOLO(str(weights_path(name)))
    S.risk, _, S.high, _ = fit_risk(S.data, S.man, S.t_low, name)
    S.tex_q = json.load(open(ROOT / "results/miss_risk/summary.json", encoding="utf-8"))["주변결_등급기준(고리, val 3분위)"]

    test = S.man[S.man["labeled"] & (S.man["split"] == "test")]
    S.samples = {"ng": [], "ok": [], "faint": []}
    S.paths, S.truth = {}, {}
    for i, r in test.iterrows():
        S.samples["ng"].append(dict(id=i, machine=int(r.machine)))
        S.samples["ok"].append(dict(id=i, machine=int(r.machine)))
        S.paths[("ng", i)] = S.data / "clean/images" / f"{i}.png"
        S.paths[("ok", i)] = S.data / "normal/test" / f"{i}.png"
        S.truth[("ng", i)] = M.load_gt([i], S.data / "clean/labels", {i: (r.w, r.h)})[i].tolist()
        S.truth[("ok", i)] = []
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
    print("준비 완료:", {KINDS[k]: len(v) for k, v in S.samples.items()}, "합격선", S.t_low, "불합격선", S.t_high, flush=True)


def guess_machine(w, h):
    return 3 if w > 500 else 1 if w > 330 else 2


def inspect(gray, machine, truth=None, cleaned=False):
    S = State
    t0 = time.perf_counter()
    with S.lock:
        res = S.model.predict(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), imgsz=640, conf=0.001, max_det=100, verbose=False)[0]
    t1 = time.perf_counter()
    xyxy, conf = res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()
    top = float(conf.max()) if len(conf) else 0.0
    I = Img.from_array(gray)
    h, w = gray.shape
    boxes = []
    for (x0, y0, x1, y1), s in sorted(zip(xyxy[conf >= S.t_low], conf[conf >= S.t_low]), key=lambda b: -b[1]):
        cx, cy = int(np.clip((x0 + x1) / 2, 0, w - 1)), int(np.clip((y0 + y1) / 2, 0, h - 1))
        t = ring_texture(I, cx, cy)
        boxes.append(dict(x0=float(x0), y0=float(y0), x1=float(x1), y1=float(y1), score=round(float(s), 3),
                          verdict="불합격" if s >= S.t_high else "재검사", band=bool(I["band"][cy, cx]),
                          edge=round(float(I["dist"][cy, cx]), 1),
                          texture="매끈" if t < S.tex_q[0] else "중간" if t < S.tex_q[1] else "거침"))
    wk = (conf >= WEAK) & (conf < S.t_low)
    weak = [dict(x0=float(x0), y0=float(y0), x1=float(x1), y1=float(y1), score=round(float(s), 3))
            for (x0, y0, x1, y1), s in sorted(zip(xyxy[wk], conf[wk]), key=lambda b: -b[1])[:5]]
    r = risk_map(S.risk, I, machine)
    ok = ~np.isnan(r)
    hi = np.nan_to_num(r) >= S.high
    over = np.zeros((h, w, 4), np.uint8)
    over[hi] = (*RISK_RGB, 120)
    t2 = time.perf_counter()
    return dict(w=w, h=h, machine=machine, top=round(top, 3), t_low=S.t_low, t_high=S.t_high,
                verdict="불합격" if top >= S.t_high else "재검사" if top >= S.t_low else "합격",
                boxes=boxes, weak=weak, truth=truth, cleaned=cleaned, image=png_url(gray), risk=png_url(over),
                risk_area=round(float(hi[ok].mean()) if ok.any() else 0.0, 3),
                ms=dict(infer=round((t1 - t0) * 1000, 1), explain=round((t2 - t1) * 1000, 1)))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, body, ctype="application/json; charset=utf-8", code=200):
        if not isinstance(body, bytes):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        S = State
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path == "/":
            return self.send((ROOT / "demo" / "index.html").read_bytes(), "text/html; charset=utf-8")
        if u.path == "/api/info":
            return self.send(dict(model=S.name, t_low=S.t_low, t_high=S.t_high, guarantee=S.guarantee, weak=WEAK, kinds=KINDS,
                                  samples=S.samples))
        if u.path == "/api/thumb":
            p = S.paths.get((q.get("kind"), unquote(q.get("id", ""))))
            return self.send(p.read_bytes(), "image/png") if p else self.send({"error": "없는 사진"}, code=404)
        if u.path == "/api/inspect":
            key = (q.get("kind"), unquote(q.get("id", "")))
            if key not in S.paths:
                return self.send({"error": "없는 사진"}, code=404)
            gray = np.asarray(Image.open(S.paths[key]).convert("L"))
            machine = next(s["machine"] for s in S.samples[key[0]] if s["id"] == key[1])
            res = dict(id=key[1], kind=key[0], **inspect(gray, machine, S.truth[key]))
            if q.get("light"):                      # 묶음 전체 검사: 그림은 빼고 판정만 보낸다
                res.pop("image"), res.pop("risk")
            return self.send(res)
        self.send({"error": "없는 주소"}, code=404)

    def do_POST(self):
        u = urlparse(self.path)
        if u.path != "/api/upload":
            return self.send({"error": "없는 주소"}, code=404)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            rgb = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
        except Exception:
            return self.send({"error": "사진을 읽을 수 없습니다"}, code=400)
        gray = np.asarray(Image.fromarray(rgb).convert("L"))
        mask = color_mask(rgb)
        if mask.any():                                    # 색 표시는 지우고 판정한다
            gray = restore(gray, mask, np.random.default_rng(0))
        h, w = gray.shape
        machine = int(q["machine"]) if q.get("machine") in ("1", "2", "3") else guess_machine(w, h)
        self.send(dict(id=unquote(q.get("name", "올린 사진")), kind="upload", **inspect(gray, machine, None, bool(mask.any()))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo", default=None, help="모델 이름 (기본: configs/pipeline.yaml 의 최종 모델)")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    name = args.yolo or yaml.safe_load(open(ROOT / "configs" / "pipeline.yaml", encoding="utf-8"))["final"]["name"]
    setup(name)
    print(f"http://127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
