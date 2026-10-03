"""추론 속도와 형식 변환 검증: 같은 최종 모델을 여러 실행 형식으로 돌려 한 장당 처리 시간을 재고, 판정이 바뀌지 않는지 확인한다.

형식
  PyTorch FP32 (GPU)  : 평가 · 제출에 쓴 원래 방식 (기준)
  PyTorch FP16 (GPU)  : 반정밀도
  ONNX (GPU / CPU)    : onnxruntime. 입력 640×640 고정
  TensorRT FP16 (GPU) : NVIDIA 전용 최적화 엔진. 입력 640×640 고정
  PyTorch (CPU)       : GPU 없는 현장 PC 기준
사진: 시험 판정 세트(data/judge_test.csv: 가짜 정상 73 · 실제 불량 73 · 합성 불량 292 = 438장), 한 장씩(batch 1) 처리.
시간: ultralytics 가 재는 전처리 · 추론 · 후처리(ms)와, 파일 읽기를 포함한 한 장 전체 시간. 앞 20장은 예열로 버린다.
판정 비교: 사진 최고 점수로 보장 기준선(risk_threshold 채택값) 3단 판정을 내려 기준 형식과 비교하고,
           실제 이물 139개를 박스 기준선에서 몇 개 찾는지도 함께 잰다.

추가 패키지가 필요하다 (재현 환경을 바꾸지 않도록 requirements.txt 에는 넣지 않음):
  pip install onnx onnxslim onnxruntime-gpu==1.22.0 tensorrt-cu12
  주의: onnxruntime-gpu 1.23 이상의 PyPI 기본판은 CUDA 13 용이라 CUDA 12.8 PyTorch 환경에서는 GPU 를 못 잡고
  조용히 CPU 로 돈다. 또 ultralytics 가 ONNX 변환 중 CPU 판 onnxruntime 을 자동 설치해 GPU 판을 덮을 수 있어,
  변환 파일이 이미 있으면 다시 변환하지 않고, 실제로 쓴 실행 장치(provider)를 결과에 적는다.
실행: python src/speed.py --yolo ratio3_e100
결과: results/speed/ (summary.json, speed.csv, models/ 에 변환한 모델)
"""
import argparse
import json
import platform
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

import metrics as M
from judge import tiers
from train_yolo import weights_path

ROOT = Path(__file__).resolve().parents[1]
WARMUP = 20


def run(model, paths, device, half):
    """한 장씩 추론. 반환: 사진별 (최고 점수, 박스들), 시간 표."""
    tops, boxes, rows = [], [], []
    for k, p in enumerate([str(x) for x in paths]):
        t0 = time.perf_counter()
        r = model.predict(p, imgsz=640, conf=0.001, max_det=100, device=device, half=half, verbose=False)[0]
        c = r.boxes.conf.cpu().numpy()
        xyxy = r.boxes.xyxy.cpu().numpy()
        wall = (time.perf_counter() - t0) * 1000
        tops.append(float(c.max()) if len(c) else 0.0)
        boxes.append(np.c_[xyxy, c] if len(c) else np.zeros((0, 5)))
        if k >= WARMUP:
            rows.append(dict(전처리=r.speed["preprocess"], 추론=r.speed["inference"], 후처리=r.speed["postprocess"], 전체=wall))
    return np.array(tops), boxes, pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo", default="ratio3_e100")
    ap.add_argument("--skip", nargs="*", default=[], help="건너뛸 형식 이름")
    args = ap.parse_args()
    import torch
    from ultralytics import YOLO
    try:
        import onnxruntime
        onnxruntime.preload_dlls()          # PyTorch 에 든 CUDA · cuDNN DLL 을 onnxruntime 이 찾게 한다
    except (ImportError, AttributeError):
        pass
    out = ROOT / "results" / "speed"
    mdir = out / "models"
    mdir.mkdir(parents=True, exist_ok=True)
    src = mdir / f"{args.yolo}.pt"
    shutil.copy(weights_path(args.yolo), src)          # 변환 파일이 원래 가중치 옆에 생기지 않도록 복사본에서 변환

    df = pd.read_csv(ROOT / "data/judge_test.csv")
    paths = df["path"].tolist()
    th = json.load(open(ROOT / "results/risk_threshold/summary.json", encoding="utf-8"))["채택"]
    thr_box = json.load(open(ROOT / f"results/yolo_{args.yolo}/metrics.json", encoding="utf-8"))["thresholds"]["F1최대"]
    man = pd.read_csv(ROOT / "data/manifest.csv").set_index("id")
    real = df[df.kind == "real_ng"]
    gt = M.load_gt(real["img"].tolist(), ROOT / "data/clean/labels", {i: (man.w[i], man.h[i]) for i in real["img"]})

    def export(fmt, **kw):
        f = src.with_suffix({"onnx": ".onnx", "engine": ".engine"}[fmt])
        if not f.exists():
            f = YOLO(str(src)).export(format=fmt, imgsz=640, verbose=False, **kw)
        return str(f)

    forms = [("PyTorch FP32 (GPU)", lambda: str(src), 0, False),
             ("PyTorch FP16 (GPU)", lambda: str(src), 0, True),
             ("ONNX (GPU)", lambda: export("onnx", simplify=True), 0, False),
             ("TensorRT FP16 (GPU)", lambda: export("engine", half=True, device=0), 0, True),
             ("ONNX (CPU)", lambda: str(mdir / f"{args.yolo}.onnx"), "cpu", False),
             ("PyTorch (CPU)", lambda: str(src), "cpu", False)]
    res, ref = {}, None
    for name, get, device, half in forms:
        if name in args.skip:
            continue
        try:
            t0 = time.perf_counter()
            f = get()
            conv = time.perf_counter() - t0
            model = YOLO(f, task="detect")
            tops, bx, tm = run(model, paths, device, half)
            prov = None
            if f.endswith(".onnx"):          # 실제로 쓴 실행 장치 (GPU 를 못 잡으면 CPU 로 조용히 넘어간다)
                sess = getattr(getattr(model.predictor, "model", None), "session", None)
                prov = sess.get_providers()[0] if sess is not None else None
        except Exception as e:           # 설치 · 장치 문제로 안 되는 형식은 이유를 남기고 넘어간다
            res[name] = {"오류": f"{type(e).__name__}: {str(e)[:200]}"}
            print(name, "실패", e)
            continue
        verdict = tiers(tops, th["합격선"], th["불합격선"])
        pr = pd.DataFrame([(i, *b) for i, bb in zip(df["img"], bx) for b in bb if i in gt],
                          columns=["id", "x0", "y0", "x1", "y1", "score"])
        ev = M.evaluate(pr, gt, thr_box, "center")
        r = {"파일": Path(f).name, "파일크기_MB": round(Path(f).stat().st_size / 1e6, 1) if Path(f).is_file() else None,
             "변환_초": round(conv, 1), "실행장치": prov,
             "한장_전체_ms": {"중앙": round(float(tm["전체"].median()), 1), "95%": round(float(tm["전체"].quantile(0.95)), 1)},
             "추론만_ms_중앙": round(float(tm["추론"].median()), 2),
             "전처리_ms_중앙": round(float(tm["전처리"].median()), 2), "후처리_ms_중앙": round(float(tm["후처리"].median()), 2),
             "초당_장수": round(1000 / float(tm["전체"].median()), 1),
             "실제이물_찾음": f"{ev['TP']}/{ev['n_gt']}", "실제사진_헛경보": ev["FP"]}
        if ref is None:
            ref = (tops, verdict)
            r["판정분포"] = pd.Series(verdict).value_counts().to_dict()
        else:
            r["기준과_판정같음"] = f"{int((verdict == ref[1]).sum())}/{len(verdict)}"
            r["기준과_최고점수_최대차이"] = round(float(np.abs(tops - ref[0]).max()), 4)
            diff = np.where(verdict != ref[1])[0]
            r["판정다른사진"] = [dict(사진=df["img"].iloc[i], 종류=df["kind"].iloc[i], 기준=ref[1][i], 이형식=verdict[i],
                                   기준점수=round(float(ref[0][i]), 4), 이형식점수=round(float(tops[i]), 4)) for i in diff[:10]]
        res[name] = r
        print(name, json.dumps({k: v for k, v in r.items() if k != "판정다른사진"}, ensure_ascii=False))
        del model
        torch.cuda.empty_cache()

    env = {"GPU": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "CPU": platform.processor(),
           "torch": torch.__version__, "사진수": len(paths), "예열": WARMUP, "보장기준선": [th["합격선"], th["불합격선"]],
           "박스기준선": round(thr_box, 4)}
    try:
        import onnxruntime, tensorrt
        env.update(onnxruntime=onnxruntime.__version__, tensorrt=tensorrt.__version__)
    except ImportError:
        pass
    json.dump({"환경": env, "형식": res}, open(out / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    pd.DataFrame([dict(형식=k, **{kk: (json.dumps(vv, ensure_ascii=False) if isinstance(vv, (dict, list)) else vv)
                                 for kk, vv in v.items()}) for k, v in res.items()]) \
        .to_csv(out / "speed.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
