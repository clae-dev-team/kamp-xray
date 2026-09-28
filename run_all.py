"""전처리부터 학습·추론·결과 생성까지 한 번에 실행한다.

  python run_all.py                    # 기본: 최종 모델까지 학습하고 모든 평가·제출 파일 생성 (GPU 약 1시간)
  python run_all.py --skip-train       # 학습은 건너뛰고 저장된 가중치로 평가·제출 파일만 다시 생성 (약 15분)
  python run_all.py --all-experiments  # 보고서의 비교 모델까지 모두 다시 학습 (약 3~4시간)
  python run_all.py --from judge       # 특정 단계부터 이어서

단계와 결과 위치는 README.md 참고. 단계별 소요 시간은 results/run_all.json 에 남는다.
원본 데이터 경로는 configs/data.yaml, 모델 설정은 configs/pipeline.yaml 에서 바꾼다.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
PY = sys.executable
STAGES = ["prepare", "baseline", "defect_stats", "synth", "augment", "train", "normal_set", "judge",
          "predict", "synth_eval", "location", "shortcut"]


def sh(*args, log=None):
    cmd = [PY, *map(str, args)]
    print("  $", " ".join(cmd[1:]), flush=True)
    if log:
        with open(ROOT / "results" / "logs" / log, "w", encoding="utf-8") as f:
            subprocess.run(cmd, cwd=ROOT, check=True, stdout=f, stderr=subprocess.STDOUT,
                           env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
    else:
        subprocess.run(cmd, cwd=ROOT, check=True, env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})


def train(m, skip=False):
    """skip=True 면 학습 없이 저장된 가중치로 채점만 (기록되는 설정값은 학습 때와 같게 넘긴다)."""
    a = ["src/train_yolo.py", "--name", m["name"], "--epochs", m["epochs"], "--patience", m["patience"]]
    if m.get("data"):
        a += ["--data", m["data"]]
    if m.get("variant"):
        a += ["--variant", m["variant"]]
    sh(*a, *(["--skip-train"] if skip else []), log=f"{'score' if skip else 'train'}_{m['name']}.log")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-train", action="store_true", help="학습 없이 runs/ 의 가중치로 평가")
    ap.add_argument("--all-experiments", action="store_true", help="비교 모델까지 다시 학습·평가")
    ap.add_argument("--from", dest="start", default="prepare", choices=STAGES)
    args = ap.parse_args()
    pipe = yaml.safe_load(open(ROOT / "configs" / "pipeline.yaml", encoding="utf-8"))
    final, exps = pipe["final"], pipe["experiments"] if args.all_experiments else []
    (ROOT / "results" / "logs").mkdir(parents=True, exist_ok=True)
    yolo_models = [final["name"]] + [e["name"] for e in exps if e.get("variant") != "raw"]
    timing = {}

    def stage(name, fn):
        if STAGES.index(name) < STAGES.index(args.start):
            return
        print(f"[{name}]", flush=True)
        t = time.time()
        fn()
        timing[name] = round(time.time() - t, 1)

    stage("prepare", lambda: sh("src/prepare.py", log="prepare.log"))
    stage("baseline", lambda: (sh("src/baseline.py", log="baseline_clean.log"),
                               sh("src/baseline.py", "--variant", "raw", log="baseline_raw.log")))
    stage("defect_stats", lambda: sh("src/defect_stats.py", log="defect_stats.log"))
    stage("synth", lambda: sh("src/synth.py", log="synth.log"))

    def augment():
        sh("src/augment.py", "--variants", final.get("variants", 3), log="augment.log")
        for e in exps:
            if e.get("data", "").startswith("data/aug"):
                a = ["src/augment.py", "--variants", e.get("variants", 3)] + (["--noise"] if e.get("noise") else [])
                sh(*a, log=f"augment_{e['name']}.log")
    stage("augment", augment)

    def train_all():
        for m in [final] + exps:
            train(m, skip=args.skip_train)
    stage("train", train_all)
    stage("normal_set", lambda: sh("src/normal_set.py", log="normal_set.log"))
    stage("judge", lambda: sh("src/judge.py", "--yolo", *yolo_models, log="judge.log"))
    stage("predict", lambda: sh("src/predict.py", "--name", final["name"], log="predict.log"))
    stage("synth_eval", lambda: [sh("src/synth_eval.py", "--yolo", m, log=f"synth_eval_{m}.log") for m in yolo_models])
    stage("location", lambda: [sh("src/location_test.py", "--yolo", m, log=f"location_{m}.log") for m in yolo_models])
    if any(e.get("variant") == "raw" for e in exps):
        stage("shortcut", lambda: sh("src/shortcut_test.py", "--clean", final["name"], "--raw", "y26s_640_raw",
                                     log="shortcut.log"))

    json.dump({"final": final["name"], "models": yolo_models, "skip_train": args.skip_train, "seconds": timing},
              open(ROOT / "results" / "run_all.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("완료:", json.dumps(timing, ensure_ascii=False))


if __name__ == "__main__":
    main()
