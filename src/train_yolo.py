"""YOLO 학습 → 예측 → 공용 기준(metrics.py) 채점.

실행 예:
  .venv\\Scripts\\python.exe src\\train_yolo.py --name y26s_640
  .venv\\Scripts\\python.exe src\\train_yolo.py --name y26s_640 --skip-train   # 저장된 가중치로 채점만

결과
  runs/<name>/                 ultralytics 학습 기록·가중치 (저장소 제외)
  results/yolo_<name>/         분할별 예측(pred_*.csv), metrics.json, PR 곡선
판정 임계값은 val로 정하고 test는 마지막에 한 번만 채점한다 (베이스라인과 같은 절차).
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import metrics as M

ROOT = Path(__file__).resolve().parents[1]
RECALL_TARGET = 0.95


def weights_path(name):
    """평가에 쓸 가중치. 에폭 고정 학습(final.pt)이 있으면 그것, 없으면 val 기준 best.pt.

    val 은 실제 이물만 있어 거의 만점이라 best.pt 가 이른 에폭에 멈출 수 있다. 모델끼리 비교할 때는
    --patience 0 으로 끝까지 돌리고 마지막 에폭(final.pt)을 쓴다.
    """
    w = ROOT / "runs" / name / "weights"
    return w / "final.pt" if (w / "final.pt").exists() else w / "best.pt"


def predict(model, paths, ids, imgsz, batch=32):
    rows = []
    for k in range(0, len(paths), batch):
        res = model.predict([str(p) for p in paths[k:k + batch]], imgsz=imgsz, conf=0.001,
                            max_det=100, verbose=False)
        for i, r in zip(ids[k:k + batch], res):
            b = r.boxes
            for (x0, y0, x1, y1), s in zip(b.xyxy.cpu().numpy(), b.conf.cpu().numpy()):
                rows.append(dict(id=i, x0=x0, y0=y0, x1=x1, y1=y1, score=float(s)))
    return pd.DataFrame(rows, columns=["id", "x0", "y0", "x1", "y1", "score"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "data.yaml"))
    ap.add_argument("--variant", default="clean")
    ap.add_argument("--data", default=None, help="학습용 데이터셋 yaml (기본: data/<variant>.yaml). 채점은 항상 <variant> 영상으로")
    ap.add_argument("--model", default="yolo26s.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patience", type=int, default=50, help="조기 종료 (0 = 끄고 epochs 만큼 다 돈다. 모델끼리 비교할 때)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    from ultralytics import YOLO

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    data = ROOT / cfg["out_dir"]
    out = ROOT / "results" / f"yolo_{args.name}"
    out.mkdir(parents=True, exist_ok=True)
    run_dir = ROOT / "runs" / args.name

    if not args.skip_train:
        model = YOLO(args.model)
        model.train(
            data=str(ROOT / args.data) if args.data else str(data / f"{args.variant}.yaml"), imgsz=args.imgsz, epochs=args.epochs,
            batch=args.batch, seed=args.seed, deterministic=True, workers=2,
            project=str(ROOT / "runs"), name=args.name, exist_ok=True,
            # 흑백 X-ray라 색 증강은 끈다. 상하·좌우 뒤집기는 물리적으로 자연스럽다.
            hsv_h=0.0, hsv_s=0.0, hsv_v=0.3, flipud=0.5, fliplr=0.5,
            patience=args.patience if args.patience > 0 else args.epochs + 1, plots=True, verbose=False,
        )
    if not args.skip_train and args.patience == 0:
        import shutil
        shutil.copy(run_dir / "weights" / "last.pt", run_dir / "weights" / "final.pt")
    model = YOLO(str(weights_path(args.name)))

    man = pd.read_csv(data / "manifest.csv")
    man = man[man["labeled"]].set_index("id")
    sizes = {i: (r.w, r.h) for i, r in man.iterrows()}
    split = {s: man.index[man["split"] == s].tolist() for s in ["train", "val", "test"]}
    img_dir, label_dir = data / args.variant / "images", data / args.variant / "labels"
    gt = {s: M.load_gt(ids, label_dir, sizes) for s, ids in split.items()}
    preds = {s: predict(model, [img_dir / f"{i}.png" for i in ids], ids, args.imgsz)
             for s, ids in split.items()}

    n_val = sum(len(g) for g in gt["val"].values())
    pm_val, _ = M.match(preds["val"], gt["val"])
    thr = {"F1최대": M.best_f1_threshold(pm_val, n_val)}
    thr[f"재현율{int(RECALL_TARGET * 100)}"] = min(thr["F1최대"], M.recall_threshold(pm_val, n_val, RECALL_TARGET))

    res = {"params": vars(args), "thresholds": thr, "metrics": {}}
    for s in ["train", "val", "test"]:
        for name, t in thr.items():
            for rule in ["center", "iou50"]:
                res["metrics"][f"{s}/{name}/{rule}"] = M.evaluate(preds[s], gt[s], t, rule)
    for s, p in preds.items():
        pm, _ = M.match(p, gt[s])
        pm.to_csv(out / f"pred_{s}.csv", index=False, encoding="utf-8-sig")
    json.dump(res, open(out / "metrics.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    plt.rcParams["font.family"] = "Malgun Gothic"
    fig, ax = plt.subplots(figsize=(5, 4.2), dpi=150)
    n_test = sum(len(g) for g in gt["test"].values())
    for rule, c in [("center", "#1f5fa8"), ("iou50", "#9aa5b1")]:
        pm, _ = M.match(preds["test"], gt["test"], rule)
        _, prec, rec, apv = M.pr_curve(pm, n_test)
        ax.plot(rec, prec, color=c, lw=1.8, label=f"{rule}  AP {apv:.3f}")
    ax.set(xlabel="재현율", ylabel="정밀도", xlim=(0, 1), ylim=(0, 1.02), title=f"YOLO {args.name} · test")
    ax.grid(alpha=.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "pr_test.png")

    for k, v in res["metrics"].items():
        if k.startswith(("val", "test")):
            print(k, {x: v[x] for x in ["AP", "thr", "TP", "FP", "FN", "precision", "recall", "F1"]})


if __name__ == "__main__":
    main()
