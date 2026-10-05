"""교차 호기 실험 (leave-one-machine-out): 두 호기 사진으로만 학습하고, 한 번도 보지 못한 나머지 호기로 시험한다.

분할(prepare.py)은 호기·날짜 묶음이라 세 호기가 train·val·test 에 모두 들어 있다. 그래서 지금까지의 수치는
'같은 장비에서 다른 날 찍은 사진'에 대한 것이고, 새 장비에 그대로 옮겼을 때는 말해 주지 않는다.

  학습   : 남은 두 호기의 train 사진 (+ --variants 가 0 이 아니면 그 사진으로 만든 합성 증강 data/aug)
  기준선 : 남은 두 호기의 val 사진에서 F1 최대 (빼놓은 호기는 기준선에도 쓰지 않는다)
  시험   : 빼놓은 호기의 라벨 사진 전부 (train·val·test 구분 없이, 학습에 쓰인 적이 없으므로)
  비교   : 세 호기를 모두 학습한 모델(--ref)을 빼놓은 호기의 test 분할에서 채점한 값, 같은 사진에서의 교차 호기 값
  합성   : 빼놓은 호기 test 사진으로 만든 합성 저대비 이물(data/synth) 검출률

학습 설정은 최종 모델과 같다 (YOLO26s, 640, 100에폭 고정, 시드 0).

실행: .venv\\Scripts\\python.exe src\\cross_machine.py --holdout 3              # 합성 3배
      .venv\\Scripts\\python.exe src\\cross_machine.py --holdout 3 --variants 0 # 합성 없이
      .venv\\Scripts\\python.exe src\\cross_machine.py --summary                # 표로 모으기
결과: results/cross_machine/<이름>/metrics.json, results/cross_machine/summary.json
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import metrics as M
from synth_eval import score_defects, yolo_preds
from train_yolo import predict, weights_path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "cross_machine"


def run_name(holdout, variants):
    return f"lomo{holdout}_ratio{variants}"


def write_lists(data, man, holdout, variants):
    """남은 두 호기의 학습·검증 목록과 데이터셋 yaml."""
    d = data / "lomo"
    d.mkdir(parents=True, exist_ok=True)
    rest = man[man["machine"] != holdout]
    img = lambda i: str((data / "clean/images" / f"{i}.png").resolve())
    tr = [img(i) for i in rest.index[rest["split"] == "train"]]
    for i in rest.index[rest["split"] == "train"]:
        tr += [str((data / "aug/images" / f"{i}__a{v}.png").resolve()) for v in range(variants)]
    va = [img(i) for i in rest.index[rest["split"] == "val"]]
    name = run_name(holdout, variants)
    (d / f"{name}_train.txt").write_text("\n".join(tr) + "\n", encoding="utf-8")
    (d / f"{name}_val.txt").write_text("\n".join(va) + "\n", encoding="utf-8")
    ds = {"path": str((data / "clean").resolve()), "train": str((d / f"{name}_train.txt").resolve()),
          "val": str((d / f"{name}_val.txt").resolve()), "names": {0: "Defect"}}
    yaml.safe_dump(ds, open(d / f"{name}.yaml", "w", encoding="utf-8"), allow_unicode=True)
    return d / f"{name}.yaml", len(tr), len(va)


def score_set(model, ids, data, sizes, imgsz):
    gt = M.load_gt(ids, data / "clean/labels", sizes)
    pred = predict(model, [data / "clean/images" / f"{i}.png" for i in ids], ids, imgsz)
    return pred, gt


def evaluate(pred, gt, thr):
    return {rule: M.evaluate(pred, gt, thr, rule) for rule in ["center", "iou50"]}


def synth_rate(name, data, holdout, thr, man):
    """빼놓은 호기 test 사진으로 만든 합성 저대비 이물 검출률 (채점 방식은 synth_eval.py 와 같다)."""
    imgs = pd.read_csv(data / "synth/images.csv")
    imgs = imgs[imgs["machine"] == holdout]
    defects = pd.read_csv(data / "synth/defects.csv")
    defects = defects[defects["machine"] == holdout].reset_index(drop=True)
    half = json.load(open(data / "synth/config.json", encoding="utf-8"))["box"] / 2
    real = {}
    for s in imgs["src"].unique():
        b = np.loadtxt(data / "clean/labels" / f"{s}.txt", ndmin=2)
        w, h = man.loc[s, "w"], man.loc[s, "h"]
        real[s] = np.stack([(b[:, 1] - b[:, 3] / 2) * w, (b[:, 2] - b[:, 4] / 2) * h,
                            (b[:, 1] + b[:, 3] / 2) * w, (b[:, 2] + b[:, 4] / 2) * h], 1)
    pred = yolo_preds(name, [data / "synth/images" / f"{i}.png" for i in imgs["img"]], imgs["img"].tolist())
    best, fp = score_defects(pred, defects, real, thr, half)
    return {"이물수": len(defects), "검출률": round(float((best >= thr).mean()), 4),
            "오검출_영상당": round(float(np.mean(list(fp.values()))), 4)}


def run(args):
    from ultralytics import YOLO

    cfg = yaml.safe_load(open(ROOT / "configs" / "data.yaml", encoding="utf-8"))
    data = ROOT / cfg["out_dir"]
    man = pd.read_csv(data / "manifest.csv")
    man = man[man["labeled"]].set_index("id")
    sizes = {i: (r.w, r.h) for i, r in man.iterrows()}
    name = run_name(args.holdout, args.variants)
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    ds, n_tr, n_va = write_lists(data, man, args.holdout, args.variants)
    run_dir = ROOT / "runs" / name

    if not args.skip_train:
        YOLO(args.model).train(
            data=str(ds), imgsz=args.imgsz, epochs=args.epochs, batch=16, seed=args.seed, deterministic=True, workers=2,
            project=str(ROOT / "runs"), name=name, exist_ok=True,
            hsv_h=0.0, hsv_s=0.0, hsv_v=0.3, flipud=0.5, fliplr=0.5,          # train_yolo.py 와 같은 설정
            patience=args.epochs + 1, plots=False, verbose=False,
        )
        shutil.copy(run_dir / "weights" / "last.pt", run_dir / "weights" / "final.pt")
    model = YOLO(str(weights_path(name)))

    rest_val = man.index[(man["machine"] != args.holdout) & (man["split"] == "val")].tolist()
    held_all = man.index[man["machine"] == args.holdout].tolist()
    held_test = man.index[(man["machine"] == args.holdout) & (man["split"] == "test")].tolist()

    pv, gv = score_set(model, rest_val, data, sizes, args.imgsz)
    n_val = sum(len(g) for g in gv.values())
    thr = M.best_f1_threshold(M.match(pv, gv)[0], n_val)
    ph, gh = score_set(model, held_all, data, sizes, args.imgsz)
    pm, _ = M.match(ph, gh)
    pm.to_csv(out / "pred_holdout.csv", index=False, encoding="utf-8-sig")
    tp = pm[pm["tp"] == 1]

    res = {"이름": name, "빼놓은_호기": args.holdout, "합성_배수": args.variants, "학습_사진수": n_tr, "검증_사진수": n_va,
           "기준선(남은 호기 val F1최대)": thr,
           "남은_호기_val": evaluate(pv, gv, thr),
           "빼놓은_호기_전체": evaluate(ph, gh, thr),
           "빼놓은_호기_test분할": evaluate(ph[ph["id"].isin(held_test)], {i: gh[i] for i in held_test}, thr),
           "빼놓은_호기_실제이물_최저점수": round(float(tp["score"].min()), 4) if len(tp) else None,
           "빼놓은_호기_합성": synth_rate(name, data, args.holdout, thr, man)}

    # 비교: 세 호기를 모두 학습한 모델을 같은 test 분할 사진에서, 그 모델 자신의 기준선으로
    ref = args.ref or ("ratio3_e100" if args.variants else "ratio0_e100")
    rf = ROOT / "results" / f"yolo_{ref}" / "metrics.json"
    if rf.exists() and weights_path(ref).exists():
        rthr = json.load(open(rf, encoding="utf-8"))["thresholds"]["F1최대"]
        pr, gr = score_set(YOLO(str(weights_path(ref))), held_test, data, sizes, args.imgsz)
        res["비교_전체호기학습"] = {"모델": ref, "기준선": rthr, "빼놓은_호기_test분할": evaluate(pr, gr, rthr),
                             "빼놓은_호기_합성": synth_rate(ref, data, args.holdout, rthr, man)}
    json.dump(res, open(out / "metrics.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    c = res["빼놓은_호기_전체"]["center"]
    print(name, "기준선", round(thr, 4), {k: c[k] for k in ["n_images", "n_gt", "AP", "TP", "FP", "FN", "precision", "recall", "F1"]},
          "합성", res["빼놓은_호기_합성"])


def summary():
    rows = []
    for f in sorted(OUT.glob("lomo*/metrics.json")):
        r = json.load(open(f, encoding="utf-8"))
        a, t = r["빼놓은_호기_전체"], r["빼놓은_호기_test분할"]
        row = {"빼놓은_호기": r["빼놓은_호기"], "합성_배수": r["합성_배수"], "학습_사진수": r["학습_사진수"],
               "기준선": round(r["기준선(남은 호기 val F1최대)"], 4),
               "전체_사진수": a["center"]["n_images"], "전체_이물수": a["center"]["n_gt"],
               "전체_재현율": a["center"]["recall"], "전체_정밀도": a["center"]["precision"], "전체_F1": a["center"]["F1"],
               "전체_놓침": a["center"]["FN"], "전체_오검출": a["center"]["FP"], "전체_AP": a["center"]["AP"],
               "전체_F1_iou50": a["iou50"]["F1"], "test분할_F1": t["center"]["F1"], "test분할_재현율": t["center"]["recall"],
               "실제이물_최저점수": r["빼놓은_호기_실제이물_최저점수"], "합성_검출률": r["빼놓은_호기_합성"]["검출률"]}
        ref = r.get("비교_전체호기학습")
        if ref:
            row.update({"비교모델": ref["모델"], "비교_test분할_F1": ref["빼놓은_호기_test분할"]["center"]["F1"],
                        "비교_test분할_재현율": ref["빼놓은_호기_test분할"]["center"]["recall"],
                        "비교_합성_검출률": ref["빼놓은_호기_합성"]["검출률"]})
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(["합성_배수", "빼놓은_호기"])
    df.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    json.dump(df.to_dict("records"), open(OUT / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(df.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", type=int, choices=[1, 2, 3])
    ap.add_argument("--variants", type=int, default=3, help="학습 사진 한 장당 합성 증강 수 (0 = 합성 없이)")
    ap.add_argument("--model", default="yolo26s.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ref", default=None, help="비교할 전체 호기 학습 모델 (기본: 합성 있으면 ratio3_e100, 없으면 ratio0_e100)")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if args.summary:
        return summary()
    if args.holdout is None:
        ap.error("--holdout 이 필요합니다")
    run(args)


if __name__ == "__main__":
    main()
