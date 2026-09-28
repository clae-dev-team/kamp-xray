"""최종 모델로 test 영상을 예측해 제출용 결과 파일을 만든다.

판정 기준선(합격선·불합격선)은 judge.py 가 val 에서 정한 값을 그대로 쓴다.
  영상 판정: 영상 최고 점수 < 합격선 → 합격, < 불합격선 → 재검사, 그 이상 → 불합격
  박스는 합격선 이상인 것만 남긴다 (재검사·불합격 근거 위치).

결과 (results/submission/)
  test_boxes.csv        id, x0, y0, x1, y1, score, 박스판정
  test_images.csv       id, 호기, 최고점수, 판정, 박스수
  labels/<id>.txt       YOLO 형식 (class cx cy w h score, 0~1 정규화)
  thresholds.json       사용한 모델·기준선

실행: .venv\\Scripts\\python.exe src\\predict.py --name ratio3_e100 [--split test] [--unlabeled]
  --unlabeled 를 주면 라벨 없는 영상(약 2,000장)도 따로 예측해 unlabeled_images.csv 로 남긴다 (참고용).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from train_yolo import weights_path

ROOT = Path(__file__).resolve().parents[1]


def run(model, rows, img_dir, t_low, t_high, batch=32):
    boxes, images = [], []
    for k in tqdm(range(0, len(rows), batch), desc="예측"):
        chunk = rows.iloc[k:k + batch]
        res = model.predict([str(img_dir / f"{i}.png") for i in chunk["id"]], imgsz=640, conf=0.001,
                            max_det=100, verbose=False)
        for (_, r), p in zip(chunk.iterrows(), res):
            xyxy, conf = p.boxes.xyxy.cpu().numpy(), p.boxes.conf.cpu().numpy()
            top = float(conf.max()) if len(conf) else 0.0
            keep = conf >= t_low
            for (x0, y0, x1, y1), s in zip(xyxy[keep], conf[keep]):
                boxes.append(dict(id=r["id"], x0=round(float(x0), 2), y0=round(float(y0), 2), x1=round(float(x1), 2),
                                  y1=round(float(y1), 2), score=round(float(s), 4),
                                  박스판정="불합격" if s >= t_high else "재검사", w=r["w"], h=r["h"]))
            images.append(dict(id=r["id"], 호기=r["machine"], 최고점수=round(top, 4), 박스수=int(keep.sum()),
                               판정="불합격" if top >= t_high else ("재검사" if top >= t_low else "합격")))
    return pd.DataFrame(boxes), pd.DataFrame(images)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="ratio3_e100")
    ap.add_argument("--split", default="test")
    ap.add_argument("--unlabeled", action="store_true")
    args = ap.parse_args()
    from ultralytics import YOLO

    cfg = yaml.safe_load(open(ROOT / "configs" / "data.yaml", encoding="utf-8"))
    data = ROOT / cfg["out_dir"]
    out = ROOT / "results" / "submission"
    (out / "labels").mkdir(parents=True, exist_ok=True)
    th = json.load(open(ROOT / "results/judge/summary.json", encoding="utf-8"))[args.name]["기준선"]
    t_low, t_high = th["합격선"], th["불합격선"]
    model = YOLO(str(weights_path(args.name)))

    man = pd.read_csv(data / "manifest.csv")
    rows = man[man["labeled"] & (man["split"] == args.split)]
    boxes, images = run(model, rows, data / "clean/images", t_low, t_high)
    boxes.drop(columns=["w", "h"]).to_csv(out / f"{args.split}_boxes.csv", index=False, encoding="utf-8-sig")
    images.to_csv(out / f"{args.split}_images.csv", index=False, encoding="utf-8-sig")
    for i, g in boxes.groupby("id") if len(boxes) else []:
        w, h = g["w"].iloc[0], g["h"].iloc[0]
        lines = [f"0 {(r.x0 + r.x1) / 2 / w:.6f} {(r.y0 + r.y1) / 2 / h:.6f} {(r.x1 - r.x0) / w:.6f} "
                 f"{(r.y1 - r.y0) / h:.6f} {r.score:.4f}" for r in g.itertuples()]
        (out / "labels" / f"{i}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for i in set(rows["id"]) - set(boxes["id"] if len(boxes) else []):
        (out / "labels" / f"{i}.txt").write_text("", encoding="utf-8")        # 합격 영상은 빈 파일
    json.dump({"model": args.name, "weights": str(weights_path(args.name).relative_to(ROOT)),
               "합격선": t_low, "불합격선": t_high, "기준선_출처": "results/judge/summary.json (val에서 결정)"},
              open(out / "thresholds.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(images["판정"].value_counts().to_dict(), "박스", len(boxes))

    if args.unlabeled:
        un = man[~man["labeled"]]
        _, im = run(model, un, data / "clean/images", t_low, t_high)
        im.to_csv(out / "unlabeled_images.csv", index=False, encoding="utf-8-sig")
        print("라벨 없는 영상", im["판정"].value_counts().to_dict())


if __name__ == "__main__":
    main()
