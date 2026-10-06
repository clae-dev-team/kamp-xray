"""최종 모델로 test 영상을 예측해 제출용 결과 파일을 만든다.

판정 기준선(합격선·불합격선)은 risk_threshold.py 가 val 에서 정한 보장 기준선을 쓴다 (기본값 --policy guarantee).
  그 결과 파일이 없을 때만 judge.py 가 val 에서 정한 값을 쓴다.
  영상 판정: 영상 최고 점수 < 합격선 → 합격, < 불합격선 → 재검사, 그 이상 → 불합격
  박스는 합격선 이상인 것만 남긴다 (재검사·불합격 근거 위치).

결과 (results/submission/)
  test_boxes.csv        id, x0, y0, x1, y1, score, 박스판정
  test_images.csv       id, 호기, 최고점수, 박스수, 판정
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

ROOT = Path(__file__).resolve().parents[1]   # 저장소 최상위 폴더


def run(model, rows, img_dir, t_low, t_high, batch=32):
    """영상들을 예측해 박스 표와 영상 표를 만든다.

    rows: manifest 의 행들(열 id, machine, w, h 를 쓴다). img_dir: <id>.png 가 있는 폴더.
    t_low · t_high: 합격선 · 불합격선(신뢰도 0~1). batch: 한 번에 추론하는 장수.
    반환: (박스 표, 영상 표).
      박스 표: id, x0, y0, x1, y1(원본 픽셀), score, 박스판정, w, h. 합격선 이상인 박스만 들어간다.
      영상 표: id, 호기, 최고점수, 박스수, 판정. 영상 한 장이 한 행이다.
    """
    boxes, images = [], []
    for k in tqdm(range(0, len(rows), batch), desc="예측"):
        chunk = rows.iloc[k:k + batch]
        # 추론 설정(640px, conf 0.001, 영상당 100개)은 기준선을 정한 judge.py · risk_threshold.py 와 같다
        res = model.predict([str(img_dir / f"{i}.png") for i in chunk["id"]], imgsz=640, conf=0.001,
                            max_det=100, verbose=False)
        for (_, r), p in zip(chunk.iterrows(), res):
            xyxy, conf = p.boxes.xyxy.cpu().numpy(), p.boxes.conf.cpu().numpy()
            top = float(conf.max()) if len(conf) else 0.0   # 영상 점수 = 박스 최고 신뢰도, 박스가 없으면 0
            keep = conf >= t_low
            for (x0, y0, x1, y1), s in zip(xyxy[keep], conf[keep]):
                # w, h 는 뒤에서 YOLO 형식(0~1 비율)으로 바꿀 때 쓰려고 함께 적어 둔다
                boxes.append(dict(id=r["id"], x0=round(float(x0), 2), y0=round(float(y0), 2), x1=round(float(x1), 2),
                                  y1=round(float(y1), 2), score=round(float(s), 4),
                                  박스판정="불합격" if s >= t_high else "재검사", w=r["w"], h=r["h"]))
            # 영상 판정은 반올림하지 않은 최고 점수로 한다. 표에는 소수 넷째 자리까지 적는다
            images.append(dict(id=r["id"], 호기=r["machine"], 최고점수=round(top, 4), 박스수=int(keep.sum()),
                               판정="불합격" if top >= t_high else ("재검사" if top >= t_low else "합격")))
    return pd.DataFrame(boxes), pd.DataFrame(images)


def main():
    """기준선을 고르고, 지정한 분할을 예측해 results/submission/ 에 제출 파일을 쓴다."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="ratio3_e100")
    ap.add_argument("--split", default="test")
    ap.add_argument("--unlabeled", action="store_true")
    ap.add_argument("--policy", default="guarantee", choices=["guarantee", "spec", "all"],
                    help="guarantee = 보장 기준선(risk_threshold.py, 권장) / spec = 사양 기준(최약 불량 점수) / all = val 불량 전체 99%")
    args = ap.parse_args()
    from ultralytics import YOLO

    cfg = yaml.safe_load(open(ROOT / "configs" / "data.yaml", encoding="utf-8"))
    data = ROOT / cfg["out_dir"]
    out = ROOT / "results" / "submission"
    (out / "labels").mkdir(parents=True, exist_ok=True)
    # 기준선 고르기. 위에서부터 조건이 맞는 첫 번째를 쓴다
    #   보장기준   : risk_threshold.py 의 채택값. --policy guarantee 이고 그 채택값이 이 모델의 것일 때
    #   사양기준   : judge.py 의 사양 기준. --policy spec 이거나 보장 기준선을 쓸 수 없을 때
    #   불량전체99 : judge.py 의 기본 기준선. --policy all 이거나 사양 기준이 없을 때
    js = json.load(open(ROOT / "results/judge/summary.json", encoding="utf-8"))[args.name]
    rf = ROOT / "results/risk_threshold/summary.json"
    rk = json.load(open(rf, encoding="utf-8")).get("채택") if rf.exists() else None
    if args.policy == "guarantee" and rk and rk.get("모델") == args.name:
        policy, th = "보장기준", rk
    elif args.policy in ("guarantee", "spec") and "사양기준" in js:
        policy, th = "사양기준", js["사양기준"]["기준선"]
    else:
        policy, th = "불량전체99", js["기준선"]
    t_low, t_high = th["합격선"], th["불합격선"]   # 요약 파일에 소수 넷째 자리로 적힌 값을 그대로 쓴다
    model = YOLO(str(weights_path(args.name)))

    # 정답(라벨)이 있는 영상 중 지정한 분할만 예측한다. 영상은 색 표시를 지운 정제본을 쓴다
    man = pd.read_csv(data / "manifest.csv")
    rows = man[man["labeled"] & (man["split"] == args.split)]
    boxes, images = run(model, rows, data / "clean/images", t_low, t_high)
    boxes.drop(columns=["w", "h"]).to_csv(out / f"{args.split}_boxes.csv", index=False, encoding="utf-8-sig")
    images.to_csv(out / f"{args.split}_images.csv", index=False, encoding="utf-8-sig")
    # 영상별 YOLO 형식 라벨: class cx cy w h score. 좌표는 픽셀을 영상 너비 · 높이로 나눈 0~1 비율이고 class 는 0 하나다
    for i, g in boxes.groupby("id") if len(boxes) else []:
        w, h = g["w"].iloc[0], g["h"].iloc[0]
        lines = [f"0 {(r.x0 + r.x1) / 2 / w:.6f} {(r.y0 + r.y1) / 2 / h:.6f} {(r.x1 - r.x0) / w:.6f} "
                 f"{(r.y1 - r.y0) / h:.6f} {r.score:.4f}" for r in g.itertuples()]
        (out / "labels" / f"{i}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for i in set(rows["id"]) - set(boxes["id"] if len(boxes) else []):
        (out / "labels" / f"{i}.txt").write_text("", encoding="utf-8")        # 합격 영상은 빈 파일
    # 어떤 모델 · 어떤 기준선으로 만든 결과인지 함께 남긴다
    json.dump({"model": args.name, "weights": str(weights_path(args.name).relative_to(ROOT)),
               "합격선": t_low, "불합격선": t_high, "기준선_방식": policy,
               "기준선_출처": "results/risk_threshold/summary.json" if policy == "보장기준" else "results/judge/summary.json",
               "보장": th.get("보장") if policy == "보장기준" else None},
              open(out / "thresholds.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(images["판정"].value_counts().to_dict(), "박스", len(boxes))

    # 라벨 없는 영상은 영상 단위 판정만 남긴다 (박스 파일과 labels 는 쓰지 않는다)
    if args.unlabeled:
        un = man[~man["labeled"]]
        _, im = run(model, un, data / "clean/images", t_low, t_high)
        im.to_csv(out / "unlabeled_images.csv", index=False, encoding="utf-8-sig")
        print("라벨 없는 영상", im["판정"].value_counts().to_dict())


if __name__ == "__main__":
    main()
