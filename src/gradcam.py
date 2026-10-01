"""Grad-CAM 지름길 검증: 모델이 '이물'이라고 할 때 영상의 어디를 근거로 삼는가.

shortcut_test.py 가 행동(성능·미끼 반응)으로 본 것을, 근거 위치(Grad-CAM)로 한 번 더 확인한다.
  대상 모델: 정제본 학습(최종 모델) / 원본 학습(색 표시가 남은 영상으로 학습)
  입력: A 원본(표시 남음) · C 미끼(정제본의 이물 없는 자리에 색 네모만 그림, shortcut_test와 같은 방식·시드)
  방식: 기본 HiResCAM(기울기×활성 원소곱). 평균 기울기를 쓰는 원래 Grad-CAM(--kind gradcam)은
        탐지 모델에서 근거가 영상 전체로 번져 위치 판단에 쓰기 어려웠다 (results/gradcam_plain 에 비교용으로 남김).
  대상 층: P3 층(model.model[16], 8px 간격, 작은 물체를 맡는 층)
  역전파 값: 영상 안 모든 후보 중 최고 신뢰도 (= 영상 단위 판정 근거)

지표 (영상마다 CAM 양수 부분을 합 1로 맞춘 뒤)
  이물 비율 : 정답 이물 박스(±3px) 안에 든 CAM 비율
  미끼 비율 : 미끼 네모(±3px) 안에 든 CAM 비율 (C만)
  최고점 위치: CAM 최댓값이 이물 박스 / 미끼 네모 / 그 밖 어디에 떨어졌나
  면적 비율 : 같은 영역이 영상에서 차지하는 넓이 (우연 수준 비교용)

실행: .venv\\Scripts\\python.exe src\\gradcam.py --clean ratio3_e100 --raw y26s_640_raw
결과: results/gradcam/ (summary.json, per_image.csv, bait_cam.png, raw_cam.png)
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

import metrics as M
from shortcut_test import draw_bait
from train_yolo import weights_path

ROOT = Path(__file__).resolve().parents[1]
IMGSZ = 640
LAYER = 16
PAD = 3


class GradCAM:
    def __init__(self, name, device, kind="hires"):
        from ultralytics import YOLO
        self.kind = kind
        self.m = YOLO(str(weights_path(name))).model.float().eval().to(device)
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.device = device
        self.st = {}

        def hook(_, __, o):
            self.st["a"] = o
            o.register_hook(lambda g: self.st.__setitem__("g", g))
        self.m.model[LAYER].register_forward_hook(hook)

    def __call__(self, rgb):
        """rgb (h, w, 3) uint8 → CAM (h, w) 0 이상, 최고 신뢰도."""
        h, w = rgb.shape[:2]
        r = IMGSZ / max(h, w)
        nh, nw = round(h * r), round(w * r)
        top, left = (IMGSZ - nh) // 2, (IMGSZ - nw) // 2
        canvas = np.full((IMGSZ, IMGSZ, 3), 114, np.uint8)
        canvas[top:top + nh, left:left + nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(canvas).permute(2, 0, 1)[None].float().div(255).to(self.device).requires_grad_(True)
        out = self.m(x)
        s = out[0][:, 4].max()
        self.st.pop("g", None)
        s.backward()
        a, g = self.st["a"][0], self.st["g"][0]
        if self.kind == "hires":   # HiResCAM: 기울기×활성을 칸마다 곱해 위치를 보존 (Draelos & Carin 2020)
            cam = torch.relu((g * a).sum(0)).detach().cpu().numpy()
        else:                      # Grad-CAM: 채널별 평균 기울기로 가중 (Selvaraju 2017)
            cam = torch.relu((g.mean((1, 2), keepdim=True) * a).sum(0)).detach().cpu().numpy()
        cam = cv2.resize(cam, (IMGSZ, IMGSZ), interpolation=cv2.INTER_LINEAR)[top:top + nh, left:left + nw]
        return cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR), float(s.detach())


def region(shape, rects, pad=PAD):
    m = np.zeros(shape, bool)
    for x0, y0, x1, y1 in rects:
        m[max(0, int(y0) - pad):int(np.ceil(y1)) + pad, max(0, int(x0) - pad):int(np.ceil(x1)) + pad] = True
    return m


def overlay(rgb, cam, rects_g=(), rects_b=()):
    v = rgb.astype(np.float32)
    c = cam / (cam.max() + 1e-9)
    heat = cv2.applyColorMap((c * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[:, :, ::-1].astype(np.float32)
    c = cv2.GaussianBlur(c, (0, 0), 1.2)
    c = c / (c.max() + 1e-9)
    a = (0.85 * np.clip(c, 0, 1) ** 0.6)[..., None]     # 근거가 있는 곳에만 색을 입히고 나머지는 원본 그대로
    v = (v * (1 - a) + heat * a).astype(np.uint8)
    for x0, y0, x1, y1 in rects_g:
        cv2.rectangle(v, (int(x0) - 2, int(y0) - 2), (int(x1) + 1, int(y1) + 1), (60, 220, 90), 1)
    for x0, y0, x1, y1 in rects_b:
        cv2.rectangle(v, (int(x0) - 3, int(y0) - 3), (int(x1) + 2, int(y1) + 2), (80, 170, 255), 1)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "data.yaml"))
    ap.add_argument("--clean", default="ratio3_e100")
    ap.add_argument("--raw", default="y26s_640_raw")
    ap.add_argument("--split", default="test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kind", default="hires", choices=["hires", "gradcam"])
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    data = ROOT / cfg["out_dir"]
    out = ROOT / "results" / ("gradcam" if args.kind == "hires" else "gradcam_plain")
    out.mkdir(parents=True, exist_ok=True)

    man = pd.read_csv(data / "manifest.csv")
    man = man[man["labeled"] & (man["split"] == args.split)].set_index("id")
    ids = man.index.tolist()
    gt = M.load_gt(ids, data / "clean" / "labels", {i: (r.w, r.h) for i, r in man.iterrows()})
    marks = pd.read_csv(data / "marks.csv")
    bait = marks[(marks["kind"] == "fake") & marks["id"].isin(ids)]
    real = marks[(marks["kind"] == "real") & marks["id"].isin(ids)]

    rng = np.random.default_rng(args.seed)   # shortcut_test 와 같은 순서로 뽑아 미끼 색을 똑같이 맞춘다
    clean = {i: np.asarray(Image.open(data / "clean/images" / f"{i}.png")) for i in ids}
    inputs = {
        "A_원본": {i: np.asarray(Image.open(data / "raw/images" / f"{i}.png").convert("RGB")) for i in ids},
        "C_미끼": {i: draw_bait(clean[i], bait.loc[bait["id"] == i, ["x0", "y0", "x1", "y1"]].to_numpy(), rng)
                 for i in ids},
    }

    rows, cams = [], {}
    for tag, name in [("정제본학습", args.clean), ("원본학습", args.raw)]:
        gc = GradCAM(name, device, args.kind)
        for cond, imgs in inputs.items():
            for i in ids:
                cam, s = gc(imgs[i])
                cams[(tag, cond, i)] = cam
                shp = cam.shape
                g_m = region(shp, gt[i])
                b_rects = bait.loc[bait["id"] == i, ["x0", "y0", "x1", "y1"]].to_numpy() if cond == "C_미끼" else np.zeros((0, 4))
                b_m = region(shp, b_rects)
                k_m = region(shp, real.loc[real["id"] == i, ["x0", "y0", "x1", "y1"]].to_numpy(), pad=1)
                tot = cam.sum() + 1e-12
                py, px = np.unravel_index(np.argmax(cam), shp)
                peak = "이물" if g_m[py, px] else "미끼" if b_m[py, px] else "그밖"
                rows.append(dict(model=tag, cond=cond, id=i, top_score=round(s, 4),
                                 이물비율=cam[g_m].sum() / tot, 미끼비율=cam[b_m].sum() / tot if len(b_rects) else np.nan,
                                 표시영역비율=cam[k_m].sum() / tot if cond == "A_원본" else np.nan,
                                 이물면적=g_m.mean(), 미끼면적=b_m.mean() if len(b_rects) else np.nan,
                                 최고점=peak))
        del gc
        torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    df.round(4).to_csv(out / "per_image.csv", index=False, encoding="utf-8-sig")

    summary = {}
    for (tag, cond), g in df.groupby(["model", "cond"], sort=False):
        r = {"영상수": len(g), "최고신뢰도_중앙": round(float(g["top_score"].median()), 3),
             "이물비율_평균": round(float(g["이물비율"].mean()), 3), "이물면적_평균": round(float(g["이물면적"].mean()), 3),
             "최고점_이물": round(float((g["최고점"] == "이물").mean()), 3)}
        if cond == "C_미끼":
            r.update(미끼비율_평균=round(float(g["미끼비율"].mean()), 3), 미끼면적_평균=round(float(g["미끼면적"].mean()), 3),
                     최고점_미끼=round(float((g["최고점"] == "미끼").mean()), 3))
        else:
            r["표시영역비율_평균"] = round(float(g["표시영역비율"].mean()), 3)
        summary[f"{tag}/{cond}"] = r
        print(tag, cond, r)
    json.dump(summary, open(out / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # 예시 그림: 행 = 영상, 열 = 입력 / 원본 학습 CAM / 정제본 학습 CAM (초록 = 정답 이물, 하늘색 = 미끼 네모)
    for cond, fname in [("C_미끼", "bait_cam.png"), ("A_원본", "raw_cam.png")]:
        tiles = []
        for i in ids[::max(1, len(ids) // 4)][:4]:
            im = inputs[cond][i]
            b_rects = bait.loc[bait["id"] == i, ["x0", "y0", "x1", "y1"]].to_numpy() if cond == "C_미끼" else ()
            row = [im] + [overlay(im, cams[(t, cond, i)], gt[i], b_rects) for t in ["원본학습", "정제본학습"]]
            # 제품 부분만 잘라 크게 (배경 여백은 근거와 무관)
            ys, xs = np.nonzero(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) < np.median(im) - 25)
            y0, y1 = max(ys.min() - 8, 0), min(ys.max() + 8, im.shape[0])
            x0, x1 = max(xs.min() - 8, 0), min(xs.max() + 8, im.shape[1])
            row = [r[y0:y1, x0:x1] for r in row]
            im = row[0]
            sep = np.full((im.shape[0], 6, 3), 255, np.uint8)
            t = np.hstack([row[0], sep, row[1], sep, row[2]])
            tiles.append(cv2.resize(t, None, fx=900 / t.shape[1] * 2, fy=900 / t.shape[1] * 2,
                                    interpolation=cv2.INTER_NEAREST))
        w = max(t.shape[1] for t in tiles)
        tiles = [np.pad(t, ((0, 10), (0, w - t.shape[1]), (0, 0)), constant_values=255) for t in tiles]
        Image.fromarray(np.vstack(tiles)).save(out / fname)


if __name__ == "__main__":
    main()
