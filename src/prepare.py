"""전처리: 중복 제거 → 라벨 매칭 → 색 표시 제거·흔적 균등화 → 묶음 분할.

실행: .venv\\Scripts\\python.exe src\\prepare.py --config configs\\data.yaml

결과 (out_dir 아래)
  clean/images/*.png   색 표시를 지우고 가짜 사각형 흔적까지 균등화한 흑백 영상 (중복 제거 후 전체)
  clean/labels/*.txt   정답 라벨 (TXT 원본 그대로)
  raw/images/*.png     표시가 남은 원본 RGB (라벨 있는 것만, 지름길 비교 실험용)
  raw/labels/*.txt
  splits/{train,val,test}.txt, clean.yaml, raw.yaml   ultralytics 학습용 목록
  manifest.csv         영상별 출처·해시·호기·날짜·분할
  marks.csv            지운 사각형 목록 (kind=real 원래 표시 / fake 균등화용 가짜)
report_dir 아래
  summary.json, trace_check.json, before_after.png
"""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- 수집·중복 제거

def scan_raw(raw_root: Path) -> pd.DataFrame:
    """원본 BMP를 모두 읽어 해시로 중복을 묶는다. 같은 내용은 첫 경로 하나만 남긴다."""
    rows = []
    for p in sorted(raw_root.glob("*/*/*.bmp")):
        machine = p.parts[-3][0]                       # '1호기(2020.09.22)' → '1'
        date = p.parent.name.split("_")[1]             # 'SN77128_20200622_NgImage' → '20200622'
        rows.append(dict(src=str(p), stem=p.stem, machine=machine, date=date,
                         sha1=hashlib.sha1(p.read_bytes()).hexdigest()))
    df = pd.DataFrame(rows)
    df["n_copies"] = df.groupby("sha1")["src"].transform("size")
    uniq = df.drop_duplicates("sha1", keep="first").copy()
    # 이름이 같은데 내용이 다른 파일이 있어 id에 호기와 해시 앞자리를 붙여 구분한다
    uniq["id"] = "m" + uniq["machine"] + "_" + uniq["stem"]
    clash = uniq["id"].duplicated(keep=False)
    uniq.loc[clash, "id"] += "_" + uniq.loc[clash, "sha1"].str[:6]
    return df, uniq.reset_index(drop=True)


def attach_labels(uniq: pd.DataFrame, label_dir: Path) -> dict:
    """TXT 라벨을 파일명으로 BMP에 연결한다. 이름이 같은 BMP가 여러 개면 멈춘다."""
    labels = {}
    for t in sorted(label_dir.glob("*.txt")):
        hit = uniq[uniq["stem"] == t.stem]
        if len(hit) != 1:
            raise RuntimeError(f"라벨 {t.name} 에 맞는 영상이 {len(hit)}개")
        boxes = np.loadtxt(t, ndmin=2)
        labels[hit.iloc[0]["id"]] = boxes
    return labels


# ---------------------------------------------------------------- 표시 제거

def color_mask(rgb: np.ndarray) -> np.ndarray:
    """R·G·B가 서로 다른 픽셀 = 사람이 그린 표시. X-ray 원본은 전부 회색(R=G=B)이다."""
    return (rgb.max(-1).astype(int) - rgb.min(-1).astype(int) > 0).astype(np.uint8)


def _fill_runs(img, mask, residual, side):
    """행마다 가려진 구간을 양 끝 픽셀 사이 직선으로 채운다.

    같은 구간 폭만큼 옆(side=+1 오른쪽 / -1 왼쪽, 막히면 반대쪽)의 잡음 성분도 빌려 온다.
    (채운 값, 빌린 잡음, 구간 길이) 반환.
    """
    val = np.zeros(img.shape, np.float32)
    tex = np.zeros(img.shape, np.float32)
    run = np.full(img.shape, np.inf, np.float32)
    h, w = img.shape
    for y in np.nonzero(mask.any(1))[0]:
        m = mask[y]
        d = np.diff(np.concatenate([[0], m, [0]]).astype(np.int8))
        for s, e in zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]):
            left = float(img[y, s - 1]) if s > 0 else None
            right = float(img[y, e]) if e < w else None
            if left is None and right is None:
                continue
            left = right if left is None else left
            right = left if right is None else right
            t = (np.arange(s, e) - s + 1) / (e - s + 1)
            val[y, s:e] = left + (right - left) * t
            run[y, s:e] = e - s
            for sd in (side, -side):
                off = sd * (e - s + 1)                 # 한 칸 띄운 바로 옆 띠
                a, b = s + off, e + off
                if a >= 0 and b <= w and not m[a:b].any():
                    tex[y, s:e] = residual[y, a:b]
                    break
    return val, tex, run


def restore(gray: np.ndarray, mask: np.ndarray, rng, texture=True) -> np.ndarray:
    """표시 자리 복원 = 매끈한 바탕(선 보간) + 바로 옆 띠에서 빌려 온 잡음 결.

    세로선은 좌우, 가로선은 위아래 픽셀 사이를 직선으로 잇는다 (가려진 폭이 짧은 방향).
    선을 가로지르는 띠 경계 같은 구조가 이어져, 전방향으로 번지는 Telea 방식보다 덜 뭉개진다.
    이 영상 잡음은 이웃 픽셀끼리 상관이 커서(0.4~0.8) 백색잡음을 더하면 결이 달라지므로,
    가리지 않은 옆 띠의 잡음(고주파 성분)을 덩어리째 옮겨 붙인다.
    """
    f = gray.astype(np.float32)
    valid = (1 - mask).astype(np.float32)
    # 표시 픽셀을 빼고 흐린 바탕 (정규화 합성곱)
    base = cv2.GaussianBlur(f * valid, (0, 0), 1.0) / (cv2.GaussianBlur(valid, (0, 0), 1.0) + 1e-6)
    residual = (f - base) * valid
    side = int(rng.choice([-1, 1]))
    vh, th, rh = _fill_runs(base, mask, residual, side)
    vv, tv, rv = (a.T for a in _fill_runs(base.T, mask.T, residual.T, side))
    fill = np.where(rh < rv, vh, vv) + (np.where(rh < rv, th, tv) if texture else 0)
    tie = rh == rv
    fill[tie] = ((vh + vv) / 2 + ((th + tv) / 2 if texture else 0))[tie]
    return np.where(mask > 0, np.clip(fill.round(), 0, 255), gray).astype(np.uint8)


def product_mask(gray: np.ndarray) -> np.ndarray:
    """배경보다 어두운 제품 영역 (Otsu). 가짜 사각형은 이 안에만 놓는다."""
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    _, m = cv2.threshold(blur, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return cv2.erode(m, np.ones((15, 15), np.uint8))


def yolo_to_xyxy(boxes: np.ndarray, w: int, h: int) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros((0, 4))
    cx, cy, bw, bh = boxes[:, 1] * w, boxes[:, 2] * h, boxes[:, 3] * w, boxes[:, 4] * h
    return np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)


def place_fakes(n, gray, forbid, rng, sides, thick):
    """제품 영역 안, 진짜 표시·라벨과 겹치지 않는 곳에 가짜 사각형 테두리 n개를 그린 마스크."""
    h, w = gray.shape
    cand = product_mask(gray) & (forbid == 0)
    ys, xs = np.nonzero(cand)
    out = np.zeros_like(gray, np.uint8)
    rects = []
    for _ in range(n):
        for _try in range(200):
            if len(xs) == 0:
                break
            i = rng.integers(len(xs))
            s = int(rng.choice(sides))
            x0, y0 = int(xs[i]) - s // 2, int(ys[i]) - s // 2
            if x0 < 0 or y0 < 0 or x0 + s > w or y0 + s > h:
                continue
            ring = np.zeros_like(out)
            cv2.rectangle(ring, (x0, y0), (x0 + s - 1, y0 + s - 1), 1, thick)
            area = np.zeros_like(out)
            area[y0:y0 + s, x0:x0 + s] = 1
            if (area & (forbid | out)).any():          # 진짜 영역·다른 가짜와 겹치면 다시
                continue
            out |= ring
            rects.append((x0, y0, x0 + s, y0 + s))
            break
    return out, rects


def clean_image(rgb, boxes, rng, cfg):
    """표시를 지우고, 같은 수의 가짜 사각형을 그렸다가 똑같이 지운다.

    지운 자리는 주변보다 매끈해지는 흔적이 남는다. 이 흔적이 진짜 이물 근처에만 있으면
    모델이 흔적을 보고 맞히므로, 이물 없는 곳에도 같은 흔적을 만들어 단서 가치를 없앤다.
    """
    gray = np.asarray(Image.fromarray(rgb).convert("L"))
    h, w = gray.shape
    real = color_mask(rgb)
    if cfg["mask_dilate"] > 0:
        k = 2 * cfg["mask_dilate"] + 1
        real = cv2.dilate(real, np.ones((k, k), np.uint8))

    # 가짜 금지 영역 = 진짜 표시 주변 + 라벨 박스 주변
    forbid = cv2.dilate(real, np.ones((9, 9), np.uint8))
    for x0, y0, x1, y1 in yolo_to_xyxy(boxes, w, h).round().astype(int):
        forbid[max(0, y0 - 12):y1 + 12, max(0, x0 - 12):x1 + 12] = 1

    n_fake = int(round(cfg["fake_per_real"] * max(len(boxes), 1)))
    fake, fake_rects = place_fakes(n_fake, gray, forbid, rng, cfg["fake_side"], cfg["fake_thickness"])
    mask = real | fake

    if cfg["method"] == "line":
        out = restore(gray, mask, rng, cfg["texture"])
    else:
        out = cv2.inpaint(gray, mask, cfg["inpaint_radius"], cv2.INPAINT_TELEA)
    return gray, out, real, fake, fake_rects


def real_mark_rects(real: np.ndarray):
    n, _, st, _ = cv2.connectedComponentsWithStats(real, connectivity=8)
    return [(x, y, x + bw, y + bh) for x, y, bw, bh, _ in st[1:]]


# ---------------------------------------------------------------- 흔적 검증

def ring_features(img: np.ndarray, ring: np.ndarray) -> list:
    """사각형 테두리 픽셀이 바로 바깥 픽셀과 얼마나 다른지 (잡음 세기·밝기·경사)."""
    outer = cv2.dilate(ring, np.ones((5, 5), np.uint8)) & (1 - ring)
    f = img.astype(np.float32)
    res = f - cv2.medianBlur(img, 3).astype(np.float32)
    grad = np.abs(cv2.Laplacian(f, cv2.CV_32F))
    r, o = ring > 0, outer > 0
    return [res[r].std() / (res[o].std() + 1e-6),
            f[r].mean() - f[o].mean(),
            grad[r].mean() / (grad[o].mean() + 1e-6)]


def _rings(mask):
    n, lab = cv2.connectedComponents(mask, connectivity=8)
    return [(lab == k).astype(np.uint8) for k in range(1, n)]


def _auc(pos, neg):
    """테두리 특징별 AUC. 방향과 무관한 구분력 |AUC-0.5|+0.5 의 최댓값도 같이 준다."""
    names = ["잡음비", "밝기차", "경사비"]
    X, y = np.array(pos + neg), np.array([1] * len(pos) + [0] * len(neg))
    aucs = {nm: round(float(roc_auc_score(y, X[:, i])), 3) for i, nm in enumerate(names)}
    return {"n": [len(pos), len(neg)], "auc": aucs,
            "auc_max": round(max(abs(a - 0.5) + 0.5 for a in aucs.values()), 3)}


def trace_check(records):
    """지운 자리가 티 나는지 테두리 특징(잡음비·밝기차·경사비)의 AUC로 잰다. 0.5 = 구분 불가.

    A 흑백 변환만: 표시 자리(흑백) vs 손대지 않은 자리        → 흑백 변환만으로 부족함을 보이는 대조군
    B 복원 흔적:   가짜 자리 복원 후 vs 같은 자리 복원 전      → 복원 방법 자체가 흔적을 남기는지 (핵심 지표)
    C 정제 후:     진짜 표시 자리 vs 가짜 표시 자리 (둘 다 복원) → 참고용. 진짜 자리는 띠 왼쪽 끝이라 배경 구조가 달라 0.5가 아니어도 된다
    """
    feats = defaultdict(list)
    for rec in records:
        for r in _rings(rec["real"]):
            feats["real_gray"].append(ring_features(rec["gray_only"], r))
            feats["real_clean"].append(ring_features(rec["clean"], r))
        for r in _rings(rec["fake"]):
            feats["fake_before"].append(ring_features(rec["gray_only"], r))
            feats["fake_clean"].append(ring_features(rec["clean"], r))
    return {"A_흑백변환만": _auc(feats["real_gray"], feats["fake_before"]),
            "B_복원흔적": _auc(feats["fake_clean"], feats["fake_before"]),
            "C_정제후_진짜vs가짜": _auc(feats["real_clean"], feats["fake_clean"])}


# ---------------------------------------------------------------- 분할

def group_split(meta: pd.DataFrame, ratios: dict, seed: int) -> pd.Series:
    """(호기, 날짜) 묶음 단위로 나눈다. 호기마다 따로, 무작위 배정 여러 번 중 비율에 가장 가까운 것을 쓴다."""
    rng = np.random.default_rng(seed)
    split = pd.Series("", index=meta.index)
    names = list(ratios)
    for _, m in meta.groupby("machine"):
        groups = m.groupby("date").size()
        target = np.array([ratios[k] * len(m) for k in names])
        best, best_err = None, np.inf
        for _ in range(5000):
            assign = rng.integers(len(names), size=len(groups))
            have = np.bincount(assign, weights=groups.to_numpy(), minlength=len(names))
            if (have == 0).any():
                continue
            err = np.abs(have - target).sum()
            if err < best_err:
                best, best_err = assign, err
        for g, a in zip(groups.index, best):
            split[m.index[m["date"] == g]] = names[a]
    return split


# ---------------------------------------------------------------- 결과 그림

def before_after(records, path: Path, n=6):
    tiles = []
    for rec in records[:n]:
        x0, y0, x1, y1 = rec["crop"]
        a = cv2.resize(rec["rgb"][y0:y1, x0:x1], None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
        b = cv2.resize(rec["clean"][y0:y1, x0:x1], None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
        b = cv2.cvtColor(b, cv2.COLOR_GRAY2RGB)
        gap = np.full((a.shape[0], 8, 3), 255, np.uint8)
        tiles.append(np.hstack([a, gap, b]))
    wmax = max(t.shape[1] for t in tiles)
    tiles = [np.pad(t, ((0, 8), (0, wmax - t.shape[1]), (0, 0)), constant_values=255) for t in tiles]
    Image.fromarray(np.vstack(tiles)).save(path)


def crop_around(rects, w, h, pad=30):
    r = np.array(rects)
    x0, y0 = max(0, r[:, 0].min() - pad), max(0, r[:, 1].min() - pad)
    x1, y1 = min(w, r[:, 2].max() + pad), min(h, r[:, 3].max() + pad)
    return int(x0), int(y0), int(x1), int(y1)


# ---------------------------------------------------------------- 실행

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "data.yaml"))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    out = ROOT / cfg["out_dir"]
    rep = ROOT / cfg["report_dir"]
    for d in ["clean/images", "clean/labels", "raw/images", "raw/labels", "splits"]:
        (out / d).mkdir(parents=True, exist_ok=True)
    rep.mkdir(parents=True, exist_ok=True)

    all_df, uniq = scan_raw(Path(cfg["raw_root"]))
    labels = attach_labels(uniq, Path(cfg["label_dir"]))
    uniq["labeled"] = uniq["id"].isin(labels.keys())
    uniq["n_boxes"] = uniq["id"].map(lambda i: len(labels.get(i, [])))

    mark_rows, records, sizes, overlap = [], [], [], {"박스": 0, "중심3x3": 0}
    for row in tqdm(uniq.itertuples(), total=len(uniq), desc="표시 제거"):
        rgb = np.asarray(Image.open(row.src).convert("RGB"))
        h, w = rgb.shape[:2]
        sizes.append((w, h))
        boxes = labels.get(row.id, np.zeros((0, 5)))
        # 영상마다 고정된 난수 → 다시 돌려도 같은 결과
        rng = np.random.default_rng([cfg["seed"], int(row.sha1[:8], 16)])
        gray, clean, real, fake, fake_rects = clean_image(rgb, boxes, rng, cfg["clean"])
        Image.fromarray(clean).save(out / "clean/images" / f"{row.id}.png")

        real_rects = real_mark_rects(real)
        for kind, rects in [("real", real_rects), ("fake", fake_rects)]:
            for r in rects:
                mark_rows.append(dict(id=row.id, kind=kind, x0=r[0], y0=r[1], x1=r[2], y1=r[3]))

        if row.labeled:
            np.savetxt(out / "clean/labels" / f"{row.id}.txt", boxes, fmt="%d %.6f %.6f %.6f %.6f")
            np.savetxt(out / "raw/labels" / f"{row.id}.txt", boxes, fmt="%d %.6f %.6f %.6f %.6f")
            Image.fromarray(rgb).save(out / "raw/images" / f"{row.id}.png")
            # 라벨 박스 안에 표시 픽셀이 있으면 이물 픽셀 일부가 복원 과정에서 바뀌었을 수 있다
            for x0, y0, x1, y1 in yolo_to_xyxy(boxes, w, h).round().astype(int):
                overlap["박스"] += int(real[max(0, y0):y1, max(0, x0):x1].any())
                cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
                overlap["중심3x3"] += int(real[max(0, cy - 1):cy + 2, max(0, cx - 1):cx + 2].any())
            records.append(dict(rgb=rgb, gray_only=gray, clean=clean, real=real, fake=fake,
                                crop=crop_around(real_rects + fake_rects, w, h)))

    uniq["w"], uniq["h"] = zip(*sizes)
    lab = uniq[uniq["labeled"]]
    uniq["split"] = ""
    uniq.loc[lab.index, "split"] = group_split(lab, cfg["split"], cfg["seed"])

    # ultralytics 목록 파일과 데이터셋 yaml
    for s in cfg["split"]:
        ids = uniq.loc[uniq["split"] == s, "id"]
        (out / "splits" / f"{s}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
        for v in ["clean", "raw"]:
            paths = [str((out / v / "images" / f"{i}.png").resolve()) for i in ids]
            (out / v / f"{s}.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
    for v in ["clean", "raw"]:
        ds = {"path": str((out / v).resolve()), "train": "train.txt", "val": "val.txt",
              "test": "test.txt", "names": {0: "Defect"}}
        yaml.safe_dump(ds, open(out / f"{v}.yaml", "w", encoding="utf-8"), allow_unicode=True)

    dup_src = all_df.groupby("sha1")["src"].apply(lambda s: "|".join(s.iloc[1:]))
    uniq["dup_removed"] = uniq["sha1"].map(dup_src)
    uniq.drop(columns=["stem"]).to_csv(out / "manifest.csv", index=False, encoding="utf-8-sig")
    marks = pd.DataFrame(mark_rows)
    marks.to_csv(out / "marks.csv", index=False, encoding="utf-8-sig")

    check = trace_check(records)
    json.dump(check, open(rep / "trace_check.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    before_after(records[::80], rep / "before_after.png")

    lab = uniq[uniq["labeled"]]
    summary = {
        "원본_BMP": len(all_df),
        "중복_제거": int(len(all_df) - len(uniq)),
        "고유_영상": len(uniq),
        "라벨_영상": int(len(lab)),
        "라벨_박스": int(lab["n_boxes"].sum()),
        "표시가_걸친_라벨박스_수": overlap,
        "분할_영상수": lab.groupby("split").size().to_dict(),
        "분할_박스수": lab.groupby("split")["n_boxes"].sum().astype(int).to_dict(),
        "분할별_호기": {s: g.groupby("machine").size().to_dict() for s, g in lab.groupby("split")},
        "분할별_날짜묶음": lab.groupby("split")["date"].apply(lambda d: sorted(set(d))).to_dict(),
        "지운_사각형": marks.groupby("kind").size().to_dict(),
        "흔적_구분력_AUC": {k: v["auc_max"] for k, v in check.items()},
    }
    json.dump(summary, open(rep / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
