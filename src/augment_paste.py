"""진짜 점 이식 증강: 실제 이물 점을 잘라 다른 자리에 옮겨 붙인다 (작은 물체 복사-붙이기, Kisantal 2019).

realism.py 에서 구 모양 합성 이물은 실제 이물과 거의 완벽히 구분됐다 (AUC 0.998). 실제 이물은 픽셀 칸에 맞은
2~4칸짜리 고른 덩어리인데 구 합성은 십자 모양으로 번지기 때문이다. 그래서 모양을 만들지 않고 진짜 점을 그대로 쓴다.

  1. 점 은행: 사진에서 이물 점만 지운 배경(er)을 만들고, 점 영역의 투과율 T = 실제 / 지운 배경 을 9×9 조각으로 저장.
     X선은 겹친 물체의 투과율이 곱해지므로(Beer–Lambert) 더하지 않고 곱으로 옮긴다 → 밝은 자리에선 많이, 어두운 자리에선 적게 어두워진다.
  2. 이식: 같은 호기의 다른 사진 점을 고르고, 좌우·상하 뒤집기와 대각 뒤집기를 무작위로 한 뒤 T^s 를 곱한다.
     s 는 세기(같은 재질에서 두께가 s 배). s < 1 이면 실제보다 옅은 이물이 된다.
  3. 은행은 train 사진으로만 만든다 (평가용 은행은 paste_eval.py 가 test 사진으로 따로 만든다).

  --mode paste : 합성 이물을 모두 이식으로 (data/aug_paste)
  --mode mix   : 이물마다 절반은 구 합성(augment.py 와 같은 조건), 절반은 이식 (data/aug_mix)
  --mode ref   : 인접 프레임을 기준으로 떼어 낸 이식만 (reference_residual.py, data/aug_ref)
  --mode all3  : 구 합성 · 이식 · 인접 프레임 이식을 1/3 씩 (data/aug_all3)
  --placement context : 인접 프레임 이식의 자리를 '원래 자리와 둘레 구조가 닮은 자리'로 고른다 (결과 폴더 이름 뒤에 ctx)
자리 뽑기·장수·한 장당 개수는 augment.py 와 같다 (비교를 위해 모양만 다르게).
인접 프레임 기준은 이물의 약 1/4 에서만 찾아지므로, 그 호기에 조각이 없거나 놓을 자리가 없으면 그 이물은 넣지 않고 수를 config.json 에 남긴다.

실행: .venv\\Scripts\\python.exe src\\augment_paste.py --mode paste
결과: data/aug_<mode>/{images,labels,train.txt,defects.csv,config.json}, data/aug_<mode>.yaml
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm

import augment as A
import reference_residual as RR
from location_test import dot_mask
from normal_set import dot_centers
from prepare import product_mask, restore
from synth import band_mask, insert

ROOT = Path(__file__).resolve().parents[1]
R = 4                        # 조각 반폭 (9×9)
S_RANGE = (0.15, 1.0)        # 이식 세기
MIN_DEPTH = 0.05             # 이보다 옅게 잘린 점은 은행에서 뺀다 (지우기 실패)
SEED_OFFSET = 15485863       # 다른 합성 스크립트와 겹치지 않게
# --mode 별로 섞는 이물 종류. 구 = synth.insert 합성, 이식 = 지운 배경 기준 조각, 기준이식 = 인접 프레임 기준 조각
KINDS = {"paste": ["이식"], "mix": ["구", "이식"], "ref": ["기준이식"], "all3": ["구", "이식", "기준이식"]}
TOPK = 8                     # 구조 맞춤 자리: 비용이 낮은 후보 TOPK 곳 가운데 무작위 (학습 사진마다 자리가 달라지게)


def build_bank(data, rows, seed):
    """{호기: [(사진 id, T 조각 9×9)]}. rows = manifest 의 해당 분할 행.

    data: 전처리 결과 폴더, seed: configs/data.yaml 의 시드 (지운 배경을 만들 때 쓴다).
    T 조각은 float32 투과율로, 점 영역 밖은 1 · 점 영역은 0.02~1 이다. 호기 번호는 int 로 바꿔 열쇠로 쓴다.
    """
    bank = {}
    for r in tqdm(list(rows.itertuples()), desc="점 은행"):
        g = np.asarray(Image.open(data / "clean/images" / f"{r.id}.png"))
        h, w = g.shape
        # dot_centers 는 픽셀 경계 기준 좌표(한가운데가 k+0.5)를 주고, dot_mask 는 한가운데가 정수인 좌표를 받으므로 0.5 를 뺀다
        cs = dot_centers(g, np.loadtxt(data / "clean/labels" / f"{r.id}.txt", ndmin=2), w, h)
        masks = [dot_mask(g, cx - 0.5, cy - 0.5) for cx, cy in cs]
        # 한 사진의 이물 점을 모두 합쳐 한 번에 지운다. er = 이물만 지운 배경
        union = np.zeros_like(g)
        for m in masks:
            union |= m
        er = restore(g, union, np.random.default_rng([seed + SEED_OFFSET, int(r.sha1[:8], 16)])).astype(np.float32)
        for (cx, cy), m in zip(cs, masks):
            x, y = int(round(cx - 0.5)), int(round(cy - 0.5))
            # 9×9 조각이 영상 밖으로 나가는 점은 은행에 넣지 않는다
            if x - R < 0 or y - R < 0 or x + R + 1 > w or y + R + 1 > h:
                continue
            sl = (slice(y - R, y + R + 1), slice(x - R, x + R + 1))
            # 투과율 = 실제 / 지운 배경. 이 점의 마스크 밖은 1 이라 옮겨 붙여도 그 자리는 바뀌지 않는다 (0 으로 나누지 않게 분모는 1 이상)
            T = np.where(m[sl] > 0, g[sl].astype(np.float32) / np.maximum(er[sl], 1), 1.0)
            # 1 을 넘는 값(지운 배경보다 밝은 픽셀)은 1 로, 0.02 아래는 0.02 로 자른다
            T = np.clip(T, 0.02, 1.0)
            # 가장 어두운 픽셀의 깊이(1 - 최소 투과율)가 MIN_DEPTH 이상인 점만 쓴다
            if 1 - T.min() >= MIN_DEPTH:
                bank.setdefault(int(r.machine), []).append((r.id, T))
    return bank


def paste(f, x, y, T, s, rng):
    """f(float32)의 (x, y) 화소 중심에 점 조각을 세기 s 로 곱해 넣는다. 영상 밖으로 나가면 False.

    x, y: 조각 한가운데가 놓일 픽셀의 정수 좌표, T: (9, 9) 투과율 조각, s: 세기(T 의 지수).
    f 를 제자리에서 고친다. 넣었으면 True.
    """
    h, w = f.shape
    if x - R < 0 or y - R < 0 or x + R + 1 > w or y + R + 1 > h:
        return False
    # 좌우 뒤집기, 상하 뒤집기, 대각 뒤집기(전치)를 각각 절반 확률로 해서 8가지 방향이 나온다
    if rng.random() < 0.5:
        T = T[:, ::-1]
    if rng.random() < 0.5:
        T = T[::-1]
    if rng.random() < 0.5:
        T = T.T
    # 두께가 s 배면 exp(-μ·s·t) = T^s 이므로 세기는 지수로 준다
    f[y - R:y + R + 1, x - R:x + R + 1] *= T ** s
    return True


def main():
    """train 영상마다 이식(또는 혼합) 증강 영상을 만들고 학습 목록·데이터셋 yaml 을 쓴다.

    defects.csv 는 넣은 이물별 img, src, machine, cx, cy(px), in_band, kind(구/이식/기준이식), d, c0(구만), s(이식만), donor(조각을 떼어 온 사진 id).
    인접 프레임 이식을 쓰는 모드는 찾은 기준 프레임 목록을 references.csv 로도 남긴다.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="paste", choices=list(KINDS))
    ap.add_argument("--placement", default="random", choices=["random", "context"])
    ap.add_argument("--variants", type=int, default=A.VARIANTS)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "configs" / "data.yaml", encoding="utf-8"))
    data = ROOT / cfg["out_dir"]
    kinds = KINDS[args.mode]
    tag = f"aug_{args.mode}" + ("ctx" if args.placement == "context" else "")
    out = data / tag
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(data / "manifest.csv")
    tr = man[man["labeled"] & (man["split"] == "train")]
    # 은행은 쓰는 종류만 만든다. bank = 지운 배경 기준 조각, ref_bank = 인접 프레임 기준 조각 (둘 다 호기별)
    bank = build_bank(data, tr, cfg["seed"]) if "이식" in kinds else {}
    ref_bank, refs = RR.build_ref_bank(data, tr) if "기준이식" in kinds else ({}, [])

    # skipped = 조각이 없거나 놓을 자리가 없어 넣지 못한 이물 수 (config.json 의 n_skipped)
    rows, paths, skipped = [], [], 0
    for r in tqdm(list(tr.itertuples()), desc="이식 증강"):
        g = np.asarray(Image.open(data / "clean/images" / f"{r.id}.png"))
        h, w = g.shape
        real = np.loadtxt(data / "clean/labels" / f"{r.id}.txt", ndmin=2)
        pm = product_mask(g)
        band = band_mask(g, pm)
        # 실제 이물 박스와 그 둘레 10px 에는 넣지 않는다. real_xyxy 는 구조 맞춤 자리 고르기에 넘길 실제 박스(px)
        forbid = np.zeros_like(pm)
        real_xyxy = []
        for b in real:
            cx, cy, bw, bh = b[1] * w, b[2] * h, b[3] * w, b[4] * h
            real_xyxy.append((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
            forbid[max(0, int(cy - bh / 2 - 10)):int(cy + bh / 2 + 10),
                   max(0, int(cx - bw / 2 - 10)):int(cx + bw / 2 + 10)] = 1
        cand = {"band": np.argwhere(band & (forbid == 0)), "any": np.argwhere((pm > 0) & (forbid == 0))}
        # 조각은 같은 호기의 다른 사진에서만 가져온다 (자기 사진의 점은 뺀다)
        donors = [d for d in bank.get(int(r.machine), []) if d[0] != r.id]
        ref_donors = [d for d in ref_bank.get(int(r.machine), []) if d["id"] != r.id]
        # 구조 맞춤 자리 고르기는 사진마다 한 번 준비해 변형들이 함께 쓴다
        picker = RR.ContextPicker(g, pm) if (args.placement == "context" and ref_donors) else None
        for v in range(args.variants):
            rng = np.random.default_rng([cfg["seed"] + SEED_OFFSET, int(r.sha1[:8], 16), v])
            f = g.astype(np.float32)
            placed, labels = [], [tuple(b) for b in real]
            # 파일 이름 꼬리: paste · mix 는 p, 그 밖은 모드 이름 첫 글자 (ref → r, all3 → a) + 변형 번호
            sid = f"{r.id}__{'p' if args.mode in ('paste', 'mix') else args.mode[0]}{v}"
            for _ in range(rng.integers(A.N_RANGE[0], A.N_RANGE[1] + 1)):
                pool = cand["band"] if (rng.random() < A.BAND_P and len(cand["band"])) else cand["any"]
                # 먼저 넣은 이물과 24px 이상 떨어진 픽셀을 100번까지 뽑는다 (끝내 못 찾으면 마지막에 뽑은 자리를 쓴다)
                for _try in range(100):
                    y, x = pool[rng.integers(len(pool))]
                    if all(np.hypot(x - px, y - py) >= 24 for px, py in placed):
                        break
                if args.mode == "mix":                       # 이전 판과 같은 난수 순서 (data/aug_mix 재현)
                    kind = "구" if rng.random() < 0.5 else "이식"
                else:
                    kind = kinds[rng.integers(len(kinds))] if len(kinds) > 1 else kinds[0]
                if kind == "구":                             # augment.py 와 같은 구·파편 합성
                    cx, cy = x + rng.random(), y + rng.random()
                    c0, d = rng.uniform(*A.C0_RANGE), rng.uniform(*A.D_RANGE)
                    shard = rng.random() < A.SHARD_P
                    aspect = rng.uniform(*A.ASPECT_RANGE) if shard else 1.0
                    angle = rng.uniform(0, np.pi)
                    insert(f, cx, cy, d, c0, aspect, angle)
                    # 박스 크기 계산도 augment.py 와 같다 (모양이 차지하는 범위 + 여유, 최소 MIN_BOX)
                    ex = abs(d * aspect / 2 * np.cos(angle)) + abs(d / 2 * np.sin(angle))
                    ey = abs(d * aspect / 2 * np.sin(angle)) + abs(d / 2 * np.cos(angle))
                    bw, bh = max(A.MIN_BOX, 2 * ex + 6), max(A.MIN_BOX, 2 * ey + 6)
                    info = dict(kind="구", d=d, c0=c0, s=None, donor=None)
                elif kind == "이식":
                    did, T = donors[rng.integers(len(donors))]
                    s = rng.uniform(*S_RANGE)
                    if not paste(f, int(x), int(y), T, s, rng):
                        skipped += 1
                        continue
                    # 조각을 픽셀 (x, y) 한가운데에 놓았으므로 라벨 중심은 +0.5, 박스는 한 변 MIN_BOX 로 고정
                    cx, cy = x + 0.5, y + 0.5
                    bw = bh = A.MIN_BOX
                    info = dict(kind="이식", d=None, c0=None, s=s, donor=did)
                else:                                        # 인접 프레임 기준 이식
                    if not ref_donors:
                        skipped += 1
                        continue
                    dn = ref_donors[rng.integers(len(ref_donors))]
                    s = rng.uniform(*S_RANGE)
                    # 조각과 박스 크기는 떼어 낸 이물의 박스 크기(px) 그대로다. 기본은 뽑은 픽셀이 박스 가운데가 되게 놓는다
                    bw, bh = dn["bw"], dn["bh"]
                    x0, y0 = int(x) - bw // 2, int(y) - bh // 2
                    # 구조 맞춤: 위에서 뽑은 자리 대신, 실제 박스와 먼저 넣은 이물(둘레 12px)을 피한 후보 중 비용이 낮은 TOPK 곳에서 고른다
                    if picker is not None:
                        taken = real_xyxy + [(px - 12, py - 12, px + 12, py + 12) for px, py in placed]
                        cs = picker.candidates(dn["desc"], bw, bh, taken)[:TOPK]
                        if not cs:
                            skipped += 1
                            continue
                        _, x0, y0 = cs[rng.integers(len(cs))]
                    if not RR.apply(f, x0, y0, dn["patch"], s):
                        skipped += 1
                        continue
                    cx, cy = x0 + bw / 2, y0 + bh / 2
                    # 실제로 놓인 자리로 x, y 를 다시 잡는다 (아래 거리 검사 목록과 띠 안 여부에 쓴다)
                    x, y = int(cx), int(cy)
                    info = dict(kind="기준이식", d=None, c0=None, s=s, donor=dn["id"])
                labels.append((0, cx / w, cy / h, bw / w, bh / h))
                placed.append((x, y))
                rows.append(dict(img=sid, src=r.id, machine=r.machine, cx=cx, cy=cy, in_band=bool(band[int(y), int(x)]), **info))
            Image.fromarray(np.clip(f.round(), 0, 255).astype(np.uint8)).save(out / "images" / f"{sid}.png")
            np.savetxt(out / "labels" / f"{sid}.txt", np.array(labels), fmt="%d %.6f %.6f %.6f %.6f")
            paths.append(str((out / "images" / f"{sid}.png").resolve()))

    orig = (data / "clean" / "train.txt").read_text(encoding="utf-8").split()
    (out / "train.txt").write_text("\n".join(orig + paths) + "\n", encoding="utf-8")
    ds = {"path": str((data / "clean").resolve()), "train": str((out / "train.txt").resolve()),
          "val": "val.txt", "test": "test.txt", "names": {0: "Defect"}}
    yaml.safe_dump(ds, open(data / f"{tag}.yaml", "w", encoding="utf-8"), allow_unicode=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "defects.csv", index=False, encoding="utf-8-sig")
    if refs:
        pd.DataFrame(refs).to_csv(out / "references.csv", index=False, encoding="utf-8-sig")
    json.dump(dict(mode=args.mode, placement=args.placement, variants=args.variants, n_range=A.N_RANGE, s_range=S_RANGE,
                   band_p=A.BAND_P, bank={str(k): len(v) for k, v in bank.items()},
                   ref_bank={str(k): len(v) for k, v in ref_bank.items()}, n_references=len(refs),
                   n_images=len(paths), n_defects=len(rows), n_skipped=skipped,
                   kinds=df["kind"].value_counts().to_dict(), n_train_total=len(orig) + len(paths)),
              open(out / "config.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(len(orig), "+", len(paths), "장, 넣은 이물", df["kind"].value_counts().to_dict(), "건너뜀", skipped,
          "이식 은행", {k: len(v) for k, v in bank.items()}, "기준 은행", {k: len(v) for k, v in ref_bank.items()})


if __name__ == "__main__":
    main()
