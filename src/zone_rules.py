"""구역별 기준선(A)과 놓침 후보 표시(B): 놓침 위험 지도의 고위험 구역을 판정 규칙에 쓴다. 모델은 다시 학습하지 않는다.

고위험 구역 = miss_risk.fit_risk 의 위험 지도(기준 이물 지름 2px · 대비 0.15)에서 val 제품 화소 상위 20% 이상인 자리.
박스마다 그 사진 자체로 계산한 위험 지도에서 고위험 구역 안인지 붙인다(현장과 같은 조건).

A. 구역별 기준선: 고위험 구역 안 박스는 t_zone(< 합격선), 밖 박스는 합격선과 비교한다.
   사진 점수 s' = max(밖 박스 점수, 안 박스 점수 × 합격선 / t_zone) 가 합격선 미만이면 합격, 원래 최고 점수가 불합격선 이상이면 불합격.
   선택 규칙(미리 고정): val 가짜 정상 합격률이 지금보다 1장(77장 중 1.3%p) 넘게 떨어지지 않는 범위에서
   val 시험편 검출률이 가장 높은 t_zone. 같으면 높은(덜 공격적인) 쪽. 이득이 없으면 채택 안 함.
B. 놓침 후보 표시 (판정은 그대로): 점수가 t_cand 이상 · 합격선 미만인 박스가 고위험 구역 안이면 '놓쳤을 수 있는 후보'.
   선택 규칙(미리 고정): val 가짜 정상 중 후보가 뜨는 사진이 10% 이하인 가장 낮은 t_cand.
   비교용으로 구역 제한 없이(제품 어디든) 같은 규칙으로 고른 결과도 낸다 (위험 지도가 보태는 몫).
선택은 val 표만 보고, test 는 고른 값으로 한 번 잰다.
주의: 가짜 정상에는 이물을 지운 흔적이 고위험 구역(띠 끝)에 남아 있어, A 의 정상 합격률 하락과 B 의 괜한 표시가
      실제 생산 사진보다 크게 나올 수 있다. 시험편 사진은 가짜 정상을 약 49번씩 배경으로 다시 쓰므로 헛경보는 고유 자리로 센다.

실행: .venv\\Scripts\\python.exe src\\zone_rules.py
결과: results/zone_rules/ (summary.json, curve_A.csv, curve_B.csv, examples.png, preds_*.csv)
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from conditions import Img
from judge import in_spec
from miss_risk import REF_C, REF_D, design, fit_risk
from synth_eval import MARGIN, yolo_preds
from testpiece import HALF

ROOT = Path(__file__).resolve().parents[1]
MIN_SCORE = 0.05                     # 이보다 점수가 낮은 박스는 버린다 (t_cand 후보의 최솟값과 같음)
NEAR = HALF + MARGIN                 # ±7px (사양표 채점과 같음)
NORMAL_DROP = 1                      # A: val 가짜 정상 합격 감소 허용 (장)
FLAG_MAX = 0.10                      # B: val 가짜 정상 괜한 표시 허용 비율
T_ZONE_GRID = np.round(np.arange(0.30, 0.6128, 0.02), 3)      # A: 시험할 t_zone 후보 (0.30 부터 0.02 간격, 0.6128 미만)
T_CAND_GRID = np.round(np.arange(0.05, 0.61, 0.05), 3)        # B: 시험할 t_cand 후보 (0.05 부터 0.05 간격, 0.60 까지)


def zone_of(model, high, preds, path_of, machine_of):
    """박스마다 그 사진의 위험 지도에서 고위험 구역 안인가. 사진 지도는 한 장씩 만들고 버린다(메모리).

    model: 놓침 모형, high: 고위험 기준값, preds: 열 img, px, py 인 예측 표,
    path_of · machine_of: {사진 이름: 경로} · {사진 이름: 호기}. 반환: preds 행 순서의 bool 배열.
    """
    zone = np.zeros(len(preds), bool)
    for img, idx in preds.groupby("img").groups.items():
        path = path_of[img]
        I = Img.get(path)
        h, w = I["g"].shape
        q = preds.loc[idx]
        # 박스 중심을 가장 가까운 화소로 반올림하고 영상 안으로 당긴다
        x = np.clip(q.px.round().astype(int).to_numpy(), 0, w - 1)
        y = np.clip(q.py.round().astype(int).to_numpy(), 0, h - 1)
        # 지도 전체를 계산하지 않고 박스 자리의 값만 뽑아 기준 이물의 놓침 확률을 구한다 (지도는 [y, x] 순서)
        X = design(REF_C, REF_D, I["dist"][y, x], I["band"][y, x], I["bg"][y, x], I["tex"][y, x], machine_of[img])
        r = model.p(X)
        r[I["pm"][y, x] == 0] = 0.0          # 제품 밖은 위험 0 으로 두어 고위험 구역에서 뺀다
        zone[preds.index.get_indexer(idx)] = r >= high
        Img.cache.pop(path, None)
    return zone


def point_zone(model, high, path, machine, xs, ys):
    """사진 한 장에서 좌표 (xs, ys) 들이 고위험 구역 안인가. zone_of 와 같은 계산을 좌표 배열에 바로 한다.

    path: 사진 경로, machine: 호기, xs · ys: 픽셀 좌표 배열. 반환: 같은 길이의 bool 배열.
    """
    I = Img.get(path)
    h, w = I["g"].shape
    x, y = np.clip(np.round(xs).astype(int), 0, w - 1), np.clip(np.round(ys).astype(int), 0, h - 1)
    r = model.p(design(REF_C, REF_D, I["dist"][y, x], I["band"][y, x], I["bg"][y, x], I["tex"][y, x], machine))
    r[I["pm"][y, x] == 0] = 0.0
    Img.cache.pop(path, None)
    return r >= high


def image_scores(preds, ids):
    """사진별 (원래 최고 점수, 구역 밖 최고, 구역 안 최고).

    preds: 열 img, score, zone 인 예측 표, ids: 사진 이름 목록. 반환: ids 순서의 배열 셋. 박스가 없으면 0.
    """
    g = preds.groupby("img")
    raw = g.score.max().reindex(ids, fill_value=0.0).to_numpy()
    out = preds[~preds.zone].groupby("img").score.max().reindex(ids, fill_value=0.0).to_numpy()
    inn = preds[preds.zone].groupby("img").score.max().reindex(ids, fill_value=0.0).to_numpy()
    return raw, out, inn


def verdict(raw, out, inn, t_low, t_high, t_zone):
    """구역별 기준선으로 사진을 합격 / 재검사 / 불합격으로 나눈다.

    raw · out · inn: image_scores 의 세 배열, t_low: 합격선, t_high: 불합격선, t_zone: 고위험 구역 안 박스에 쓰는 기준선.
    반환: 사진 순서의 문자열 배열.
    """
    # 구역 안 점수에 t_low / t_zone 을 곱하면 't_zone 이상'이 '합격선 이상'과 같아져, 합격선 하나로 비교할 수 있다.
    # 불합격은 바꾸지 않은 원래 최고 점수로 판단한다
    s2 = np.maximum(out, inn * t_low / t_zone)
    v = np.where(s2 < t_low, "합격", np.where(raw >= t_high, "불합격", "재검사"))
    return v


def defect_hits(preds, defects):
    """시험편마다 ±7px 안 박스의 (구역 밖 최고, 구역 안 최고) 점수.

    preds: 열 img, px, py, score, zone. defects: 시험편 표(열 img, cx, cy). 반환: defects 행 순서의 배열 둘, 없으면 0.
    """
    out = np.zeros(len(defects))
    inn = np.zeros(len(defects))
    by = dict(tuple(preds.groupby("img")))
    for k, r in enumerate(defects[["img", "cx", "cy"]].itertuples(index=False)):
        q = by.get(r.img)
        if q is None:
            continue
        near = (np.abs(q.px.to_numpy() - r.cx) <= NEAR) & (np.abs(q.py.to_numpy() - r.cy) <= NEAR)
        z = q.zone.to_numpy()
        s = q.score.to_numpy()
        if (near & ~z).any():
            out[k] = s[near & ~z].max()
        if (near & z).any():
            inn[k] = s[near & z].max()
    return out, inn


def stray_spots(preds, defects, src_of):
    """시험편과 무관한 박스(±7px 밖)를 (원본 사진, 6px 격자)로 묶은 표.

    src_of: {시험편 사진 이름: 배경으로 쓴 원본 사진 이름}. 반환: preds 에서 남은 행에 '자리' 열을 붙인 표.
    같은 배경을 다시 쓴 사본에서 반복되는 박스는 '자리' 값이 같아진다.
    """
    by = dict(tuple(defects.groupby("img")))
    keep = []
    for r in preds.itertuples():
        g = by.get(r.img)
        if g is not None and ((np.abs(g.cx - r.px) <= NEAR) & (np.abs(g.cy - r.py) <= NEAR)).any():
            continue
        keep.append(r.Index)
    p = preds.loc[keep].copy()
    p["자리"] = p.img.map(src_of) + "@" + (p.px / 6).round().astype(int).astype(str) + "," + (p.py / 6).round().astype(int).astype(str)
    return p


def build_split(split, model, high, args, data, man, out):
    """한 split 의 판정 세트 · 시험편 예측과 구역 표시. 예측은 저장해 두고 다시 쓴다.

    반환: (js, td, ti, res). js = 판정 세트 사진 표(kind: normal / real_ng / synth_ng), td = 시험편 표(zone 열 포함),
    ti = 시험편 사진 표, res = {"판정": 예측 표, "시험편": 예측 표} (열 img, px, py, score, zone).
    """
    js = pd.read_csv(data / f"judge_{split}.csv")
    js["y"] = (js.kind != "normal").astype(int)
    tdir = data / ("testpiece" if split == "test" else "testpiece_val")
    td, ti = pd.read_csv(tdir / "defects.csv"), pd.read_csv(tdir / "images.csv")
    res = {}
    for name, paths, ids, mach in [("판정", js.path.tolist(), js.img.tolist(), js.machine.tolist()),
                                   ("시험편", [tdir / "images" / f"{i}.png" for i in ti.img], ti.img.tolist(), ti.machine.tolist())]:
        f = out / f"preds_{split}_{name}.csv"
        if args.reuse and f.exists():
            p = pd.read_csv(f)
        else:
            p = yolo_preds(args.yolo, paths, ids, machines=mach)
            p = p[p.score >= MIN_SCORE].reset_index(drop=True)
            path_of = dict(zip(ids, map(str, paths)))
            mach_of = dict(zip(ids, mach))
            p["zone"] = zone_of(model, high, p, path_of, mach_of)
            p.to_csv(f, index=False, encoding="utf-8-sig")
        p["zone"] = p["zone"].astype(bool)       # csv 에서 다시 읽은 경우에도 bool 로 맞춘다
        res[name] = p
    # 시험편 자리 자체가 고위험 구역인가 (배경 = 시험편이 들어간 사진, 시험편이 결을 조금 바꾸지만 박스 판정과 같은 조건)
    zf = out / f"defect_zone_{split}.csv"
    if args.reuse and zf.exists():
        td["zone"] = pd.read_csv(zf)["zone"].astype(bool).to_numpy()
    else:
        z = np.zeros(len(td), bool)
        for img, idx in td.groupby("img").groups.items():
            q = td.loc[idx]
            z[td.index.get_indexer(idx)] = point_zone(model, high, str(tdir / "images" / f"{img}.png"), int(q.machine.iloc[0]),
                                                      q.cx.to_numpy(), q.cy.to_numpy())
        td["zone"] = z
        td[["img", "zone"]].to_csv(zf, index=False, encoding="utf-8-sig")
    return js, td, ti, res


def evaluate_A(js, td, ti, res, th, t_zone, spec):
    """구역별 기준선 t_zone 하나를 재 본다. t_zone 이 합격선과 같으면 지금 규칙 그대로다.

    js · td · ti · res: build_split 의 반환값, th: 채택 기준선(합격선 · 불합격선), spec: val 검출 사양표.
    반환: 판정 세트의 정상 합격률 · 불량 놓침 수, 시험편 검출률(전체 · 구역별 · 호기별), 시험편 사진의 헛경보 수를 담은 dict.
    """
    t_low, t_high = th["합격선"], th["불합격선"]
    # 사진 단위: 판정 세트를 새 규칙으로 다시 판정한다
    raw, o, i = image_scores(res["판정"], js.img.tolist())
    v = verdict(raw, o, i, t_low, t_high, t_zone)
    # 사진 종류별 가리개: 가짜 정상 / 실제 불량 / 합성 불량
    n, r_, sn = (js.kind == "normal").to_numpy(), (js.kind == "real_ng").to_numpy(), (js.kind == "synth_ng").to_numpy()
    ins = in_spec(js, spec)
    # 이물 단위: 시험편 근처 박스가 구역 밖이면 합격선, 구역 안이면 t_zone 이상일 때 검출로 본다
    do, di = defect_hits(res["시험편"], td)
    hit = (do >= t_low) | (di >= t_zone)
    # 헛경보: 시험편과 무관한 박스에 같은 구역별 기준선을 적용한다
    stray = stray_spots(res["시험편"], td, dict(zip(ti.img, ti.src)))
    fp = stray[(stray.score >= t_low) | (stray.zone & (stray.score >= t_zone))]
    return {"t_zone": float(t_zone),
            "정상_합격": round(float((v[n] == "합격").mean()), 4), "정상_합격_장": int((v[n] == "합격").sum()),
            "정상_재검사": round(float((v[n] == "재검사").mean()), 4), "정상_불합격": round(float((v[n] == "불합격").mean()), 4),
            "실제불량_놓침": int((v[r_] == "합격").sum()),
            "사양안_합성_놓침": f"{int(((v == '합격') & sn & ins).sum())}/{int((sn & ins).sum())}",
            "사양밖_합성_놓침": f"{int(((v == '합격') & sn & ~ins).sum())}/{int((sn & ~ins).sum())}",
            "시험편_검출률": round(float(hit.mean()), 4),
            "시험편_검출률_고위험구역": round(float(hit[td.zone.to_numpy()].mean()), 4),
            "시험편_검출률_구역밖": round(float(hit[~td.zone.to_numpy()].mean()), 4),
            "시험편_검출률_호기": {int(m): round(float(hit[(td.machine == m).to_numpy()].mean()), 4) for m in sorted(td.machine.unique())},
            "시험편사진_헛경보_고유자리": int(fp.자리.nunique()), "시험편사진_헛경보_건수": int(len(fp))}


def evaluate_B(js, td, ti, res, th, t_cand, spec, zone_only=True):
    """놓침 후보 표시 기준 t_cand 하나를 재 본다. zone_only 가 False 면 구역 제한 없이 같은 점수 범위의 박스를 모두 후보로 본다.

    반환: 합격한 가짜 정상 중 후보가 뜬 사진 수와 비율(괜한 표시), 놓친 시험편 중 후보로 잡힌 수와 비율(포착률),
    시험편 사진의 무관한 후보 자리 수 등을 담은 dict.
    """
    t_low = th["합격선"]
    pj = res["판정"]
    # 판정은 그대로 두므로 원래 규칙(최고 점수 < 합격선)으로 합격한 사진을 먼저 가린다
    raw = pj.groupby("img").score.max().reindex(js.img, fill_value=0.0)
    passed = set(js.img[(raw < t_low).to_numpy()])
    # 후보 박스: 점수가 t_cand 이상 합격선 미만, (구역 제한이면) 고위험 구역 안, 합격한 사진의 박스
    cand = pj[(pj.score >= t_cand) & (pj.score < t_low) & (pj.zone if zone_only else True) & pj.img.isin(passed)]
    normals = js[js.kind == "normal"]
    npass = normals[normals.img.isin(passed)]
    flagged = npass.img.isin(set(cand.img))
    # 놓친 시험편 자리에 후보 박스가 뜨는가 (사진 판정과 무관하게 이물 단위로)
    pt = res["시험편"]
    do, di = defect_hits(pt, td)
    miss = np.maximum(do, di) < t_low                 # 근처 박스의 최고 점수가 합격선 아래인 시험편
    pc = pt[(pt.score >= t_cand) & (pt.score < t_low) & (pt.zone if zone_only else True)]
    co, ci = defect_hits(pc, td)
    caught = miss & (np.maximum(co, ci) > 0)          # 놓쳤지만 근처에 후보 박스가 하나라도 있는 시험편
    ins = in_spec(td.assign(kind="synth_ng"), spec)
    z = td.zone.to_numpy()
    stray = stray_spots(pc, td, dict(zip(ti.img, ti.src)))
    return {"t_cand": float(t_cand), "구역제한": zone_only,
            "가짜정상_합격사진": int(len(npass)), "괜한표시_사진": int(flagged.sum()),
            "괜한표시_비율": round(float(flagged.mean()), 4) if len(npass) else None,
            "놓친_시험편": int(miss.sum()), "후보로_잡힘": int(caught.sum()),
            "놓침_포착률": round(float(caught.sum() / max(miss.sum(), 1)), 4),
            "놓침_포착률_사양안": round(float((caught & ins).sum() / max((miss & ins).sum(), 1)), 4),
            "놓침_포착률_고위험구역": round(float((caught & z).sum() / max((miss & z).sum(), 1)), 4),
            "시험편사진_무관한후보_고유자리": int(stray.자리.nunique()),
            "시험편사진_한장당_후보": round(float(len(pc) / len(ti)), 3)}


def main():
    """val 에서 t_zone 과 t_cand 를 고르고, 고른 값으로 test 를 한 번 재어 results/zone_rules 에 저장한다."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo", default="ratio3_e100")
    ap.add_argument("--reuse", action="store_true", help="저장된 예측 · 구역 표시를 다시 씀")
    args = ap.parse_args()
    data = ROOT / "data"
    out = ROOT / "results" / "zone_rules"
    out.mkdir(parents=True, exist_ok=True)
    th = json.load(open(ROOT / "results/risk_threshold/summary.json", encoding="utf-8"))["채택"]
    man = pd.read_csv(data / "manifest.csv").set_index("id")
    model, _, high, _ = fit_risk(data, man, th["합격선"], args.yolo)
    spec = pd.read_csv(ROOT / f"results/testpiece_val_{args.yolo}/spec.csv")
    spec = spec[spec["model"] == "YOLO"]
    summary = {"합격선": th["합격선"], "불합격선": th["불합격선"], "고위험_기준값": round(high, 4)}

    S = {s: build_split(s, model, high, args, data, man, out) for s in ["val", "test"]}

    # A: val 로 고르기
    # 후보 끝에 합격선 자체를 넣어 '지금 규칙'(base)도 같은 표에서 함께 잰다
    ca = pd.DataFrame([evaluate_A(*S["val"], th, t, spec) for t in list(T_ZONE_GRID) + [th["합격선"]]])
    ca.to_csv(out / "curve_A.csv", index=False, encoding="utf-8-sig")
    base = ca[np.isclose(ca.t_zone, th["합격선"])].iloc[0]
    ok = ca[ca.정상_합격_장 >= base.정상_합격_장 - NORMAL_DROP]
    # 검출률이 가장 높은 것, 같으면 t_zone 이 높은(지금 규칙에 가까운) 쪽
    best = ok.sort_values(["시험편_검출률", "t_zone"], ascending=[False, False]).iloc[0]
    # 지금 규칙보다 검출률이 높고, 고른 값이 합격선 자체가 아닐 때만 이득이 있다고 적는다
    adopt = bool(best.시험편_검출률 > base.시험편_검출률 and not np.isclose(best.t_zone, th["합격선"]))
    summary["A_구역별기준선"] = {"선택규칙": f"val 가짜 정상 합격 감소 {NORMAL_DROP}장 이내에서 val 시험편 검출률 최대",
                            "선택_t_zone": float(best.t_zone), "이득있음": adopt,
                            "val_기존": base.to_dict(), "val_선택": best.to_dict(),
                            "test_기존": evaluate_A(*S["test"], th, th["합격선"], spec),
                            "test_선택": evaluate_A(*S["test"], th, float(best.t_zone), spec)}
    print("A", json.dumps({k: summary["A_구역별기준선"][k] for k in ["선택_t_zone", "이득있음"]}, ensure_ascii=False))
    for k in ["test_기존", "test_선택"]:
        print(" ", k, json.dumps(summary["A_구역별기준선"][k], ensure_ascii=False))

    # B: val 로 고르기 (구역 제한 / 제한 없음)
    for zone_only, key in [(True, "B_놓침후보_고위험구역"), (False, "B_비교_구역제한없음")]:
        cb = pd.DataFrame([evaluate_B(*S["val"], th, t, spec, zone_only) for t in T_CAND_GRID])
        cb.to_csv(out / f"curve_B{'' if zone_only else '_nozone'}.csv", index=False, encoding="utf-8-sig")
        okb = cb[cb.괜한표시_비율 <= FLAG_MAX]
        tc = float(okb.t_cand.min()) if len(okb) else None
        summary[key] = {"선택규칙": f"val 가짜 정상 괜한 표시 {FLAG_MAX:.0%} 이하인 가장 낮은 t_cand", "선택_t_cand": tc,
                        "val_선택": okb.sort_values("t_cand").iloc[0].to_dict() if len(okb) else None,
                        "test_선택": evaluate_B(*S["test"], th, tc, spec, zone_only) if tc is not None else None}
        print(key, tc, json.dumps(summary[key]["test_선택"], ensure_ascii=False))

    # 동작 확인: t_cand = 합격선이면 후보 0
    chk = evaluate_B(*S["test"], th, th["합격선"], spec, True)
    summary["동작확인_t_cand=합격선_후보"] = chk["후보로_잡힘"] + chk["괜한표시_사진"]

    # 예시: test 에서 후보로 잡힌 놓친 시험편 4개 (64px → 256px, 청록 고리 = 후보 박스, 주황 점선 없음)
    tc = summary["B_놓침후보_고위험구역"]["선택_t_cand"]
    if tc is not None:
        js, td, ti, res = S["test"]
        pt = res["시험편"]
        do, di = defect_hits(pt, td)
        pc = pt[(pt.score >= tc) & (pt.score < th["합격선"]) & pt.zone]
        co, ci = defect_hits(pc, td)
        idx = np.where((np.maximum(do, di) < th["합격선"]) & (np.maximum(co, ci) > 0))[0]
        rng = np.random.default_rng(0)
        tiles = []
        for k in rng.choice(idx, size=min(4, len(idx)), replace=False):
            r = td.iloc[k]
            g = np.asarray(Image.open(data / "testpiece/images" / f"{r.img}.png").convert("L"))
            g = np.pad(g, 40, mode="edge")           # 가장자리 근처 시험편도 64px 조각을 자를 수 있게 40px 씩 덧댄다
            x, y = int(round(r.cx)) + 40, int(round(r.cy)) + 40
            c = cv2.cvtColor(cv2.resize(g[y - 32:y + 32, x - 32:x + 32], (256, 256), interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2RGB)
            cv2.circle(c, (128, 128), 26, (15, 118, 110), 2, cv2.LINE_AA)
            tiles += [c, np.full((256, 10, 3), 255, np.uint8)]
            summary.setdefault("예시", []).append({"img": r.img, "호기": int(r.machine), "지름": float(r.d), "대비": round(float(r.c_meas), 3)})
        if tiles:
            Image.fromarray(np.hstack(tiles[:-1])).save(out / "examples.png")
    json.dump(summary, open(out / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=str)


if __name__ == "__main__":
    main()
