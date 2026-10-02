"""재현성 확인: 두 실행 폴더(예: 원래 작업 폴더와 새로 받은 저장소에서 run_all.py 를 다시 돌린 폴더)의 결과를 비교한다.

1. 데이터 단계 (결정적이어야 함): 분할 목록과 전처리 · 합성 · 가짜 정상 · 시험편 파일이 바이트 단위로 같은지 (MD5)
2. 결과 요약 (JSON 의 모든 숫자): 완전히 같은 값 / 상대 차이 1% 안 / 그보다 큰 차이 의 개수와 가장 큰 차이
3. 핵심 수치 표: 시험 성능 · 판정 기준선 · 판정 비율 · 검출 사양 · FROC 등을 나란히
4. 제출 파일: 사진별 판정이 같은지, 박스 수

GPU 학습은 같은 시드여도 병렬 계산 순서 때문에 완전히 같지 않을 수 있다(Pham et al. 2020).
그래서 데이터 단계는 '같음'을, 학습 이후는 '차이가 얼마인지와 결론이 바뀌는지'를 본다.

실행: .venv\\Scripts\\python.exe src\\repro_check.py --a C:\\workspace\\kamp-xray --b C:\\workspace\\kamp-xray-repro
결과: <b>/results/repro_check/ (summary.json, numbers.csv, headline.csv)
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_SETS = {"분할 목록": "data/manifest.csv", "정제 사진": "data/clean/images", "정제 라벨": "data/clean/labels",
             "합성 학습 사진": "data/synth", "증강 학습 목록": "data/aug", "가짜 정상": "data/normal",
             "판정 평가 목록": "data/judge_test.csv", "시험편(test)": "data/testpiece", "시험편(val)": "data/testpiece_val"}
JSONS = ["results/baseline_clean/metrics.json", "results/cnn_cnn_aug/metrics.json", "results/yolo_ratio3_e100/metrics.json",
         "results/judge/summary.json", "results/risk_threshold/summary.json", "results/submission/thresholds.json",
         "results/testpiece/summary.json", "results/synth_eval_ratio3_e100/summary.json",
         "results/location_test_ratio3_e100/summary.json", "results/froc/summary.json", "results/conditions/summary.json",
         "results/ensemble/summary.json", "results/monitor/summary.json", "results/monitor_reference/summary.json",
         "results/monitor_cusum/summary.json", "results/realism/summary.json", "results/normal_set/check.json"]
# (이름, 파일, 키 경로) — 키 경로는 / 로 구분
HEADLINE = [
    ("YOLO 시험 F1 (중심)", "results/yolo_ratio3_e100/metrics.json", "metrics/test/F1최대/center/F1"),
    ("YOLO 시험 놓침 (중심)", "results/yolo_ratio3_e100/metrics.json", "metrics/test/F1최대/center/FN"),
    ("YOLO 시험 헛경보 (중심)", "results/yolo_ratio3_e100/metrics.json", "metrics/test/F1최대/center/FP"),
    ("YOLO 시험 F1 (IoU 0.5)", "results/yolo_ratio3_e100/metrics.json", "metrics/test/F1최대/iou50/F1"),
    ("YOLO 박스 기준선 (val F1 최대)", "results/yolo_ratio3_e100/metrics.json", "thresholds/F1최대"),
    ("CNN 시험 F1 (중심)", "results/cnn_cnn_aug/metrics.json", "metrics/test/F1최대/center/F1"),
    ("규칙 기반 시험 F1 (중심)", "results/baseline_clean/metrics.json", "metrics/test/F1최대/center/F1"),
    ("사양 기준 합격선", "results/judge/summary.json", "ratio3_e100/사양기준/기준선/합격선"),
    ("사양 기준 불합격선", "results/judge/summary.json", "ratio3_e100/사양기준/기준선/불합격선"),
    ("보장 합격선 (채택)", "results/risk_threshold/summary.json", "채택/합격선"),
    ("보장 불합격선 (채택)", "results/risk_threshold/summary.json", "채택/불합격선"),
    ("점수 흔들림 최대", "results/risk_threshold/summary.json", "점수_흔들림/최대"),
    ("시험편 검출률 (YOLO)", "results/testpiece/summary.json", "models/YOLO/검출률_전체"),
    ("합성 평가 검출률 (YOLO)", "results/synth_eval_ratio3_e100/summary.json", "YOLO/검출률_전체"),
    ("시험편 CPM (최종)", "results/froc/summary.json", "시험편/ratio3_e100/CPM"),
    ("실제 CPM (최종)", "results/froc/summary.json", "실제/ratio3_e100/CPM"),
    ("사양 안 시험편 놓침률", "results/conditions/summary.json", "놓침/사양안/놓침률"),
    ("상시 점검 첫 경보 프레임", "results/monitor/summary.json", "첫_경보_프레임"),
    ("합성 현실성 AUC (구, 모양)", "results/realism/summary.json", "실제 vs 맞춘 합성(구)/AUC_모양(11x11)"),
]


def md5_tree(root: Path, base: Path):
    """파일별 MD5. 목록 파일(txt · csv · yaml)은 실행 폴더 경로를 <ROOT> 로 바꿔 비교하고, YOLO 캐시(*.cache)는 뺀다."""
    if not root.exists():
        return None
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file() and p.suffix != ".cache")
    out = {}
    for p in files:
        b = p.read_bytes()
        if p.suffix in (".txt", ".csv", ".yaml"):
            for form in {str(base), str(base).replace("\\", "/")}:
                b = b.replace(form.encode("utf-8"), b"<ROOT>")
        out[str(p.relative_to(root.parent if root.is_file() else root))] = hashlib.md5(b).hexdigest()
    return out


def flat(d, pre=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flat(v, f"{pre}/{k}" if pre else str(k)))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            out.update(flat(v, f"{pre}[{i}]"))
    elif isinstance(d, (int, float)) and not isinstance(d, bool):
        out[pre] = float(d)
    return out


def get_any(d, path):
    """키에 / 가 들어 있을 수 있어, 가능한 분할을 모두 시도한다."""
    parts = path.split("/")

    def rec(node, i):
        if i == len(parts):
            return node
        if not isinstance(node, dict):
            return None
        for j in range(len(parts), i, -1):
            k = "/".join(parts[i:j])
            if k in node:
                r = rec(node[k], j)
                if r is not None:
                    return r
        return None
    return rec(d, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="기준 실행 폴더 (원래 작업 폴더)")
    ap.add_argument("--b", required=True, help="비교 실행 폴더 (새 환경 재실행)")
    args = ap.parse_args()
    A, B = Path(args.a), Path(args.b)
    out = B / "results" / "repro_check"
    out.mkdir(parents=True, exist_ok=True)
    summary = {"기준": str(A), "비교": str(B)}

    # 1. 데이터
    ds = {}
    for name, rel in DATA_SETS.items():
        ha, hb = md5_tree(A / rel, A), md5_tree(B / rel, B)
        if ha is None or hb is None:
            ds[name] = {"상태": "한쪽 없음"}
            continue
        keys = set(ha) | set(hb)
        same = sum(ha.get(k) == hb.get(k) for k in keys)
        ds[name] = {"파일수": len(keys), "같음": same, "다름": len(keys) - same,
                    "다른예": sorted(k for k in keys if ha.get(k) != hb.get(k))[:3]}
    summary["데이터"] = ds
    print(json.dumps(ds, ensure_ascii=False, indent=1))

    # 2. JSON 숫자
    rows, js = [], {}
    for rel in JSONS:
        fa, fb = A / rel, B / rel
        if not (fa.exists() and fb.exists()):
            js[rel] = {"상태": "한쪽 없음"}
            continue
        a, b = flat(json.load(open(fa, encoding="utf-8"))), flat(json.load(open(fb, encoding="utf-8")))
        keys = sorted(set(a) & set(b))
        d = np.array([0.0 if (np.isnan(a[k]) and np.isnan(b[k])) else abs(a[k] - b[k]) for k in keys])   # 둘 다 빈 값이면 같음
        rel_d = np.array([0.0 if x == 0 else x / max(abs(a[k]), abs(b[k]), 1e-12) for k, x in zip(keys, d)])
        for k, x, r in zip(keys, d, rel_d):
            rows.append(dict(파일=rel, 키=k, 기준=a[k], 비교=b[k], 차이=x, 상대차이=r))
        js[rel] = {"숫자수": len(keys), "완전히같음": int((d == 0).sum()), "1%안": int(((d > 0) & (rel_d <= 0.01)).sum()),
                   "1%초과": int((rel_d > 0.01).sum()), "한쪽만": len(set(a) ^ set(b))}
    summary["결과요약"] = js
    pd.DataFrame(rows).to_csv(out / "numbers.csv", index=False, encoding="utf-8-sig")

    # 3. 핵심 수치
    head = []
    for name, rel, key in HEADLINE:
        va = vb = None
        if (A / rel).exists() and (B / rel).exists():
            va = get_any(json.load(open(A / rel, encoding="utf-8")), key)
            vb = get_any(json.load(open(B / rel, encoding="utf-8")), key)
        head.append(dict(항목=name, 기준=va, 재현=vb, 차이=(vb - va) if isinstance(va, (int, float)) and isinstance(vb, (int, float)) else None))
    hd = pd.DataFrame(head)
    hd.to_csv(out / "headline.csv", index=False, encoding="utf-8-sig")
    summary["핵심수치"] = head
    print(hd.to_string())

    # 4. 제출 파일
    sa, sb = A / "results/submission/test_images.csv", B / "results/submission/test_images.csv"
    if sa.exists() and sb.exists():
        a, b = pd.read_csv(sa).set_index("id"), pd.read_csv(sb).set_index("id")
        m = a.join(b, lsuffix="_기준", rsuffix="_재현", how="outer")
        same = int((m["판정_기준"] == m["판정_재현"]).sum())
        summary["제출"] = {"사진수": len(m), "판정같음": same, "판정다름": len(m) - same,
                         "다른사진": m.index[m["판정_기준"] != m["판정_재현"]].tolist(),
                         "판정분포_기준": a["판정"].value_counts().to_dict(), "판정분포_재현": b["판정"].value_counts().to_dict(),
                         "최고점수_최대차이": round(float((m["최고점수_기준"] - m["최고점수_재현"]).abs().max()), 4),
                         "박스수_기준": int(a["박스수"].sum()), "박스수_재현": int(b["박스수"].sum())}
        print(summary["제출"])
    json.dump(summary, open(out / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=str)


if __name__ == "__main__":
    main()
