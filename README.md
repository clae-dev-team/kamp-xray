# X-ray 영상 기반 이물질 탐지 및 미탐지 조건 분석

제6회 K-인공지능 제조데이터 분석 경진대회 과제 ④ 출품 코드입니다.

## 환경

- Python 3.12, CUDA 12.8 GPU (RTX 5060 Laptop 8GB에서 확인)
- 설치

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.lock.txt`에 실험에 쓴 전체 패키지 버전이 기록되어 있습니다.

## 추론 속도 (선택, 2026-10-04)

같은 최종 모델을 여러 실행 형식으로 돌려 한 장당 시간을 재고, 시험 판정 세트 438장의 판정이 바뀌지 않는지 확인합니다(`src/speed.py`, 결과 `results/speed`). 추가 패키지가 필요해 `run_all.py`에는 넣지 않았습니다.

```bash
pip install onnx onnxslim onnxruntime-gpu==1.22.0 tensorrt-cu12   # 1.23 이상 기본판은 CUDA 13 용이라 GPU를 못 잡음
python src/speed.py --yolo ratio3_e100
```

| 형식 (RTX 5060 Laptop, 한 장씩) | 한 장 전체 | 그중 추론 | 초당 | 판정 (기준과 같음) |
|---|---:|---:|---:|---:|
| PyTorch FP32 GPU (기준) | 21.9 ms | 14.2 ms | 46장 | - |
| ONNX GPU | 19.9 ms | 11.6 ms | 50장 | 438/438 |
| **TensorRT FP16 GPU** | **10.0 ms** | **1.4 ms** | **100장** | 438/438 |
| PyTorch CPU | 129.7 ms | 122.0 ms | 8장 | 438/438 |

모든 형식에서 실제 이물 139개를 모두 찾았습니다. FP16 형식은 점수가 최대 0.018 달라져, 보장 기준선의 흔들림 여유(0.0059, FP32 기준)보다 큽니다. FP16으로 운영하려면 그 형식으로 흔들림을 다시 재야 합니다.

## 추가 검증 (2026-10-05)

결론이 어디까지 성립하는지 따로 확인한 실험입니다. `python run_all.py --all-experiments` 의 `extra` 단계에서 함께 돌고, 재표집 구간(`uncertainty`)과 이식 시험편 평가(`paste_eval`)는 기본 실행에도 들어 있습니다.

| 실험 | 코드 | 결과 | 위치 |
|---|---|---|---|
| 수치의 오차 범위 | `src/uncertainty.py` | 시험 73장 재표집 95% 구간: 실제 이물 F1 규칙 기반 0.961~0.997 · CNN 0.933~0.977 · 최종 0.982~1.000. 최종 − 규칙 기반 차이는 0을 포함(−0.008~+0.034)하고, 합성 저대비 검출률 차이는 +26.7%p(24.1~29.2) | `results/uncertainty/summary.json` |
| 시드 반복 | `src/uncertainty.py --seeds` | 시드 0 · 1 · 2 모두 F1 0.993(놓침 0 · 오경보 2), 합성 검출률 55.3~55.6%, 실제 이물 최저 점수 0.635~0.657 | `results/uncertainty/seeds.json` |
| 교차 호기 | `src/cross_machine.py` | 두 호기로 학습 → 뺀 호기 전체로 시험. 합성 3배: 재현율 100 · 99.7 · 99.7%(오경보 10 · 5 · 5). 합성 없음: 3호기를 빼면 재현율 89.4%(놓침 39) | `results/cross_machine/summary.csv` |
| 진짜 점 이식 | `src/augment_paste.py`, `src/paste_eval.py` | 실제 이물 점을 투과율로 오려 옅게 옮겨 붙인 시험편 3,504개. 검출률: 규칙 기반 38.7% · 합성 없음 15.2% · CNN 56.6% · 최종(구 합성) 68.6% · 이식 학습 68.5% · 구 합성 + 이식 71.0% | `results/paste_eval/summary.json` |

## 검사 화면 (선택)

```bash
python src/demo.py     # http://127.0.0.1:8765
```

영상을 고르거나 올리면 3단 판정, 점수와 기준선, 검출 위치별 설명, 놓치기 쉬운 구역을 보여 줍니다. 추가 패키지 없이 표준 라이브러리로 돌고, 모델 · 기준선 · 위험 지도는 제출 결과와 같은 것을 씁니다 (`run_all.py` 를 먼저 실행해 결과 파일이 있어야 합니다).

## 재현성 확인 (2026-10-03)

GitHub에서 새로 받은 저장소와 새 가상환경(`requirements.txt`)으로 `python run_all.py`를 처음부터 다시 돌려, 원래 작업 폴더의 결과와 비교했습니다.

```bash
python src/repro_check.py --a <원래 폴더> --b <다시 돌린 폴더>   # 결과: <다시 돌린 폴더>/results/repro_check/
```

- 데이터: 분할 목록 · 정제 사진 2,532장 · 라벨 500개 · 합성 3,507개 · 가짜 정상 150장 · 시험편 7,204개 파일이 모두 바이트 단위로 같음 (목록 파일의 실행 폴더 경로는 정규화)
- 결과: 최종 YOLO 재학습을 포함해 결과 요약 17개 파일의 숫자 1,158개가 모두 같음 (시험 F1 0.9929, 보장 합격선 0.6128 등)
- 제출 파일: 시험 사진 73장의 판정(재검사 72 · 불합격 1)과 박스 141개가 같음
- 같은 GPU · 드라이버 · 패키지 버전(`requirements.lock.txt`)에서 확인한 결과이며, 다른 GPU에서는 학습 계산 순서 차이로 소수점 아래가 달라질 수 있습니다.

## 데이터

KAMP에서 받은 X-ray 검사장비 AI 데이터셋을 사용합니다. 원본 경로는 `configs/data.yaml`의 `raw_root`, `label_dir`에서 지정합니다.
원본 이미지 일부에 포함된 색상 사각형 표시는 전처리 단계에서 제거하며, 방법과 영향은 결과보고서에 기술합니다.

## 한 번에 실행

```bash
python run_all.py                    # 전처리 → 베이스라인 → 합성 증강 → 최종 모델 학습 → 판정 → 제출 파일 (RTX 5060 Laptop 실측 약 1시간 20분)
python run_all.py --skip-train       # 학습 없이 runs/ 의 가중치로 나머지 전부 다시 생성 (약 15분)
python run_all.py --all-experiments  # 보고서의 비교 모델 · 교차 호기 · 시드 반복까지 모두 다시 학습 (비교 모델 약 3~4시간 + 추가 검증 약 5시간)
python run_all.py --from judge       # 특정 단계부터 이어서
```

최종 모델과 비교 모델 설정은 `configs/pipeline.yaml`에 있습니다. 단계별 로그는 `results/logs/`, 소요 시간은 `results/run_all.json`.
난수 시드를 모두 고정해 전처리·합성 데이터는 몇 번을 실행해도 파일이 똑같이 나옵니다. GPU 학습은 CUDA 연산 특성상 소수점 끝자리가 조금 달라질 수 있습니다.

**테스트 예측 결과 파일**: `results/submission/`
- `test_images.csv` 영상별 최고 점수와 판정(합격 / 재검사 / 불합격)
- `test_boxes.csv` 이물 위치 박스와 점수
- `labels/<id>.txt` YOLO 형식 박스 (class cx cy w h score)
- `thresholds.json` 사용한 모델과 판정 기준선

## 단계

| 단계 | 코드 | 하는 일 | 결과 |
|---|---|---|---|
| prepare | `src/prepare.py` | 중복 277장 제거, 라벨 연결, 색 표시 제거·흔적 균등화, (호기·날짜) 묶음 70/15/15 분할 | `data/clean`, `data/raw`, `results/prepare` |
| baseline | `src/baseline.py` | black top-hat 고전 영상처리. 구조요소·평활은 train AP, 기준선은 val로 결정 | `results/baseline_clean`, `results/baseline_raw` |
| defect_stats | `src/defect_stats.py` | 실제 이물 대비·크기 측정 | `results/defect_stats` |
| synth | `src/synth.py` | test 영상에 Beer–Lambert 곱셈 합성 이물 7,008개 (대비 11 × 지름 5) | `data/synth` |
| augment | `src/augment.py` | train 영상에만 합성 이물을 넣은 증강셋 (30%는 길쭉한 파편) | `data/aug`, `data/aug.yaml` |
| train | `src/train_yolo.py` | YOLO26s 학습과 공용 기준 채점. 에폭 고정 학습은 마지막 에폭(`final.pt`)을 씀 | `runs/<이름>`, `results/yolo_<이름>` |
| cnn | `src/cnn.py` | 32×32 조각 분류 CNN (비교 모델). 패딩 없는 합성곱이라 영상 전체에 한 번에 적용, 어려운 음성 재수집 | `runs/cnn_aug`, `results/cnn_cnn_aug` |
| normal_set | `src/normal_set.py` | val·test 이물 점만 지운 가짜 정상 + 합성 불량 | `data/normal`, `data/synth_ng` |
| judge | `src/judge.py` | 영상 단위 합격/재검사/불합격 기준선과 확률보정(온도 스케일링). 권장 기준선은 "검출 사양 이상 이물은 합격시키지 않는다"(사양 기준) | `results/judge` |
| risk | `src/risk_threshold.py` | 보장 기준선: 점수 흔들림(묶음 1·8·32 재추론) 측정 + Learn-then-Test 방식으로 놓침률 ≤1%·정상 폐기율 ≤5%를 95% 확률로 보장하는 합격·불합격선 (제출 파일 기본값). 보장은 검증 불량 328개(실제 77 + 합성 251) 기준이며 실제 이물만으로는 3.8%까지, 검증과 같은 분포 가정 | `results/risk_threshold` |
| predict | `src/predict.py` | 최종 모델로 test 예측, 제출 파일 | `results/submission` |
| synth_eval | `src/synth_eval.py` | 합성 이물 대비·크기별 검출률 | `results/synth_eval_<이름>` |
| location | `src/location_test.py` | 실제 이물 자리에서 점 지움·교체 실험 (위치 의존 검증) | `results/location_test_<이름>` |
| spec_val | `src/testpiece.py --split val` | 판정 기준선용 검출 사양을 val 가짜 정상으로 산출 | `results/testpiece_val_<이름>` |
| testpiece | `src/testpiece.py` | 가상 테스트피스: 가짜 정상 사진에 크기 4종 × 진하기 8종 시험편을 칸당 150개씩 넣어 호기별 검출 사양(90% 보장 진하기) 산출 | `data/testpiece`, `results/testpiece` |
| froc | `src/froc.py` | FROC(사진 한 장당 헛경보 수별 이물 검출률)과 CPM(LUNA16 방식 7점 평균): 실제 이물 139개 · 가상 시험편 14,400개, 규칙 기반 · CNN · 합성 전/최종 YOLO 비교 | `results/froc` |
| conditions | `src/conditions.py` | 놓침·헛경보 조건 정리: 시험편 놓침을 대비·지름·호기·띠·주변 결로 나눠 보고(로지스틱 회귀·결정나무), 헛경보는 원본 사진의 고유 자리 기준으로 지운 자리·가장자리·띠와 비교 | `results/conditions` |
| miss_risk | `src/miss_risk.py` | 놓침 위험 지도: 검증 시험편으로 놓침 모형(대비·지름·띠·가장자리·주변 결·밝기·호기)을 만들고, 사진마다 "옅은 이물이 있었다면 놓쳤을 구역"을 표시. 시험 시험편으로 검증(같은 세기에서 위치 정보만으로 AUC 0.81, 제품 면적 11%에 놓침 46%), 사진별 리포트 | `results/miss_risk` |
| zone_rules | `src/zone_rules.py` | 위험 지도를 판정 규칙에 쓰는 시험: A 고위험 구역만 기준선 낮추기(시험편 검출률 +0.35%p에 정상 1장 · 헛경보 자리 2→11, 채택 안 함), B 합격 사진의 약한 신호(0.3 이상)를 놓침 후보로 표시(구역 제한 없이 사양 안 놓침의 40% 포착, 정상 사진 13%에 표시. 구역 제한은 이득 없음) | `results/zone_rules` |
| ensemble | `src/ensemble.py` | YOLO + CNN 뒤집기 TTA 앙상블 판정과 흔들림 기반 재검사 비교 (val에서만 선택) | `results/ensemble` |
| reference | `src/reference_set.py` | 기준 정상 영상: 호기별로 가장 깨끗한 val 가짜 정상 5장 (운영 점검 기준 사진, 골든 샘플 역할이나 실물 양품은 아님) | `results/reference` |
| monitor | `src/monitor.py` | 운영 중 상시 점검 모의 시연: 생산 흐름에 시험편을 섞고, 가정한 장비 열화에서 경보 시점 확인. `--reference`는 기준 정상 영상 위 시험편 + 잡음 표류 지표 | `results/monitor`, `results/monitor_reference` |
| cusum | `src/monitor_cusum.py` | 상시 점검 경보 규칙 비교: 최근 20개 창 vs 베르누이 CUSUM (평상시 오경보 간격을 같게 맞춤) | `results/monitor_cusum` |
| diagnose | `src/diagnose.py` | 장비 고장 vs AI 고장 원인 분리: 기준 정상 영상 고정 자리 시험편의 CNR·영상 잡음(영상별 기준값) 감시 (`--all-experiments` 때, AI 고장 역할에 합성 0배 모델 사용) | `results/diagnose` |
| realism | `src/realism.py` | 시험편 현실성: 같은 자리 실제 이물 vs 합성(구·칸 정렬 네모) 조각의 구분력 AUC | `results/realism` |
| shortcut | `src/shortcut_test.py` | 색 표시 지름길 검증 (`--all-experiments` 때) | `results/shortcut` |
| bait_gray | `src/bait_gray.py` | 미끼 원인 가리기: 같은 자리 미끼 네모를 색 · 같은 밝기 회색 · 어두운 회색(×0.6) · 밝은 회색(×1.3)으로 바꿔 반응률 비교 (`--all-experiments` 때) | `results/bait_gray` |
| gradcam | `src/gradcam.py` | HiResCAM으로 원본 학습·정제본 학습 모델의 판단 근거 위치 비교 (`--all-experiments` 때) | `results/gradcam` |

평가 기준은 `src/metrics.py` 하나로 모든 모델에 똑같이 적용합니다 (박스 중심 일치 기준 + IoU 0.5 기준).
판정 기준선은 모두 val에서 정하고, test는 마지막에 한 번만 채점합니다.

## 폴더 구성

```
run_all.py   전체 실행
configs/     경로·전처리·모델 설정
src/         단계별 코드
data/        전처리·합성 결과 (저장소 제외)
runs/        학습 기록·가중치 (저장소 제외)
results/     지표·그래프·제출 파일 (저장소 제외)
```
