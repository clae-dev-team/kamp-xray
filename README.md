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

## 데이터

KAMP에서 받은 X-ray 검사장비 AI 데이터셋을 사용합니다. 원본 경로는 `configs/data.yaml`의 `raw_root`, `label_dir`에서 지정합니다.
원본 이미지 일부에 포함된 색상 사각형 표시는 전처리 단계에서 제거하며, 방법과 영향은 결과보고서에 기술합니다.

## 한 번에 실행

```bash
python run_all.py                    # 전처리 → 베이스라인 → 합성 증강 → 최종 모델 학습 → 판정 → 제출 파일 (GPU 약 1시간)
python run_all.py --skip-train       # 학습 없이 runs/ 의 가중치로 나머지 전부 다시 생성 (약 15분)
python run_all.py --all-experiments  # 보고서의 비교 모델까지 모두 다시 학습 (약 3~4시간)
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
| predict | `src/predict.py` | 최종 모델로 test 예측, 제출 파일 | `results/submission` |
| synth_eval | `src/synth_eval.py` | 합성 이물 대비·크기별 검출률 | `results/synth_eval_<이름>` |
| location | `src/location_test.py` | 실제 이물 자리에서 점 지움·교체 실험 (위치 의존 검증) | `results/location_test_<이름>` |
| spec_val | `src/testpiece.py --split val` | 판정 기준선용 검출 사양을 val 가짜 정상으로 산출 | `results/testpiece_val_<이름>` |
| testpiece | `src/testpiece.py` | 가상 테스트피스: 가짜 정상 사진에 크기 4종 × 진하기 8종 시험편을 칸당 150개씩 넣어 호기별 검출 사양(90% 보장 진하기) 산출 | `data/testpiece`, `results/testpiece` |
| ensemble | `src/ensemble.py` | YOLO + CNN 뒤집기 TTA 앙상블 판정과 흔들림 기반 재검사 비교 (val에서만 선택) | `results/ensemble` |
| golden | `src/golden_set.py` | 호기별로 가장 깨끗한 val 가짜 정상 5장 (운영 점검 기준 사진) | `results/golden` |
| monitor | `src/monitor.py` | 운영 중 상시 점검 모의 시연: 생산 흐름에 시험편을 섞고, 가정한 장비 열화에서 경보 시점 확인. `--golden`은 골든 사진 위 시험편 + 잡음 표류 지표 | `results/monitor`, `results/monitor_golden` |
| shortcut | `src/shortcut_test.py` | 색 표시 지름길 검증 (`--all-experiments` 때) | `results/shortcut` |
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
