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

KAMP에서 받은 X-ray 검사장비 AI 데이터셋을 사용합니다. 원본 경로는 설정 파일에서 지정합니다.
원본 이미지 일부에 포함된 색상 사각형 표시는 전처리 단계에서 제거하며, 방법과 영향은 결과보고서에 기술합니다.

## 실행

### 1. 전처리

원본 경로는 `configs/data.yaml`의 `raw_root`, `label_dir`에서 지정합니다.

```bash
python src/prepare.py --config configs/data.yaml
```

- 폴더 간 완전 중복 영상 제거 (SHA-1)
- TXT 라벨을 파일명으로 원본 BMP와 연결
- 색상 사각형 표시 제거: R·G·B가 다른 픽셀을 표시로 보고, 선에 수직인 방향으로 직선 보간한 뒤 바로 옆 띠의 잡음 결을 옮겨 붙임
- 흔적 균등화: 이물이 없는 제품 영역에도 같은 크기의 가짜 사각형을 그렸다가 똑같이 복원
- 같은 호기·같은 날짜 영상을 묶어 train/val/test 70/15/15 분할
- 결과: `data/clean`(정제본), `data/raw`(표시가 남은 원본, 비교 실험용), `data/*.yaml`, `results/prepare/`(요약·흔적 검증·전후 비교 그림)

같은 설정이면 몇 번을 실행해도 결과 파일이 동일합니다.

### 2. 베이스라인 (고전 영상처리)

```bash
python src/baseline.py            # 정제본
python src/baseline.py --variant raw
```

black top-hat으로 주변보다 어두운 작은 점을 찾습니다. 구조요소·평활·점수 방식은 train AP로, 판정 임계값은 val로 정하고
test는 마지막에 한 번만 채점합니다. 결과는 `results/baseline_<variant>/`(격자 탐색표, 분할별 예측, 지표, PR 곡선).
평가 기준은 `src/metrics.py` 하나로 모든 모델에 똑같이 적용합니다 (박스 중심 일치 기준 + IoU 0.5 기준).

### 3. YOLO

```bash
python src/train_yolo.py --name y26s_640                 # YOLO26s, 입력 640, 150에폭(조기 종료 50)
python src/train_yolo.py --name y26s_640 --skip-train    # 저장된 가중치로 채점만
```

학습 기록·가중치는 `runs/<name>/`, 공용 기준 채점 결과는 `results/yolo_<name>/`에 저장됩니다.

## 폴더 구성

```
configs/     경로·전처리 설정
src/         파이프라인 코드
notebooks/   분석 노트북
data/        전처리 결과 (저장소 제외)
results/     예측 결과·그래프 (저장소 제외)
```
