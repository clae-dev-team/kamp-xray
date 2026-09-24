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

> 작성 예정: 전처리 → 학습 → 추론 → 결과 생성을 한 번에 실행하는 스크립트

## 폴더 구성

```
src/         파이프라인 코드
notebooks/   분석 노트북
data/        전처리 결과 (저장소 제외)
results/     예측 결과·그래프 (저장소 제외)
```
