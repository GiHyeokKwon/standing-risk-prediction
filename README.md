# Standing Risk Prediction

**Design of a QR-Based Standing Passenger Risk Notification Service Using AI Models
for Transportation-Vulnerable Individuals**

KT · Korea Road Traffic Authority · Korea (May. 2026 ~ Aug. 2026)

교통약자를 위한 QR 기반 입석 위험 안내 시스템의 AI 파트. 정류장에서 QR을 찍으면
출발/도착 정류장을 기준으로 **입석 여부**와 **입석 예상 시간**을 예측해,
승차 전에 입석 위험도(낮음/보통/높음)를 미리 안내한다.

## 모델 개요

두 개의 LightGBM 모델을 순차적으로 사용하는 2단계 구조.

- **모델 A (분류)**: 승차 시점에 입석하는지(Y/N) 예측
- **모델 B (회귀)**: 모델 A가 Y로 판단한 경우에 한해, 착석까지 걸리는 시간(초) 예측

정답 레이블(`is_standing`, `standing_seconds`)은 교통카드 원시 데이터(TCD)에서
재차인원을 복원하고 FIFO 좌석배정을 시뮬레이션해 산출했다.

**입력 피처 (8개)**: `route_id`, `board_stop_id`, `alight_stop_id`, `weekday`,
`weather`, `hour`, `is_holiday`, `headway_sec`

## 최종 모델 성능

성북구 2024년 1년치(약 3.3억 행) 데이터, 시간분할 검증(1~11월 학습 → 12월 검증) 기준.

| | 모델 A (AUC) | 모델 A (Accuracy) | 모델 B (MAE) | 모델 B (RMSE) |
| --- | --- | --- | --- | --- |
| baseline (num_leaves=31) | 0.9141 | 88.12% | 158.3초 | 243.1초 |
| num_leaves=63 | 0.9167 | 88.24% | 155.3초 | 240.8초 |
| **num_leaves=127 (최종 채택)** | **0.9180** | **88.29%** | **153.5초** | **239.5초** |

배포용 최종 모델은 검증에 썼던 12월 데이터까지 포함해 1~12월 전체로 재학습
(`n_estimators=3000`, `num_leaves=127`).

**피처 중요도**: 승·하차 정류장 조합이 두 모델 모두에서 가장 큰 영향을 미치는 변수로 확인.
어느 구간을 타는지가 혼잡도를 가장 직접적으로 결정한다는 뜻으로 해석.

## 실험 과정

1000만 단위 이하 규모(서초구, 1개월치)로 시작해, 실제 배포 규모(성북구, 1년치, 3억 행
이상)까지 점진적으로 확장하며 진행.

- **트리 수 증가 (300 → 5000)**: 개선폭이 점차 줄어드는 수확체감 확인 → 다른 하이퍼파라미터로 전환
- **num_leaves 확장 (31 → 63 → 127)**: 지금까지 시도한 파라미터 중 가장 크고 일관된 개선폭 확인
- **검증 방식 전환**: 랜덤 분할이 실제보다 낙관적인 성능을 보인다는 것을 확인하고, 시간분할(과거 → 미래) 검증으로 전환
- **데이터 규모 확장 시 메모리 최적화**: 로컬 환경(16GB RAM)에서 3억 행 처리를 위해 스트리밍 병합, 범주형 조기 인코딩, train/test 분리 저장 등 적용. 이후 클라우드 GPU(RTX A5000, 64코어/240GB RAM)로 이전해 전체 학습 진행

## 백엔드 연동

학습은 Python(LightGBM), 배포는 Java(Spring) 환경이라 **PMML**로 모델을 변환해 전달.

1. LightGBM 모델(`.txt`)을 [jpmml-lightgbm](https://github.com/jpmml/jpmml-lightgbm)으로 PMML 변환
2. 범주형 피처(정류장ID, 노선ID 등)의 정수 코드 매핑표를 모델 파일에서 직접 추출해 함께 전달
3. **Golden test 샘플** 20개(일반/희귀/경계값 케이스)를 만들어, Python 원본 예측값과 PMML(Java) 예측값이 일치하는지 백엔드와 교차 검증

## 구조

```text
.
├── docs/                                   # 회의록, 데이터 스펙 문서
├── schema/                                 # 예시 스키마 (실제 데이터 아님)
├── src/
│   ├── merge_data.py                       # 일별 원시 데이터를 하나의 parquet으로 병합
│   ├── train.py                            # 검증용 학습 (시간분할, 하이퍼파라미터 실험)
│   └── train_deploy.py                     # 배포용 최종 학습 (전체 기간, 검증 없이 재학습)
├── experiments/
│   └── {실행시각}/
│       ├── train_log.txt
│       ├── metrics.json
│       ├── model_a.pmml / model_b.pmml
│       └── category_code_mapping_model_a.json / _model_b.json
├── .gitignore
└── requirements.txt
```

## 실행

```bash
pip install -r requirements.txt

# 1. 일별 원시 데이터 병합
python src/merge_data.py

# 2. 하이퍼파라미터 실험 (시간분할 검증)
python src/train.py

# 3. 배포용 최종 학습 (검증 없이 전체 기간 학습)
python src/train_deploy.py
```

  메모리 최적화(스트리밍 처리, 청크 단위 로딩)가 함께 필요했다.
- 프레임워크 간(Python ↔ Java) 모델 이관에서는 변환 자체보다 **검증**이 핵심이었다 —
  golden test처럼 결과를 대조할 수 있는 절차를 마련해두는 것이 실제 배포 신뢰성을 좌우했다.
