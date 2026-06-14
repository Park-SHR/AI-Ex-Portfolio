# YT 경로·혼잡 예측 알고리즘 검증 리포트
## 1. 최종본 개발 가능성 검토
- 현재 TOS 데이터가 Bitt/선석 정적 좌표 중심인 경우, 실제 YT 궤적 지도학습 성능을 바로 주장할 수는 없습니다.
- 그러나 정적 좌표를 Anchor로 활용해 도로망 그래프, OD 후보, SOP 기반 이벤트, 링크 단위 시계열 텐서를 생성하는 1차 검증은 가능합니다.
- 따라서 본 프로젝트는 Meta K-Port 연동 전 단계에서 SOP 시뮬레이션, Baseline/ML/ST-GNN 모델 입출력, Ablation 검증을 수행하도록 구성되었습니다.

## 2. TOS 데이터 요약
- json_files_or_snapshots: 40
- unique_anchor_points: 71
- total_records: 2840
- x_min: -27.65
- x_max: 1196.0
- y_values: [0.0, 2.5]
- coordinate_changes_by_snapshot: 0

## 3. SOP/경로 선택 검증
- Top1_Dijkstra: 0.3810
- Top3_KShortest: 0.9206
- Top5_KShortest: 1.0000
- NonShortestRate: 0.6190

### Route Choice Model
| model         |   Top1Acc |   Top3Acc | classes      |
|:--------------|----------:|----------:|:-------------|
| RouteChoiceRF |    0.3125 |    0.9375 | [1, 2, 3, 4] |

## 4. 링크 혼잡 예측 모델 비교
| model                    |       MAE |     RMSE |   Congestion_Precision |   Congestion_Recall |   Congestion_F1 |   Heatmap_IoU | device   |   epochs |
|:-------------------------|----------:|---------:|-----------------------:|--------------------:|----------------:|--------------:|:---------|---------:|
| Persistence_CurrentCount | 0.025641  | 0.160128 |              0.0294118 |           0.0294118 |       0.0294118 |     0.0149254 | nan      |      nan |
| RandomForest             | 0.0251193 | 0.114179 |              0         |           0         |       0         |     0         | nan      |      nan |
| GradientBoosting         | 0.025221  | 0.114797 |              0         |           0         |       0         |     0         | nan      |      nan |
| MLPRegressor             | 0.0297885 | 0.115594 |              0         |           0         |       0         |     0         | nan      |      nan |
| SimpleSTGNN              | 0.102029  | 0.198575 |              0         |           0         |       0         |     0         | cpu      |        2 |

## 5. 산출물
- tables/anchor_master.csv: TOS Bitt/선석 Anchor 정리
- tables/node_master.csv: QC/Yard/Gate/교차로 보완 Node Master
- tables/link_master.csv: ST-GNN 입력용 도로 링크 정의
- tables/sop_template.csv: SOP 작업 유형 정의
- tables/synthetic_trace.csv: SOP 기반 가상 YT 주행 이벤트
- tables/link_time_samples.csv: 링크×시간 Feature 및 혼잡 점수
- tables/model_metrics.csv: Baseline/ML/ST-GNN 예측 성능 비교
- plots/*.png: 그래프, 혼잡 Heatmap, 모델 비교 시각화
- unreal_prediction_sample.json: 추후 META K-Port/Unreal 연동용 예측 결과 샘플

## 6. 해석 주의사항
- 현재 결과는 실제 항만 성능 수치가 아니라, 발표자료상의 방법론이 개발·검증 가능한지 확인하는 사전 검증 결과입니다.
- 실제 YT 로그가 확보되면 synthetic_trace를 실측 trace로 교체하고, 동일한 평가 파이프라인으로 Top-K, ETA, DTW, 혼잡 F1을 산출하면 됩니다.
