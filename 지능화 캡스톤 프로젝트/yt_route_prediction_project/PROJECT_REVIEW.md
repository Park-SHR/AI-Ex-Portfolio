# 최종본 수행 가능성 검토

## 결론

발표자료의 프로젝트는 개발 검증이 가능합니다. 다만 현재 TOS 데이터가 정적 Bitt/선석 좌표 중심이므로, 1차 개발 목표는 실제 항만 성능 산출이 아니라 다음 범위로 잡는 것이 타당합니다.

- TOS 좌표를 Anchor로 정리
- Anchor 기반 도로망 Node/Link Master 생성
- SOP 기반 OD 및 작업 이벤트 생성
- Dijkstra/K-shortest를 후보 경로·Baseline으로 사용
- 지연·대기·장비 간섭·수요 변동을 주입한 synthetic trace 생성
- 링크×시간 Feature와 ST-GNN 입력 텐서 생성
- Baseline/ML/ST-GNN 모델 입출력과 평가 흐름 검증

## 가능한 검증

| 검증 항목 | 가능 여부 | 구현 방식 |
|---|---:|---|
| TOS 좌표 정합 | 가능 | JSON의 bitt/berth/x/y를 Anchor Master로 구성 |
| 도로망 그래프 생성 | 가능 | QC/Yard/Gate/교차로 Node Master를 보완 정의 |
| SOP 적용 | 가능 | 양하/선적/Gate/Yard 재배치 Template 생성 |
| Dijkstra 모방 문제 검증 | 가능 | K-shortest는 후보로만 사용, route_rank와 NonShortestRate 평가 |
| 시공간 데이터 생성 | 가능 | synthetic_trace와 link_time_samples 생성 |
| ML 모델 검증 | 가능 | RandomForest/GradientBoosting/MLP로 미래 링크 통행량 예측 |
| ST-GNN 검증 | 가능 | 링크 인접행렬 A와 시간 윈도우 X로 미래 링크 통행량 예측 |
| Meta K-Port 연동 전 포맷 | 가능 | link polyline과 congestion_score를 JSON으로 출력 |

## 현재 데이터만으로 어려운 검증

- 실제 YT 운전자의 경로 선택 정확도
- 실제 ETA 오차
- 실제 혼잡 F1/Heatmap IoU
- 실제 운영 개선 효과

위 항목은 YT ID별 위치 로그, 작업지시, OD, 통과시각, 대기시간 확보 후 가능함.
