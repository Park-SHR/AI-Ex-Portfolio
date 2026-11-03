# HJNC Final Modeling v05

요청 범위인 **Graph 최종 Freeze → 모델 학습/비교 → 실제 HOLDOUT 최종평가/Ablation → 논문용 결과표·그림 정리**를 수행한 패키지입니다.

## 핵심 결론
- 모델용 Graph: **1,007 Link / 5,900 directed edge**, HOLDOUT 가중 전이 coverage **98.29%**
- Next-Link: **Graph-Markov-2 Top-1 60.99% / Top-5 95.51%**
- 3-Link Top-3: **MLP+augmentation 47.02%**
- Link 집중도 최종 후보: **LightGBM**
- 실제 HOLDOUT에서 Persistence 대비 LightGBM count MAE 개선: **t+5 21.36%, t+10 25.52%, t+30 26.32%**
- ST-GNN-lite(GCN-GRU)는 비교 수행했으나 현재 데이터에서는 LightGBM보다 일관된 우위 없음

## 폴더
- `graph_freeze/`: v05 Frozen Graph 및 adjacency
- `models/`: 학습 모델 및 설정
- `results/`: 정량평가 CSV
- `figures/`: 논문용 고해상도 그림
- `paper/PAPER_RESULTS_KO.md`: 논문 본문에 바로 옮길 수 있는 결과·해석·표·그림 캡션
- `MODEL_TRAINING_REPORT_KO.md`: 전체 학습/평가 방법과 결과

## 주의
실제 고밀도 위치로그는 사실상 2026-01-08 하루이며, 실제 HOLDOUT은 같은 날짜에서 분리한 20대 YT입니다. 30일 데이터는 synthetic scenario입니다. 따라서 본 결과는 연구모델 검증으로 사용하고 실제 운영 일반화 성능으로 과장하지 않습니다.
