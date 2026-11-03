# 논문용 결과 정리 — HJNC YT 경로·Link 집중도 예측

## 표 1. 최종 공간 Graph Freeze
| 항목 | 결과 |
|---|---:|
| Link node | 1,007 |
| Directed model edge | 5,900 |
| CAD 근거 Edge | 951 |
| TRAIN 실측 근거 Edge | 5,711 |
| HOLDOUT weighted transition coverage | 98.29% |
| HOLDOUT unique transition coverage | 83.54% |
| HOLDOUT의 Freeze 사용 | 사용하지 않음 |

**논문 문장 예시**  
도면 기반 v03 Lane/Link 구조를 모델링 Graph로 확정하기 위해 CAD에서 분석용 주행이 허용된 전이와 TRAIN 차량에서 2회 이상 반복 관측된 고신뢰 전이를 통합하였다. 그 결과 1,007개 Link와 5,900개 directed transition으로 구성된 모델용 Graph를 구축하였으며, Graph 구축에 사용하지 않은 20대 YT HOLDOUT 전이의 98.29%를 가중 빈도 기준으로 포괄하였다. 이는 주요 반복 운행구간에 대한 공간·전이 구조가 학습용으로 안정화되었음을 의미한다. 다만 본 Graph는 현장 운영 승인용 Routing Graph가 아니라 연구·예측 모델용 Freeze 버전이다.

## 표 2. 실제 HOLDOUT Next-Link 예측
| model               |   top1 |   top3 |   top5 |   mrr5 |
|:--------------------|-------:|-------:|-------:|-------:|
| Graph-Markov-1      |  54.86 |  88.25 |  94.59 |  71.69 |
| Graph-Markov-2      |  60.99 |  90.61 |  95.51 |  75.79 |
| MLP real-only       |  60.5  |  89.26 |  94.81 |  75    |
| MLP + augmentation  |  60.87 |  89.98 |  95.25 |  75.53 |
| LSTM + augmentation |  56.75 |  86.77 |  93.62 |  72.09 |
| GRU + augmentation  |  57.98 |  87.73 |  93.88 |  73    |

**논문 해석**  
최근 두 개 Link의 실측 전이 빈도를 사용하는 Graph-Markov-2가 Top-1 60.99%, Top-3 90.61%, Top-5 95.51%로 가장 높은 즉시 다음 Link 성능을 보였다. 증강 MLP는 Top-1 60.87%, Top-5 95.25%로 근접했으며 real-only MLP보다 모든 Top-K 지표가 향상되었다. 이는 증강이 즉시 다음 Link 정확도를 크게 바꾸지는 않지만 실측 경로 조합의 일반화를 소폭 개선했음을 보여준다.

## 표 3. 3-Link 연속 경로 후보
| model              |   route3_top1 |   route3_top3 |   route3_linkhit_top1 |
|:-------------------|--------------:|--------------:|----------------------:|
| Graph-Markov-2     |         22.16 |         42.78 |                 41.82 |
| MLP + augmentation |         24.71 |         47.02 |                 45.38 |

**논문 해석**  
3-Link 연속 경로에서는 증강 MLP가 Top-3 47.03%로 Graph-Markov-2의 42.78%보다 높았다. 단일 Link 선택에서는 반복 전이 규칙이 강하지만, 여러 Link를 연속적으로 예측할 때는 증강된 경로 조합을 학습한 MLP가 더 유리한 결과를 보였다.

## 표 4. Synthetic held-out scenario에서 Link 집중도 예측
| model       | horizon   |   count_mae |   top10_concentration_f1 |
|:------------|:----------|------------:|-------------------------:|
| Persistence | t+5       |      0.6674 |                   0.5153 |
| Persistence | t+10      |      0.7266 |                   0.4863 |
| Persistence | t+30      |      0.752  |                   0.4644 |
| LightGBM    | t+5       |      0.5272 |                   0.6017 |
| LightGBM    | t+10      |      0.544  |                   0.5893 |
| LightGBM    | t+30      |      0.5547 |                   0.5773 |
| LSTM        | t+5       |      0.6618 |                   0.5837 |
| LSTM        | t+10      |      0.6789 |                   0.5779 |
| LSTM        | t+30      |      0.7036 |                   0.5596 |
| GRU         | t+5       |      0.6537 |                   0.5839 |
| GRU         | t+10      |      0.6733 |                   0.5748 |
| GRU         | t+30      |      0.6995 |                   0.5581 |
| GCN-GRU     | t+5       |      0.6625 |                   0.5863 |
| GCN-GRU     | t+10      |      0.6875 |                   0.576  |
| GCN-GRU     | t+30      |      0.7107 |                   0.5596 |

**논문 해석**  
5분 단위 Link 집중도 예측에서 LightGBM은 단순 지속성(Persistence)보다 모든 horizon에서 낮은 MAE와 높은 집중구간 F1을 기록하였다. 반면 LSTM, GRU 및 GCN-GRU는 현재의 단일 실측일 기반 시나리오 데이터에서는 LightGBM을 일관되게 상회하지 못하였다. 따라서 본 데이터 범위에서 딥러닝 적용 자체가 성능 향상을 보장하지 않으며, 장기간 실측 시계열이 확보될 때 시공간 딥러닝의 추가 이득을 재평가할 필요가 있다.

## 표 5. 실제 미사용 YT HOLDOUT 최종평가
| model       | horizon   |   count_mae_20YT |   top10_concentration_f1 |
|:------------|:----------|-----------------:|-------------------------:|
| Persistence | t+5       |           0.2437 |                   0.409  |
| Persistence | t+10      |           0.253  |                   0.4074 |
| Persistence | t+30      |           0.2555 |                   0.3931 |
| LightGBM    | t+5       |           0.1916 |                   0.5503 |
| LightGBM    | t+10      |           0.1884 |                   0.5589 |
| LightGBM    | t+30      |           0.1882 |                   0.553  |
| LSTM        | t+5       |           0.2133 |                   0.5572 |
| LSTM        | t+10      |           0.216  |                   0.5543 |
| LSTM        | t+30      |           0.2213 |                   0.5421 |
| GRU         | t+5       |           0.2139 |                   0.5602 |
| GRU         | t+10      |           0.2171 |                   0.5553 |
| GRU         | t+30      |           0.2225 |                   0.5383 |
| GCN-GRU     | t+5       |           0.2114 |                   0.5595 |
| GCN-GRU     | t+10      |           0.2164 |                   0.5589 |
| GCN-GRU     | t+30      |           0.2201 |                   0.5426 |

**논문 해석**  
증강 학습된 LightGBM은 실제 HOLDOUT에서 Persistence 대비 YT count MAE를 t+5 21.36%, t+10 25.52%, t+30 26.32% 낮췄으며, 상위 집중 Link 식별 F1도 향상되었다. 여기서 count MAE는 전체 터미널 차량 수가 아닌 HOLDOUT 20대 내 상대적 YT 수 기준이다.

## 표 6. 증강/Ablation
| setting                           | horizon   |   count_mae_20YT |   top10_f1 |
|:----------------------------------|:----------|-----------------:|-----------:|
| Real-only 1D                      | t+5       |           0.1993 |     0.5554 |
| Real-only 1D                      | t+10      |           0.2021 |     0.5619 |
| Real-only 1D                      | t+30      |           0.1994 |     0.5514 |
| Synthetic NORMAL                  | t+5       |           0.1878 |     0.5529 |
| Synthetic NORMAL                  | t+10      |           0.1854 |     0.5596 |
| Synthetic NORMAL                  | t+30      |           0.1863 |     0.5525 |
| 30D scenario pool (22 train days) | t+5       |           0.1916 |     0.5503 |
| 30D scenario pool (22 train days) | t+10      |           0.1884 |     0.5589 |
| 30D scenario pool (22 train days) | t+30      |           0.1882 |     0.553  |
| 30D pool w/o graph-neighbor       | t+5       |           0.1906 |     0.5557 |
| 30D pool w/o graph-neighbor       | t+10      |           0.1898 |     0.5543 |
| 30D pool w/o graph-neighbor       | t+30      |           0.187  |     0.553  |

**논문 해석**  
실제 1일 TRAIN만 학습한 경우보다 NORMAL 증강데이터를 사용했을 때 실제 HOLDOUT의 count MAE가 약 5.7~8.3% 감소하였다. 반면 stress scenario를 포함한 30일 시나리오 pool은 normal-only 대비 실제 HOLDOUT MAE가 소폭 증가했으나 held-out stress day에서 유사하거나 미세하게 안정적이었다. 따라서 PEAK·CONGESTION·LOCAL_RESTRICTION 증강은 정상상태 정확도 향상보다는 운영변동에 대한 robustness 확보 목적이 더 적절하다.

## 논문에서 주장 가능한 범위
1. 도면 기반 Lane-level Graph를 실제 TOS Replay로 검증하고 학습용 Graph로 Freeze하였다.
2. 실제 미사용 차량 HOLDOUT을 증강/학습과 분리하여 평가하였다.
3. v03 Map Matching 기반 scenario augmentation이 real-only 학습보다 HOLDOUT 예측오차를 감소시키는 효과가 확인되었다.
4. 즉시 다음 Link는 단순 2차 전이모델이 강하고, 3-Link 연속후보에서는 증강 MLP가 더 좋은 결과를 보였다.
5. Link 집중도는 LightGBM이 현재 데이터에서 가장 안정적이며 ST-GNN-lite의 추가 우위는 확인되지 않았다.

## 논문에서 과장하면 안 되는 범위
- “실제 30일 데이터로 검증” → 불가. 30-day-equivalent synthetic scenario임.
- “3일 독립 운행일로 학습/검증” → 불가. 고밀도 좌표는 사실상 1/8 하루에 집중.
- “HOLDOUT이 미래 날짜 일반화” → 불가. 동일 1/8의 미사용 YT cross-vehicle holdout.
- “ST-GNN이 기존 모델보다 우수” → 현재 결과는 이를 지지하지 않음.
- “운영 Routing Graph 완성” → 연구용 Freeze이며 operational_routing_ready는 미승인.
- “Unreal 운영검증 완료” → 아직 별도 단계.

## 그림 목록 및 권장 캡션
1. `fig01_graph_geometry_v03.png` — 도면 기반 Lane-level 공간 Graph 최종 형상.
2. `fig02_graph_freeze_edge_sources.png` — CAD 규칙 및 TRAIN 실측 전이에 따른 v05 Graph Freeze Edge 구성.
3. `fig03_route_nextlink_topk.png` — 실제 미사용 YT HOLDOUT의 Next-Link Top-K 정확도 비교.
4. `fig04_route_3step_beam.png` — 3-Link 연속경로 Beam Search Top-K 비교.
5. `fig05_holdout_concentration_mae.png` — 실제 HOLDOUT에서 t+5/t+10/t+30 Link 집중도 예측 MAE.
6. `fig06_holdout_concentration_f1.png` — 실제 HOLDOUT 상위 집중 Link 식별 F1.
7. `fig07_ablation_holdout_mae.png` — 실데이터/증강/Graph feature Ablation.
8. `fig08_holdout_link_timeseries_t10.png` — 대표 Link(R3L00083)의 실제값과 t+10 예측 시계열.
9. `fig09_synthetic_test_count_mae.png` — 미사용 synthetic scenario day의 모델별 count MAE.
10. `fig10_scenario_ablation_t10.png` — scenario별 t+10 증강 Ablation.
