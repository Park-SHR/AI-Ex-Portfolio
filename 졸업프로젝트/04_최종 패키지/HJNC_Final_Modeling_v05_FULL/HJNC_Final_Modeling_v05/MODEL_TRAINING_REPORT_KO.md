# HJNC v05 모델 학습·비교·HOLDOUT 평가 보고서

## 1. 데이터와 평가 원칙
- Geometry/Graph: `HJNC_Node_Link_Review_v03` → 본 작업에서 `HJNC_Model_Graph_v05`로 Freeze
- Map Matching: `HJNC_TOS_MapMatching_v03`
- Augmentation: `HJNC_Scenario_Augmentation_30D_v04`
- 실제 고밀도 좌표일: **2026-01-08**
- 실제 TRAIN: **77 YT**
- 실제 HOLDOUT: **20 YT** — v04 증강 생성, v05 Graph Freeze, 모델 학습에 사용하지 않음
- 실제 HOLDOUT은 **다른 날짜 HOLDOUT이 아니라 동일 1/8 일자의 미사용 차량 HOLDOUT**임
- 30일 데이터는 실제 30일 운영로그가 아니라 **30-day-equivalent scenario augmentation**

## 2. Synthetic day split
- Train: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 22, 23, 24, 26, 27, 28 (**22일**)
- Validation: 18, 21, 29 (**3일**)
- Test: 17, 19, 20, 25, 30 (**5일**)
- Test는 LOCAL_RESTRICTION, CONGESTION, LOW, PEAK, NORMAL을 각 1일 포함

## 3. 경로/다음 Link 예측
입력은 최근 8개 Link이며, 예측 결과는 v05 Frozen Graph candidate로 제한하였다.

### 실제 미사용 YT HOLDOUT — Next-Link Top-K
| model               |   top1 |   top3 |   top5 |   mrr5 |
|:--------------------|-------:|-------:|-------:|-------:|
| Graph-Markov-1      |  54.86 |  88.25 |  94.59 |  71.69 |
| Graph-Markov-2      |  60.99 |  90.61 |  95.51 |  75.79 |
| MLP real-only       |  60.5  |  89.26 |  94.81 |  75    |
| MLP + augmentation  |  60.87 |  89.98 |  95.25 |  75.53 |
| LSTM + augmentation |  56.75 |  86.77 |  93.62 |  72.09 |
| GRU + augmentation  |  57.98 |  87.73 |  93.88 |  73    |

**핵심 결과**
- 즉시 다음 Link Top-1은 `Graph-Markov-2`가 **60.99%**로 가장 높다.
- `MLP + augmentation`은 Top-1 **60.87%**, Top-5 **95.25%**로 매우 근접한다.
- MLP의 증강 적용 효과는 real-only 대비 Top-1 **+0.37%p**, Top-3 **+0.72%p**, Top-5 **+0.44%p**이다.
- LSTM/GRU는 현재 데이터에서는 MLP/2차 Markov보다 낮다. 한 날짜의 반복적 항만 운행패턴에서는 복잡한 시퀀스 모델이 자동으로 우위가 되지 않는다는 결과다.

### 3-Link 연속경로 Beam Search (HOLDOUT 8,000 window)
| model              |   route3_top1 |   route3_top3 |   route3_linkhit_top1 |
|:-------------------|--------------:|--------------:|----------------------:|
| Graph-Markov-2     |         22.16 |         42.78 |                 41.82 |
| MLP + augmentation |         24.71 |         47.02 |                 45.38 |

즉시 다음 Link는 Markov-2가 강하지만, **3개 Link 연속경로 Top-3**에서는 증강 MLP가 **47.02%**로 Markov-2의 **42.78%**보다 높다. 따라서 현재 데이터에서는 **단기 1-step은 Graph-Markov-2, 다단계 후보경로는 증강 MLP** 조합이 합리적이다.

## 4. Link 집중도 예측 — Synthetic held-out scenario days
Target은 5분 Link별 YT 집중 share(`YT count / active YT`)이며, 동등한 YT count MAE도 함께 산출했다.

| model       | horizon   |   count_mae |   top10_concentration_f1 |   share_mae |   active_share_mae |
|:------------|:----------|------------:|-------------------------:|------------:|-------------------:|
| Persistence | t+5       |      0.6674 |                   0.5153 |      0.0071 |             0.0183 |
| Persistence | t+10      |      0.7266 |                   0.4863 |      0.0077 |             0.0193 |
| Persistence | t+30      |      0.752  |                   0.4644 |      0.008  |             0.0196 |
| LightGBM    | t+5       |      0.5272 |                   0.6017 |      0.0056 |             0.0138 |
| LightGBM    | t+10      |      0.544  |                   0.5893 |      0.0058 |             0.0137 |
| LightGBM    | t+30      |      0.5547 |                   0.5773 |      0.0059 |             0.0135 |
| LSTM        | t+5       |      0.6618 |                   0.5837 |      0.007  |             0.0135 |
| LSTM        | t+10      |      0.6789 |                   0.5779 |      0.0072 |             0.0136 |
| LSTM        | t+30      |      0.7036 |                   0.5596 |      0.0075 |             0.0139 |
| GRU         | t+5       |      0.6537 |                   0.5839 |      0.007  |             0.0136 |
| GRU         | t+10      |      0.6733 |                   0.5748 |      0.0072 |             0.0138 |
| GRU         | t+30      |      0.6995 |                   0.5581 |      0.0074 |             0.014  |
| GCN-GRU     | t+5       |      0.6625 |                   0.5863 |      0.007  |             0.0136 |
| GCN-GRU     | t+10      |      0.6875 |                   0.576  |      0.0073 |             0.0137 |
| GCN-GRU     | t+30      |      0.7107 |                   0.5596 |      0.0076 |             0.014  |

LightGBM은 Persistence 대비 count MAE를 t+5/t+10/t+30에서 각각 **21.0% / 25.1% / 26.2%** 낮췄다. GCN-GRU(ST-GNN-lite)는 공간 인접 Link의 `A_norm·X`를 GRU 입력에 결합했으나, 현재 데이터에서는 LightGBM보다 우수하지 않았다.

## 5. 최종 실제 HOLDOUT 평가
| model       | horizon   |   count_mae_20YT |   top10_concentration_f1 |   share_mae |   active_share_mae |
|:------------|:----------|-----------------:|-------------------------:|------------:|-------------------:|
| Persistence | t+5       |           0.2437 |                   0.409  |      0.0122 |             0.055  |
| Persistence | t+10      |           0.253  |                   0.4074 |      0.0126 |             0.0563 |
| Persistence | t+30      |           0.2555 |                   0.3931 |      0.0128 |             0.0563 |
| LightGBM    | t+5       |           0.1916 |                   0.5503 |      0.0096 |             0.039  |
| LightGBM    | t+10      |           0.1884 |                   0.5589 |      0.0094 |             0.0376 |
| LightGBM    | t+30      |           0.1882 |                   0.553  |      0.0094 |             0.0368 |
| LSTM        | t+5       |           0.2133 |                   0.5572 |      0.0107 |             0.0371 |
| LSTM        | t+10      |           0.216  |                   0.5543 |      0.0108 |             0.037  |
| LSTM        | t+30      |           0.2213 |                   0.5421 |      0.0111 |             0.0373 |
| GRU         | t+5       |           0.2139 |                   0.5602 |      0.0107 |             0.0372 |
| GRU         | t+10      |           0.2171 |                   0.5553 |      0.0109 |             0.0372 |
| GRU         | t+30      |           0.2225 |                   0.5383 |      0.0111 |             0.0373 |
| GCN-GRU     | t+5       |           0.2114 |                   0.5595 |      0.0106 |             0.0378 |
| GCN-GRU     | t+10      |           0.2164 |                   0.5589 |      0.0108 |             0.0376 |
| GCN-GRU     | t+30      |           0.2201 |                   0.5426 |      0.011  |             0.0377 |

LightGBM은 실제 미사용 20 YT HOLDOUT에서 Persistence 대비 count MAE를:
- t+5: **21.36% 감소** (0.2437 → 0.1916)
- t+10: **25.52% 감소**
- t+30: **26.32% 감소**

상위 10% 집중 Link 식별 F1도 모든 horizon에서 Persistence보다 향상되었다. 단, `count_mae_20YT`는 **20대 HOLDOUT subset 내부의 YT 수 오차**이며 전체 터미널 YT 수 오차로 해석하면 안 된다.

## 6. Ablation
| setting                           | horizon   |   count_mae_20YT |   top10_f1 |   active_share_mae |
|:----------------------------------|:----------|-----------------:|-----------:|-------------------:|
| Real-only 1D                      | t+5       |           0.1993 |     0.5554 |             0.0386 |
| Real-only 1D                      | t+10      |           0.2021 |     0.5619 |             0.0368 |
| Real-only 1D                      | t+30      |           0.1994 |     0.5514 |             0.0372 |
| Synthetic NORMAL                  | t+5       |           0.1878 |     0.5529 |             0.0389 |
| Synthetic NORMAL                  | t+10      |           0.1854 |     0.5596 |             0.0373 |
| Synthetic NORMAL                  | t+30      |           0.1863 |     0.5525 |             0.0371 |
| 30D scenario pool (22 train days) | t+5       |           0.1916 |     0.5503 |             0.039  |
| 30D scenario pool (22 train days) | t+10      |           0.1884 |     0.5589 |             0.0376 |
| 30D scenario pool (22 train days) | t+30      |           0.1882 |     0.553  |             0.0368 |
| 30D pool w/o graph-neighbor       | t+5       |           0.1906 |     0.5557 |             0.0389 |
| 30D pool w/o graph-neighbor       | t+10      |           0.1898 |     0.5543 |             0.0373 |
| 30D pool w/o graph-neighbor       | t+30      |           0.187  |     0.553  |             0.037  |

### Ablation 해석
1. 실제 1일 TRAIN만 사용한 LightGBM보다 `Synthetic NORMAL` 학습이 HOLDOUT count MAE를 **약 5.7~8.3%** 감소시켰다. → 증강의 유효성은 확인됨.
2. 30일 시나리오 pool(22 train days)은 실제 HOLDOUT에서 NORMAL-only보다 MAE가 아주 약간 높다. → PEAK/CONGESTION/RESTRICTION stress가 normal-domain calibration을 조금 넓힌 결과.
3. 반면 held-out synthetic scenario test 평균에서는 NORMAL-only와 30D scenario pool이 거의 동일하며, 30D pool이 stress day에서 미세하게 안정적이다. → stress 시나리오는 **정확도 상승용이라기보다 robustness용**으로 해석하는 것이 타당하다.
4. Graph-neighbor feature 제거 효과는 horizon별로 혼재되어, 현재 한 날짜 기반 데이터에서는 공간 인접정보의 추가 이득이 일관적이지 않았다.
5. 따라서 **“ST-GNN이 반드시 더 좋다”는 결론은 지지되지 않는다.** 오히려 현재 데이터에서는 LightGBM/Graph-Markov가 강한 기준선이며, 장기간 실측 로그 확보 후 ST-GNN 고도화가 필요하다.

## 7. 최종 모델 선택
- **즉시 다음 Link:** Graph-Markov-2
- **다단계 후보경로:** MLP + v04 augmentation + Frozen Graph mask
- **t+5/t+10/t+30 Link 집중도:** LightGBM
- **ST-GNN 비교군:** GCN-GRU(ST-GNN-lite), 현 데이터에서 우위 없음
- Unreal 연동 시 위 세 출력을 `prediction_result`에 병렬 제공하는 구조를 권장
