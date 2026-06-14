# YT Route & Congestion Prediction Validation Project

이 프로젝트는 발표자료의 연구 흐름을 **Meta K-Port 적용 전 Python 환경에서 검증**하기 위한 실행형 프로그램입니다.

핵심 목적은 다음과 같습니다.

1. TOS Bitt/선석 좌표 데이터를 읽어 Anchor Master 생성
2. Anchor를 기준으로 QC/Yard/Gate/교차로 Node Master와 Link Master 생성
3. 항만 SOP 기반 작업 OD와 운행 이벤트 생성
4. Dijkstra/K-shortest를 정답이 아닌 후보 경로·Baseline으로만 사용
5. 속도, 대기, 장비 간섭, 시간대별 수요 등 지연 요소를 주입해 synthetic_trace 생성
6. 링크×시간 단위 Feature와 혼잡 점수 생성
7. Baseline, RandomForest, GradientBoosting, MLP, Simple ST-GNN 모델 비교
8. 예측 결과를 표, 그래프, 리포트, Unreal 연동용 JSON으로 출력

---

## 1. 설치

```bash
pip install -r requirements.txt
```

PyTorch가 없는 환경에서는 `pip install torch`가 필요합니다. CPU 환경에서도 기본 예제는 수행됩니다.

---

## 2. 빠른 실행

실제 TOS JSON이 없을 때는 발표자료의 구조를 참고한 데모 Anchor 데이터가 자동 생성됩니다.

```bash
python run_pipeline.py --output_dir outputs --duration_minutes 120 --tasks_per_hour 45 --train_stgnn --stgnn_epochs 2 --max_links_for_stgnn 30
```

Windows에서는 다음 파일을 더블클릭해도 됩니다.

```bat
quick_start.bat
```

---

## 3. 실제 TOS JSON 적용

TOS JSON 폴더가 있을 경우:

```bash
python run_pipeline.py --tos_dir ./data/TOS_1_7 --output_dir outputs_real --train_stgnn
```

예상 JSON 구조:

```json
{
  "updateTime": "2026-01-19T00:00:00",
  "data": [
    {"bitt": "B001", "berth": "1B", "x": -27.65, "y": 0},
    {"bitt": "B002", "berth": "1B", "x": -10.00, "y": 2.5}
  ]
}
```

현재 프로그램은 `data` 내부의 `bitt/berth/x/y`를 읽습니다. 차량 궤적 로그가 추가되기 전까지는 이 데이터를 Anchor와 도로망 생성 기준으로 사용합니다.

---

## 4. 주요 산출물

실행 후 `outputs/` 아래에 다음 파일이 생성됩니다.

| 경로 | 설명 |
|---|---|
| `tables/anchor_master.csv` | TOS Bitt/선석 Anchor 정리 |
| `tables/node_master.csv` | QC/Yard/Gate/교차로 보완 노드 정의 |
| `tables/link_master.csv` | 도로망 링크 정의 |
| `tables/sop_template.csv` | SOP 작업 유형 정의 |
| `tables/task_plan.csv` | SOP 기반 작업 계획 |
| `tables/synthetic_trace.csv` | 가상 YT 주행 이벤트 |
| `tables/link_time_samples.csv` | 링크×시간 Feature 및 혼잡 점수 |
| `tables/model_metrics.csv` | Baseline/ML/ST-GNN 성능 비교 |
| `plots/01_port_graph.png` | 생성된 항만 도로망 그래프 |
| `plots/02_congestion_heatmap.png` | 링크×시간 혼잡 Heatmap |
| `plots/03_pred_vs_actual_tabular.png` | Tabular ML 예측 결과 비교 |
| `tables/stgnn_y_test.npy`, `tables/stgnn_y_pred.npy` | ST-GNN 실제/예측 배열 |
| `unreal_prediction_sample.json` | Meta K-Port/Unreal 연동용 샘플 JSON |
| `reports/final_validation_report.md` | 최종 검증 리포트 |

---

## 5. 프로그램 구조

```text
yt_route_prediction_project/
├─ run_pipeline.py
├─ requirements.txt
├─ quick_start.bat
├─ quick_start.sh
├─ yt_project/
│  ├─ tos_loader.py       # TOS JSON 로드 및 Anchor Master
│  ├─ graph_builder.py    # Node/Link Master 및 K-shortest 경로
│  ├─ sop.py              # SOP 작업 유형 및 Task Plan 생성
│  ├─ simulate.py         # 운행 Trace 및 지연 이벤트 시뮬레이션
│  ├─ features.py         # 링크×시간 Feature 및 ST-GNN Tensor 생성
│  ├─ models.py           # Baseline/ML 모델
│  ├─ stgnn.py            # Simple ST-GNN 모델
│  ├─ evaluate.py         # Top-K, MAE, RMSE, F1, IoU 평가
│  ├─ visualize.py        # 그래프/Heatmap/예측 비교 그림
│  └─ report.py           # Markdown 리포트 생성
└─ outputs/
```

---

## 6. 발표자료와의 매핑

| 발표자료 요소 | 프로그램 구현 위치 |
|---|---|
| TOS 정적 좌표 분석 | `tos_loader.py`, `anchor_master.csv` |
| 도로망 그래프 구성 | `graph_builder.py`, `node_master.csv`, `link_master.csv` |
| SOP 적용 | `sop.py`, `sop_template.csv`, `task_plan.csv` |
| 지연·대기·장비 간섭 주입 | `simulate.py`, `synthetic_trace.csv` |
| 링크 단위 시계열 텐서 | `features.py`, `link_time_samples.csv` |
| Dijkstra/K-shortest Baseline | `graph_builder.py`, `route_baseline_metrics.csv` |
| ML 모델 검증 | `models.py`, `model_metrics.csv` |
| ST-GNN 검증 | `stgnn.py`, `SimpleSTGNN.pt` |
| 예측 결과 검증 | `evaluate.py`, `plots/*.png`, `final_validation_report.md` |
| Meta K-Port 연동 전 결과 포맷 | `unreal_prediction_sample.json` |

---

## 7. 주의사항

- 현재 프로그램의 기본 실행 결과는 실제 항만 성능 수치가 아니라 **방법론 검증용 결과**입니다.
- 실제 YT ID별 위치·속도·작업 로그가 확보되면 `synthetic_trace.csv` 생성 단계를 실측 Trace 입력 단계로 교체하면 됩니다.
- `SimpleSTGNN`은 연구 파이프라인 검증용 경량 모델입니다. 논문 수준으로 확장할 때는 STGCN, ASTGCN, DCRNN 등으로 교체 가능합니다.
