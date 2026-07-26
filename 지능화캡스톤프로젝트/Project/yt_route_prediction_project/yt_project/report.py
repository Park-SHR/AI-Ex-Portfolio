from __future__ import annotations
from pathlib import Path
import pandas as pd


def write_markdown_report(output_dir: Path, tos_summary: dict, route_metrics: dict, model_metrics: pd.DataFrame, route_choice_metrics: pd.DataFrame) -> Path:
    report_path = output_dir / "reports" / "final_validation_report.md"
    lines = []
    lines.append("# YT 경로·혼잡 예측 알고리즘 검증 리포트\n")
    lines.append("## 1. 최종본 개발 가능성 검토\n")
    lines.append("- 현재 TOS 데이터가 Bitt/선석 정적 좌표 중심인 경우, 실제 YT 궤적 지도학습 성능을 바로 주장할 수는 없습니다.\n")
    lines.append("- 그러나 정적 좌표를 Anchor로 활용해 도로망 그래프, OD 후보, SOP 기반 이벤트, 링크 단위 시계열 텐서를 생성하는 1차 검증은 가능합니다.\n")
    lines.append("- 따라서 본 프로젝트는 Meta K-Port 연동 전 단계에서 SOP 시뮬레이션, Baseline/ML/ST-GNN 모델 입출력, Ablation 검증을 수행하도록 구성되었습니다.\n")
    lines.append("\n## 2. TOS 데이터 요약\n")
    for k, v in tos_summary.items():
        lines.append(f"- {k}: {v}\n")
    lines.append("\n## 3. SOP/경로 선택 검증\n")
    for k, v in route_metrics.items():
        lines.append(f"- {k}: {v:.4f}\n" if isinstance(v, float) else f"- {k}: {v}\n")
    if not route_choice_metrics.empty:
        lines.append("\n### Route Choice Model\n")
        lines.append(route_choice_metrics.to_markdown(index=False))
        lines.append("\n")
    lines.append("\n## 4. 링크 혼잡 예측 모델 비교\n")
    if not model_metrics.empty:
        lines.append(model_metrics.to_markdown(index=False))
        lines.append("\n")
    lines.append("\n## 5. 산출물\n")
    lines.append("- tables/anchor_master.csv: TOS Bitt/선석 Anchor 정리\n")
    lines.append("- tables/node_master.csv: QC/Yard/Gate/교차로 보완 Node Master\n")
    lines.append("- tables/link_master.csv: ST-GNN 입력용 도로 링크 정의\n")
    lines.append("- tables/sop_template.csv: SOP 작업 유형 정의\n")
    lines.append("- tables/synthetic_trace.csv: SOP 기반 가상 YT 주행 이벤트\n")
    lines.append("- tables/link_time_samples.csv: 링크×시간 Feature 및 혼잡 점수\n")
    lines.append("- tables/model_metrics.csv: Baseline/ML/ST-GNN 예측 성능 비교\n")
    lines.append("- plots/*.png: 그래프, 혼잡 Heatmap, 모델 비교 시각화\n")
    lines.append("- unreal_prediction_sample.json: 추후 META K-Port/Unreal 연동용 예측 결과 샘플\n")
    lines.append("\n## 6. 해석 주의사항\n")
    lines.append("- 현재 결과는 실제 항만 성능 수치가 아니라, 발표자료상의 방법론이 개발·검증 가능한지 확인하는 사전 검증 결과입니다.\n")
    lines.append("- 실제 YT 로그가 확보되면 synthetic_trace를 실측 trace로 교체하고, 동일한 평가 파이프라인으로 Top-K, ETA, DTW, 혼잡 F1을 산출하면 됩니다.\n")
    report_path.write_text("".join(lines), encoding="utf-8")
    return report_path
