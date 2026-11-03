# HJNC TOS 3-day Map Matching v03

## 수행 대상
- TOS archive: `TosIFB_Replay_all_20260106-20260108.zip`
- Spatial graph: `HJNC_Node_Link_Review_v03`
- Candidate transform: `CAD X = TOS X + 358.0`, `CAD Y = TOS Y + 469.75`
- Primary acceptance threshold: 10 local units

## 처리 결과
- Raw YT coordinate rows including missing XY: 5,685,233
- Numeric XY rows mapped: 5,677,958
- Missing XY skipped: 7,275
- Gross coordinate outliers: 3
- Unique YT: 102

## 공간 커버리지
- <=3: 3,046,007 (53.65%)
- <=5: 3,645,808 (64.21%)
- <=10: 4,120,440 (72.57%) — accepted threshold
- <=20: 4,506,667 (79.37%)
- Median match distance: 2.434
- P95 match distance: 86.830

> 이 수치는 현재 v03 도로망에 대한 **공간 커버리지**이며 정답 라벨 기반 lane accuracy가 아닙니다. Gate/North/Apron 및 일부 회전 연결이 미완성이라 해당 영역의 잔차가 포함됩니다.

## 핵심 산출물
- `yt_mapmatched_points.h5` — 전체 5,677,958건 point-level 결과
- `yt_link_sequence.csv.gz` — <=10 + jump 제외 보수적 Link sequence
- `coverage_by_yt.csv`, `coverage_by_source_day.csv`
- `link_usage_summary.csv`, `lane_usage_summary.csv`
- `transition_summary.csv`
- `unmatched_hotspots_10unit.csv`
- `yt_mapmatched_sample.csv.gz`
- `map_matching_validation.json`
- `mapmatch_overview.png`, `link_usage_map.png`, `distance_cdf.png`

## 데이터 품질 처리
- XY 결측 7,275건은 좌표 매칭 대상에서 제외했습니다.
- 비정상 대좌표 3건은 삭제하지 않고 HDF5에 status=0으로 보존했습니다.
- 속도점프 진단 533건도 삭제하지 않았고, 보수적 Link sequence 생성 시에만 제외했습니다.

## 해석
10-unit 기준 72.57%가 현 v03 Lane Graph에 직접 붙습니다. 20-unit 기준은 79.37%입니다. 나머지는 좌표 오류로 단정하지 않고 Gate/North/Apron·미복원 도로·접속 구조 및 좌표정합 오차를 함께 포함하는 **보완 우선영역**으로 분리했습니다.
