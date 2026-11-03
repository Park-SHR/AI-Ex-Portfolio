# HJNC TOS 3일 좌표 Map Matching 수행 보고서 — v03 Graph

## 1. 수행 범위

- 입력 TOS: `TosIFB_Replay_all_20260106-20260108.zip`
- 공간 Graph: `HJNC_Node_Link_Review_v03`
- 대상: TOS-11-2에서 확인되는 YT Terminal x/y 좌표
- YT 수: 102대
- 좌표 변환 후보식: `CAD local X = TOS X + 358.0`, `CAD local Y = TOS Y + 469.75`
- 1차 수용 임계값: 최근접/방향보정 Lane 거리 10 local unit 이하

본 결과는 **현재 v03 Lane Graph에 실제 TOS 좌표를 붙인 연구용 Map Matching 결과**이다. 도면에 아직 포함되지 않은 Gate/North/Apron과 운영 검증 전 Turn Connector 때문에 미매칭이 발생할 수 있으므로, 10m/20m 커버리지를 곧바로 정답 정확도로 해석하지 않는다.

## 2. 원천 좌표 품질

| 항목 | 결과 |
|---|---:|
| 원시 YT 좌표 행(결측 포함) | 5,685,233 |
| 숫자형 x/y 확보 행 | 5,677,958 |
| x/y 결측 | 7,275 |
| 비정상 대좌표 | 3 |
| 고유 YT | 102 |
| 속도점프 진단 플래그 | 533 |

비정상 대좌표 3건과 속도점프 진단값은 원본 보존 원칙에 따라 point-level HDF5에서 삭제하지 않았다. 비정상 대좌표는 `accepted=0`, 속도점프는 `jump_flag=1`로 구분했으며, 보수적 Link sequence 생성에서만 jump flag를 제외했다.

## 3. 공간 Map Matching 결과

| Lane 거리 기준 | 매칭 좌표 | 전체 숫자좌표 대비 |
|---|---:|---:|
| ≤ 3 local unit | 3,046,007 | 53.65% |
| ≤ 5 local unit | 3,645,808 | 64.21% |
| ≤ 10 local unit | 4,120,440 | 72.57% |
| ≤ 20 local unit | 4,506,667 | 79.37% |

- 거리 중앙값: **2.434 local unit**
- 95% 분위 거리: **86.830 local unit**
- 보수적 Link sequence 사용 포인트: **4,120,119건**
- 연속 Link run: **349,135개**

10-unit 기준으로 약 72.57%가 현재 도면 기반 Graph에 직접 붙는다. 반면 P95가 큰 이유는 단순 좌표 오차뿐 아니라 **현재 Graph 범위 밖의 실제 이동 좌표가 상당수 존재**하기 때문이다.

## 4. 소스 날짜별 실제 좌표량

| 소스 일자 | 숫자형 좌표 행 | ≤10 커버리지 |
|---|---:|---:|
| 2026-01-06 | 102 | 36.27% |
| 2026-01-07 | 102 | 39.22% |
| 2026-01-08 | 5,677,754 | 72.57% |

중요하게, 파일 묶음은 2026-01-06~08의 3일 구성이지만 **YT 위치 좌표의 절대다수는 1월 8일 Replay에 집중되어 있다.** 따라서 향후 학습/검증 분할에서 이를 단순한 3일 균등 데이터로 취급하면 안 된다.

## 5. Graph 사용 실적

- Lane: **49 / 49개에서 실제 매칭 관측 발생**
- Link: **1,004 / 1,007개에서 실제 매칭 관측 발생**
- 현재 관측이 없는 Link: `R3L00137`, `R3L00138`, `R3L00257`

즉 v03 주요 Lane 구조 자체는 실제 TOS 이동 영역과 상당히 폭넓게 겹친다. 사용되지 않은 3개 Link는 즉시 삭제하지 말고 도면상 짧은 교차부 Link인지, 실제 미사용 구간인지 추가 확인 대상으로 유지한다.

## 6. 미매칭 Hotspot

20-unit 초과 잔차를 10-unit 격자로 집계하면 상위 구역은 대체로 CAD local X 약 260~330, Y 약 820~890 부근에 집중된다. 이는 현재 v03 주 도로망의 최소 X(약 359.5)보다 서측에 위치하여 **현재 미복원 외곽/Gate/North 계열 공간 후보**와 일치하는 형태이다.

따라서 잔여 20.63%를 단순 Map Matching 실패로 보지 않고 다음 두 종류로 분리해야 한다.

1. Graph 외부/미복원 구간 — 신규 도로/접속 구조 보완 대상
2. Graph 내부인데 잔차가 큰 구간 — 좌표 변환식, 차선 중심선, 교차부 연결 검토 대상

## 7. Transition 검증 해석

현재 point-level 매칭은 공간거리 + 이동축/진행방향을 사용하지만 **Turn Connector 전체를 강제로 적용하는 HMM/Viterbi 방식은 사용하지 않았다.** 이유는 v03에서 수평→수직 회전 등 일부 Turn Connector가 아직 `review_required` 상태이기 때문이다.

따라서 `transition_summary.csv`의 `not_directly_encoded`나 `prohibited_by_symbol`을 즉시 실제 위반으로 단정해서는 안 된다. 짧은 교차부 Link를 좌표 업데이트 사이에 건너뛴 경우와, 교차로 근방에서 인접 Lane으로 순간 배정된 경우가 포함될 수 있다. 이 파일은 **Turn Connector 보정 우선순위를 찾기 위한 증거자료**로 사용한다.

## 8. 다음 연구 단계에 사용할 데이터

이번 결과로 다음 구조가 실제로 확보됐다.

`TOS Terminal x/y` → `CAD 좌표 변환` → `Lane` → `Link` → `YT별 시간순 Link sequence`

따라서 다음 단계에서는 `yt_link_sequence.csv.gz`를 기반으로

- OD별 실제 통과 Link 패턴 추출
- 교차로별 From-Link → To-Link 빈도 산출
- 미검증 Turn Connector의 Replay 기반 검증
- Link-Time Tensor 생성
- Link별 YT 수 / 점유 / 정차 / 통과시간 집계
- Top-K 경로 및 혼잡 예측 학습

으로 이어갈 수 있다.

## 9. 산출물

- `yt_mapmatched_points.h5`: 전체 point-level Map Matching 결과
- `yt_link_sequence.csv.gz`: 보수적 시간순 Link sequence
- `coverage_by_yt.csv`
- `coverage_by_source_day.csv`
- `link_usage_summary.csv`
- `lane_usage_summary.csv`
- `transition_summary.csv`
- `unmatched_hotspots_10unit.csv`
- `yt_mapmatched_sample.csv.gz`
- `map_matching_validation.json`
- `mapmatch_overview.png`
- `link_usage_map.png`
- `distance_cdf.png`
