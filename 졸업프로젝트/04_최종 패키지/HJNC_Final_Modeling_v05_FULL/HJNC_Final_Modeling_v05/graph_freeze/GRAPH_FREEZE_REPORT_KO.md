# HJNC Model Graph v05 — 최종 Freeze 보고서

## 1. Freeze 목적
`HJNC_Node_Link_Review_v03`의 Lane/Link geometry를 **모델 학습용 고정 Graph**로 확정하였다. 본 Freeze는 연구/모델용이며 실제 장비 자동 경로지시를 승인하는 운영 Routing Graph가 아니다.

## 2. Freeze 규칙
- Link node: v03의 **1,007개 Link를 모두 유지**
- CAD/도면 근거: `turn_permissions.analysis_routing_enabled = 1`
- 실측 근거: TRAIN YT에서 `mean_match_dist <= 5`, 전이 gap <=120초, endpoint gap <=5, 명시적 `prohibited_by_symbol`이 아닌 전이 중 **2회 이상 관측**
- 최종 edge: CAD 허용 edge **또는** TRAIN 실측 2회 이상 edge
- 실제 HOLDOUT 20대 YT는 Freeze 과정에 사용하지 않음
- `operational_routing_ready`는 계속 0으로 유지

## 3. Freeze 결과
- Link node: **1,007**
- Directed model edge: **5,900**
- CAD rule only: **189**
- CAD + empirical both: **762**
- Empirical TRAIN only: **4,949**
- Weak component: **4개**, 최대 component **1,004 Link**
- Strong component: **23개**, 최대 SCC **985 Link**
- Isolated Link: **3개** — R3L00944, R3L00976, R3L00978
- HOLDOUT transition weighted coverage: **98.29%**
- HOLDOUT unique transition coverage: **83.54%**

## 4. 해석
가중 전이 기준 HOLDOUT의 98% 이상이 HOLDOUT을 보지 않고 Freeze한 Graph에 포함되므로, 현재 주요 YT 운행영역의 **모델용 전이 구조는 충분히 안정적**이다. 반면 unique transition coverage가 83.54%인 것은 희귀/1회성 전이와 교차로 부근 Map Matching 변동이 남아 있다는 의미다. 이를 억지로 Edge로 추가하지 않고 HOLDOUT 검증 대상으로 남겼다.

## 5. 남은 비모델 영역
Gate/North/Apron의 도면 미복원 구간, 공식 측량 Control Point 기반 좌표변환, 현장 회전반경/장애물 확인은 운영 승인 관점의 후속 검증이다. **현재 논문 모델 학습용 Graph는 v05로 Freeze**한다.
