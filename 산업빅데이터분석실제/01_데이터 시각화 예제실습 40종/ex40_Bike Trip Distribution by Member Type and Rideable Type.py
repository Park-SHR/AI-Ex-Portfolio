import pandas as pd
import plotly.express as px

# ------------------------------------------------------
# 1️⃣ 실제 bike trip 데이터 로드
# ------------------------------------------------------
df = pd.read_csv("data/202509-citibike-tripdata_1.csv")  # 파일명에 맞게 수정

# ------------------------------------------------------
# 2️⃣ 데이터 전처리 및 요약
#    - 회원 유형(member_casual)과 자전거 유형(rideable_type)별로
#      전체 주행 횟수를 집계
# ------------------------------------------------------
summary = (
    df.groupby(["member_casual", "rideable_type"])
    .size()
    .reset_index(name="Trip_Count")
)

# ------------------------------------------------------
# 3️⃣ Treemap 시각화
# ------------------------------------------------------
fig = px.treemap(
    summary,
    path=["member_casual", "rideable_type"],  # 계층 구조: 회원유형 → 자전거유형
    values="Trip_Count",                      # 박스 크기 = 주행 건수
    color="Trip_Count",                       # 색상도 주행 건수에 따라 표시
    color_continuous_scale="Blues",
    title="Bike Trip Distribution by Member Type and Rideable Type"
)

fig.update_layout(
    margin=dict(t=50, l=25, r=25, b=25),
    font=dict(size=14)
)

fig.show()
