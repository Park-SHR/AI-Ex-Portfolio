import pandas as pd
import plotly.express as px

# -----------------------------------------
# 1️⃣ 데이터 로드
# -----------------------------------------
df = pd.read_csv("data/202509-citibike-tripdata_1.csv")

# -----------------------------------------
# 2️⃣ 결측 제거 및 샘플링
# -----------------------------------------
df = df.dropna(subset=["start_lat", "start_lng", "member_casual"])
if len(df) > 50000:
    df = df.sample(50000, random_state=42)

# -----------------------------------------
# 3️⃣ 밀도 등고선 시각화 (회원유형별)
# -----------------------------------------
fig = px.density_contour(
    df,
    x="start_lng",
    y="start_lat",
    color="member_casual",  # 그룹 기준 컬러 (categorical)
    nbinsx=60,
    nbinsy=60,
    title="Spatial Density of Bike Trip Start Locations by Member Type"
)

# 등고선 색 채우기 옵션
fig.update_traces(contours_coloring="fill", contours_showlabels=False)

# 산점도 점도 같이 추가 (옵션)
fig.add_scatter(
    x=df["start_lng"], y=df["start_lat"],
    mode="markers",
    marker=dict(size=1, opacity=0.3, color="black"),
    name="Trip Start Points"
)

fig.update_layout(
    xaxis_title="Longitude",
    yaxis_title="Latitude",
    legend_title_text="Member Type",
    margin=dict(t=50, l=25, r=25, b=25)
)

fig.show()
