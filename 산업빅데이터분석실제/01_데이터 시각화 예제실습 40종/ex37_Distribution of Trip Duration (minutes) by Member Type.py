import pandas as pd
import plotly.express as px

# ---------------------------------------
# 1️⃣ 데이터 로드
# ---------------------------------------
df = pd.read_csv("data/202509-citibike-tripdata_1.csv")

# ---------------------------------------
# 2️⃣ 결측값 처리 + 이용시간 계산
# ---------------------------------------
df = df.dropna(subset=["started_at", "ended_at", "member_casual"])
df["started_at"] = pd.to_datetime(df["started_at"], errors="coerce")
df["ended_at"] = pd.to_datetime(df["ended_at"], errors="coerce")

# 이용시간(분 단위) 계산
df["duration_min"] = (df["ended_at"] - df["started_at"]).dt.total_seconds() / 60

# 이상치 제거 (0분 이하 또는 180분 초과 제거)
df = df[(df["duration_min"] > 0) & (df["duration_min"] <= 180)]

# ---------------------------------------
# 3️⃣ 히스토그램 시각화
# ---------------------------------------
fig = px.histogram(
    df,
    x="duration_min",
    color="member_casual",       # 회원/비회원 그룹별 비교
    nbins=60,                    # 구간 개수 (bin)
    barmode="overlay",           # 겹쳐서 비교
    opacity=0.6,                 # 투명도
    title="Distribution of Trip Duration (minutes) by Member Type"
)

fig.update_layout(
    xaxis_title="Trip Duration (minutes)",
    yaxis_title="Count",
    legend_title_text="Member Type",
    margin=dict(t=50, l=25, r=25, b=25)
)

fig.show()
