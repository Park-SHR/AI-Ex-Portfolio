import plotly.express as px

# Plotly Express 내장 iris 데이터 불러오기
df = px.data.iris()

# 3D 산점도: sepal_length, sepal_width, petal_width 축 사용
fig = px.scatter_3d(
    df,
    x="sepal_length",
    y="sepal_width",
    z="petal_width",
    color="species",            # 종(species)에 따라 색 구분
    size="petal_length",         # 꽃잎 길이에 따라 점 크기 변화
    symbol="species",            # 종에 따라 마커 모양 변화
    opacity=0.8,                 # 투명도
    title="Iris Dataset 3D Scatter"
)

fig.update_layout(margin=dict(l=0, r=0, b=0, t=50))
fig.show()
