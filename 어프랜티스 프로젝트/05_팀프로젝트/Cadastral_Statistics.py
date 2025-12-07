import os
from pathlib import Path

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# 🔤 한글 폰트 설정 (Windows: 맑은 고딕)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지


import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import geopandas as gpd
from shapely.geometry import mapping
import folium
import h3


# =========================
# 1. CONFIG 영역 (여기만 네 환경에 맞게 수정)
# =========================

# 1) 지적통계 CSV 폴더
DATA_DIR = Path(r"./data/")  # 예: r"C:\Users\...\충북지적통계"

# 2) 연도별 CSV 파일 정보 (파일명은 네 실제 파일명으로 수정)
YEAR_FILES = [
    ("충청북도 지적통계 2017년 4분기.csv", 2017),
    ("충청북도 지적통계_20181001.csv", 2018),
    ("충청북도 지적통계_20190401.csv", 2019),
    ("충청북도 지적통계_20200109.csv", 2020),
    ("충청북도_지적통계_20230701.csv", 2023),
    ("충청북도_지적통계_20240701.csv", 2024),
    ("충청북도_지적통계_20250630.csv", 2025),
]

# 3) 공간 데이터 (충북 시군구 GeoJSON)
GEOJSON_PATH = Path(r"./data/SIG.geojson")
GEOJSON_REGION_COL = "SIG_KOR_NM"  # GeoJSON에서 시군구 이름 컬럼명

# 4) 토지소재명 컬럼명 (CSV 안에서 시군구/구 이름)
REGION_COL = "토지소재명"

# 5) 분석 타겟 연도 (도넛 차트 & 지도용)
TARGET_YEAR = 2025

# 6) H3 resolution (7~8 정도가 한국 도 단위에 적당)
H3_RESOLUTION = 7

# 7) 결과 지도 HTML 파일
H3_MAP_OUTPUT = "chungbuk_landuse_h3_cluster_map.html"


# =========================
# 2. 유틸 함수들
# =========================

def read_csv_safely(path: Path) -> pd.DataFrame:
    """cp949 / utf-8-sig 둘 다 시도해서 읽기"""
    for enc in ["cp949", "utf-8-sig", "utf-8"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    # 다 실패하면 기본 디코딩
    return pd.read_csv(path)


def load_all_years(data_dir: Path, year_files):
    """연도별 CSV를 읽어서 하나의 DataFrame으로 통합"""
    dfs = []
    for fname, year in year_files:
        fpath = data_dir / fname
        if not fpath.exists():
            print(f"[WARN] 파일 없음: {fpath}")
            continue
        df = read_csv_safely(fpath)
        df["연도"] = year
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError("연도별 지적통계 CSV를 찾을 수 없습니다. DATA_DIR / YEAR_FILES 확인 필요.")
    raw = pd.concat(dfs, ignore_index=True)
    return raw


def get_area_columns(df: pd.DataFrame):
    """면적 관련 컬럼만 자동 추출 (예: '전 면적', '대 면적', '공장용지 면적' 등)
       필요시 여기에서 제외/포함 컬럼 추가 조정 가능"""
    cols = [c for c in df.columns if "면적" in c]
    # 혹시 '총면적' 같은 게 원본에 있다면 제거
    cols = [c for c in cols if c not in ["총면적", "전체면적"]]
    return cols


def build_ratio_df(raw: pd.DataFrame, area_cols):
    """시군구·연도별 용도 비율 DataFrame(ratio_df)와
       도 전체 연도별 비율(year_ratio)을 생성"""

    raw = raw.copy()

    # 1) 면적 합산해서 총면적 계산
    raw["총면적"] = raw[area_cols].sum(axis=1)

    # 2) 총면적이 0이거나 NaN인 행은 분석 대상에서 제외 (NaN 방지)
    before = len(raw)
    raw = raw[raw["총면적"] > 0].copy()
    after = len(raw)
    print(f"[INFO] 총면적 0 또는 NaN인 행 {before - after}개 제거")

    # 3) 비율 컬럼 생성
    ratio_df = raw[["연도", REGION_COL]].copy()
    ratio_cols = []
    for col in area_cols:
        ratio_col = col.replace("면적", "비율")
        ratio_df[ratio_col] = raw[col] / raw["총면적"]
        ratio_cols.append(ratio_col)

    # 4) 혹시라도 남아있는 NaN → 0으로 채우기 (안전장치)
    ratio_df[ratio_cols] = ratio_df[ratio_cols].fillna(0)

    # 5) 도 전체 연도별 비율
    year_area = raw.groupby("연도")[area_cols].sum()
    year_area["총면적"] = year_area.sum(axis=1)
    year_ratio = year_area[area_cols].div(year_area["총면적"], axis=0)
    year_ratio = year_ratio.fillna(0)

    return ratio_df, ratio_cols, year_ratio


# =========================
# 3. 시각화 함수들
# =========================

def plot_year_use_heatmap(year_ratio: pd.DataFrame, output_path: str = None):
    """연도별 용도별 토지 구성 비율 히트맵 (도 전체)"""
    plt.figure(figsize=(14, 8))
    sns.heatmap(
        year_ratio.T,
        cmap="viridis",
        annot=False
    )
    plt.title("연도별 용도별 토지 구성 비율 (충청북도 전체)")
    plt.xlabel("연도")
    plt.ylabel("용도")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=200)
    plt.show()


def plot_main_uses_timeseries(year_ratio: pd.DataFrame, main_uses=None):
    """주요 용도 시계열 Plotly 그래프"""
    if main_uses is None:
        # 데이터에 맞게 주요 용도 후보 지정 (실제 컬럼명 확인해서 수정)
        main_uses = [
            "전 면적",
            "답 면적",
            "임야 면적",
            "대 면적",
            "공장용지 면적",
            "도로 면적",
        ]

    fig = go.Figure()
    for col in main_uses:
        if col not in year_ratio.columns:
            print(f"[WARN] 시계열에서 제외 (컬럼 없음): {col}")
            continue
        fig.add_trace(
            go.Scatter(
                x=year_ratio.index,
                y=year_ratio[col],
                mode="lines+markers",
                name=col.replace(" 면적", "")
            )
        )

    fig.update_layout(
        title="연도별 주요 용도 토지 비율 시계열 (충청북도 전체)",
        xaxis_title="연도",
        yaxis_title="비율"    
    )
    fig.show()


def plot_region_donut(ratio_df: pd.DataFrame, year: int, region_name: str):
    """특정 연도·시군구에 대한 토지 용도 도넛 차트"""
    df_year = ratio_df[ratio_df["연도"] == year]
    if df_year.empty:
        print(f"[ERROR] 연도 {year} 데이터가 없습니다.")
        return

    row = df_year[df_year[REGION_COL] == region_name]
    if row.empty:
        print(f"[ERROR] {year}년 데이터에서 '{region_name}'를 찾을 수 없습니다.")
        return

    row = row.iloc[0]
    use_cols = [c for c in df_year.columns if c.endswith("비율")]
    use_names = [c.replace(" 비율", "") for c in use_cols]

    values = row[use_cols].values
    plot_df = pd.DataFrame({
        "용도": use_names,
        "비율": values
    })

    fig = px.pie(
        plot_df,
        names="용도",
        values="비율",
        hole=0.5,
        title=f"{year}년 {region_name} 토지 용도 비율"
    )
    fig.update_layout(font=dict(family="Malgun Gothic", size=12))
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.show()


def plot_correlation_heatmap(ratio_df: pd.DataFrame, ratio_cols):
    """용도별 비율 상관관계 히트맵"""
    corr = ratio_df[ratio_cols].corr()

    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("용도별 비율 상관관계 히트맵")
    plt.tight_layout()
    plt.show()


# =========================
# 4. PCA + KMeans 클러스터링
# =========================

def run_pca_and_cluster(ratio_df: pd.DataFrame, ratio_cols, n_clusters: int = 3):
    """PCA + KMeans 실행 후 pca_df 반환 (PC1, PC2, cluster 포함)"""

    # 0) NaN 방어: 비율 컬럼 NaN → 0
    ratio_df = ratio_df.copy()
    ratio_df[ratio_cols] = ratio_df[ratio_cols].fillna(0)

    X = ratio_df[ratio_cols].values

    # 혹시 모를 NaN 체크
    if np.isnan(X).any():
        print("[WARN] PCA 입력 X에 NaN이 있습니다. NaN을 0으로 대체합니다.")
        X = np.nan_to_num(X, nan=0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    pca_df = ratio_df[["연도", REGION_COL]].copy()
    pca_df["PC1"] = X_pca[:, 0]
    pca_df["PC2"] = X_pca[:, 1]
    pca_df["cluster"] = clusters

    ratio_df["cluster"] = clusters

    print("PCA 설명분산비율:", pca.explained_variance_ratio_)

    return pca_df, ratio_df, pca, scaler, kmeans


def plot_pca_scatter(pca_df: pd.DataFrame, color_by: str = "cluster"):
    """PCA 2차원 평면상 시군구·연도별 패턴 시각화"""
    if color_by not in pca_df.columns:
        print(f"[WARN] color_by='{color_by}'는 pca_df에 없는 컬럼입니다. cluster로 대체.")
        color_by = "cluster"

    fig = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color=color_by,
        hover_data=["연도", REGION_COL, "cluster"],
        title="PCA 기반 시군구·연도별 토지 용도 패턴 분포"
    )
    fig.show()


# =========================
# 5. OSM + H3 위 군집 지도
# =========================

def build_h3_map_for_year(
    ratio_df: pd.DataFrame,
    geojson_path: Path,
    geojson_region_col: str,
    target_year: int,
    h3_resolution: int,
    output_html: str
):
    """
    특정 연도(target_year)에 대해:
        - 시군구 GeoJSON과 ratio_df(cluster 포함)를 merge
        - 각 시군구 polygon을 H3 grid로 polyfill
        - cluster별 색상으로 Folium 지도 생성
    """

    if not geojson_path.exists():
        print(f"[ERROR] GeoJSON 파일 없음: {geojson_path}")
        return

    # 1) GeoJSON 로드
    gdf = gpd.read_file(geojson_path)

    # 2) 대상 연도 데이터
    df_year = ratio_df[ratio_df["연도"] == target_year].copy()
    if df_year.empty:
        print(f"[ERROR] {target_year}년 데이터가 없습니다.")
        return

    # 3) GeoJSON과 merge
    merged = gdf.merge(df_year, left_on=geojson_region_col, right_on=REGION_COL)
    if merged.empty:
        print("[ERROR] GeoJSON과 ratio_df 병합 결과가 비었습니다. 시군구 이름/컬럼명 확인 필요.")
        return

    # 4) 각 시군구 폴리곤을 H3 hex들로 변환
    rows = []
    for idx, row in merged.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        # MultiPolygon 처리
        if geom.geom_type == "MultiPolygon":
            geoms = list(geom.geoms)
        else:
            geoms = [geom]

        for poly in geoms:
            # 🔹 shapely Polygon → GeoJSON-like dict
            geojson = poly.__geo_interface__
            # 또는: from shapely.geometry import mapping; geojson = mapping(poly)

            # 🔹 GeoJSON → LatLngPoly (H3 전용 shape)
            h3shape = h3.geo_to_h3shape(geojson)

            # 🔹 LatLngPoly → H3 셀 리스트
            hexes = h3.h3shape_to_cells(h3shape, res=h3_resolution)

            for h in hexes:
                rows.append({
                    "h3_index": h,
                    REGION_COL: row[REGION_COL],
                    "cluster": int(row["cluster"]) if "cluster" in row else -1
                })

    if not rows:
        print("[ERROR] H3 hex를 생성하지 못했습니다. 해상도/GeoJSON 좌표계 확인 필요 (EPSG:4326 권장).")
        return

    hex_df = pd.DataFrame(rows).drop_duplicates("h3_index")

    # -------- Folium 지도 생성 --------
    m = folium.Map(location=[36.8, 127.8], zoom_start=9, tiles="OpenStreetMap")

    base_colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]

    def get_color(c):
        if c < 0:
            return "#000000"
        return base_colors[c % len(base_colors)]

    for _, r in hex_df.iterrows():
        h = r["h3_index"]

        # 🔹 H3 cell → 경계 좌표 (lat, lon)
        boundary = h3.cell_to_boundary(h)
        
        folium.Polygon(
            locations=[(lat, lng) for (lat, lng) in boundary],
            color=None,
            fill=True,
            fill_opacity=0.6,
            fill_color=get_color(r["cluster"]),
            popup=f"{r[REGION_COL]} | cluster {r['cluster']}"
        ).add_to(m)

    m.save(output_html)
    print(f"[INFO] H3 클러스터 지도 저장 완료: {output_html}")

# =========================
# 6. 메인 실행부
# =========================

def main():
    # -------- 6-1. 데이터 로딩 --------
    print("[INFO] 연도별 지적통계 CSV 로딩 중...")
    raw = load_all_years(DATA_DIR, YEAR_FILES)
    print(f"[INFO] raw shape: {raw.shape}")

    # -------- 6-2. 면적/비율 데이터 구성 --------
    area_cols = get_area_columns(raw)
    print(f"[INFO] 면적 컬럼 수: {len(area_cols)}개")
    print("       예시:", area_cols[:8])

    ratio_df, ratio_cols, year_ratio = build_ratio_df(raw, area_cols)
    print(f"[INFO] ratio_df shape: {ratio_df.shape}")
    print(f"[INFO] ratio_cols 수: {len(ratio_cols)}개")

    # -------- 6-3. 시각화 ① 연도별·용도별 히트맵 --------
    plot_year_use_heatmap(year_ratio)

    # -------- 6-4. 시각화 ② 주요 용도 시계열 --------
    plot_main_uses_timeseries(year_ratio)

    # -------- 6-5. 시각화 ③ 2025년 특정 시군구 도넛 차트 --------
    # 예시: 청주 상당구 (실제 있는 이름으로 바꿔서 테스트)
    example_region = ratio_df[REGION_COL].unique()[0]
    print(f"[INFO] 도넛 차트 예시 시군구: {example_region}")
    plot_region_donut(ratio_df, TARGET_YEAR, example_region)

    # -------- 6-6. 상관관계 히트맵 --------
    plot_correlation_heatmap(ratio_df, ratio_cols)

    # -------- 6-7. PCA + KMeans 클러스터링 --------
    pca_df, ratio_df_clustered, pca, scaler, kmeans = run_pca_and_cluster(
        ratio_df,
        ratio_cols,
        n_clusters=3  # 필요시 4~5로 바꿔가며 실험
    )

    # PCA 산점도 (cluster 기준 색)
    plot_pca_scatter(pca_df, color_by="cluster")

    # 시군구 이름 기준 색도 보고 싶으면:
    # plot_pca_scatter(pca_df, color_by=REGION_COL)

    # -------- 6-8. OSM + H3 클러스터 지도 --------
    print("[INFO] OSM + H3 지도로 군집 시각화 중...")
    build_h3_map_for_year(
        ratio_df_clustered,
        GEOJSON_PATH,
        GEOJSON_REGION_COL,
        TARGET_YEAR,
        H3_RESOLUTION,
        H3_MAP_OUTPUT
    )

    print("[INFO] 전체 파이프라인 완료.")


if __name__ == "__main__":
    main()
