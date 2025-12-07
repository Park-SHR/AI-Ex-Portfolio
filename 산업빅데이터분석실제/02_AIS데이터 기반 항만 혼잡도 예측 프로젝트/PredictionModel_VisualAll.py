import json
from pathlib import Path
from math import sqrt

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    confusion_matrix,
)

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff


# ============================================================================
# 1. 데이터 로드: ais_results 폴더 내 aisResult*.json → DataFrame
# ============================================================================

def load_all_ais_results(folder: str = "./ais_results") -> pd.DataFrame:
    """
    ais_results 폴더 내의 aisResult*.json 파일들을 모두 읽어
    PredResult를 row 단위로 펼쳐서 하나의 DataFrame으로 만든다.
    """
    folder_path = Path(folder)
    rows = []

    json_files = sorted(folder_path.glob("aisResult*.json"))
    print(f"[INFO] 총 {len(json_files)}개 JSON 파일 발견")

    for file in json_files:
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] JSON 파싱 실패: {file.name} / {e}")
            continue

        mmsi = data.get("mmsi")
        pred_list = data.get("PredResult", [])

        if not mmsi or not isinstance(pred_list, list) or len(pred_list) == 0:
            print(f"[WARN] mmsi 또는 PredResult 이상: {file.name}")
            continue

        for item in pred_list:
            rows.append({
                "mmsi": str(mmsi),
                "seq": item.get("SEQ"),
                "before_lat": item.get("BEFORE_LAT"),
                "before_lon": item.get("BEFORE_LON"),
                "after_lat": item.get("AFTER_LAT"),
                "after_lon": item.get("AFTER_LON"),
                "sog": item.get("SOG"),
                "moving_time": item.get("MOVINGTIME"),
                "arrival_time": item.get("ARRIVALTIME"),
            })

    df = pd.DataFrame(rows)
    print(f"[INFO] 로드 완료: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ============================================================================
# 2. Feature Engineering: 거리 계산(Haversine), MMSI 단위 집계 등
# ============================================================================

def haversine(lat1, lon1, lat2, lon2):
    """
    위경도 두 점 사이의 거리를 km 단위로 계산 (Haversine 공식)
    """
    R = 6371.0  # km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def add_distance_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    BEFORE_LAT/LON ~ AFTER_LAT/LON 사이 이동 거리를 distance_km 컬럼으로 추가
    """
    df = df.copy()
    df["distance_km"] = haversine(
        df["before_lat"].astype(float),
        df["before_lon"].astype(float),
        df["after_lat"].astype(float),
        df["after_lon"].astype(float),
    )
    return df


def build_mmsi_level_features(df: pd.DataFrame) -> pd.DataFrame:

    agg = df.groupby("mmsi").agg({
        "distance_km": "sum",
        "moving_time": "sum",
        "sog": ["mean", "std", "max"],
    }).reset_index()

    agg.columns = [
        "mmsi",
        "total_distance",
        "total_time",
        "mean_sog",
        "std_sog",
        "max_sog",
    ]
    agg["dwell_hours"] = agg["total_time"] / 3600.0
    return agg


# ============================================================================
# 3. Clustering (K-Means)
# ============================================================================

def run_clustering(feat_df: pd.DataFrame, n_clusters: int = 4):
    """
    MMSI별 Feature에 대해 K-Means 클러스터링 수행
    """
    features = ["total_distance", "mean_sog", "std_sog", "max_sog", "dwell_hours"]
    X = feat_df[features].fillna(0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # n_init="auto" 대신 10으로 고정 (구버전 sklearn 호환)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    feat_df = feat_df.copy()
    feat_df["cluster"] = clusters
    print("[INFO] K-Means 클러스터링 완료")

    return feat_df, kmeans, scaler


# ============================================================================
# 4. Classification: 클러스터 레이블 예측
# ============================================================================

def run_cluster_classification(feat_df: pd.DataFrame):
    """
    K-Means로 생성한 cluster 레이블을 타겟으로 Classification 수행
    LogisticRegression vs RandomForestClassifier 비교
    """
    features = ["total_distance", "mean_sog", "std_sog", "max_sog", "dwell_hours"]
    X = feat_df[features].fillna(0.0)
    y = feat_df["cluster"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    log_clf = LogisticRegression(max_iter=1000)
    rf_clf = RandomForestClassifier(n_estimators=200, random_state=42)

    log_clf.fit(X_train, y_train)
    rf_clf.fit(X_train, y_train)

    pred_log = log_clf.predict(X_test)
    pred_rf = rf_clf.predict(X_test)

    print("\n[Classification: Cluster 예측]")
    print("  Logistic Accuracy:", accuracy_score(y_test, pred_log))
    print("  RandomForest Accuracy:", accuracy_score(y_test, pred_rf))
    print("  Logistic F1 (macro):", f1_score(y_test, pred_log, average="macro"))
    print("  RandomForest F1 (macro):", f1_score(y_test, pred_rf, average="macro"))

    return {
        "log_clf": log_clf,
        "rf_clf": rf_clf,
        "X_test": X_test,
        "y_test": y_test,
        "pred_log": pred_log,
        "pred_rf": pred_rf,
    }


# ============================================================================
# 5. Regression: 체류시간(dwell_hours) 예측 (구버전 sklearn 호환 RMSE)
# ============================================================================

def run_dwell_regression(feat_df: pd.DataFrame):
    """
    dwell_hours(선박 체류시간)을 예측하는 회귀 모델
    - Linear Regression
    - RandomForest Regressor
    - RMSE = sqrt(MSE) 로 직접 계산 (squared=False 안 씀)
    """
    features = ["total_distance", "mean_sog", "std_sog", "max_sog"]
    X = feat_df[features].fillna(0.0)
    y = feat_df["dwell_hours"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    lr = LinearRegression()
    rf = RandomForestRegressor(n_estimators=200, random_state=42)

    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    pred_lr = lr.predict(X_test)
    pred_rf = rf.predict(X_test)

    mse_lr = mean_squared_error(y_test, pred_lr)
    mse_rf = mean_squared_error(y_test, pred_rf)
    rmse_lr = sqrt(mse_lr)
    rmse_rf = sqrt(mse_rf)

    mae_lr = mean_absolute_error(y_test, pred_lr)
    mae_rf = mean_absolute_error(y_test, pred_rf)

    print("\n[Regression: Dwell Hours 예측]")
    print(f"  LinearRegression RMSE : {rmse_lr:.4f}, MAE: {mae_lr:.4f}")
    print(f"  RandomForest RMSE     : {rmse_rf:.4f}, MAE: {mae_rf:.4f}")

    return {
        "lr": lr,
        "rf": rf,
        "X_test": X_test,
        "y_test": y_test,
        "pred_lr": pred_lr,
        "pred_rf": pred_rf,
        "rmse_lr": rmse_lr,
        "rmse_rf": rmse_rf,
    }


# ============================================================================
# 6. 혼잡도 Classification
# ============================================================================

def run_congestion_classification(feat_df: pd.DataFrame):
    """
    dwell_hours 상위 25% 이상이면 1(혼잡), 나머지는 0(비혼잡)으로 정의하고
    RandomForest로 분류
    """
    feat_df = feat_df.copy()
    thr = feat_df["dwell_hours"].quantile(0.75)
    feat_df["congestion"] = (feat_df["dwell_hours"] >= thr).astype(int)

    features = ["total_distance", "mean_sog", "std_sog", "max_sog"]
    X = feat_df[features].fillna(0.0)
    y = feat_df["congestion"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf_clf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf_clf.fit(X_train, y_train)
    pred = rf_clf.predict(X_test)

    print("\n[Classification: 항만 혼잡도 예측]")
    print("  Accuracy:", accuracy_score(y_test, pred))
    print("  F1:", f1_score(y_test, pred))

    return {
        "rf_clf": rf_clf,
        "X_test": X_test,
        "y_test": y_test,
        "pred": pred,
        "threshold": thr,
        "feat_df": feat_df,
    }


# ============================================================================
# 7. 클러스터 요약 출력
# ============================================================================

def print_cluster_summary(feat_df: pd.DataFrame):
    summary = (
        feat_df.groupby("cluster")
        .agg(
            total_distance=("total_distance", "mean"),
            mean_sog=("mean_sog", "mean"),
            dwell_hours=("dwell_hours", "mean"),
            ship_count=("mmsi", "count"),
        )
        .reset_index()
    )
    print("\n[클러스터 요약 통계]")
    print(summary)


# ============================================================================
# 8. 시각화 함수들 (Plotly)
# ============================================================================

def plot_movement_scatter(df: pd.DataFrame):
    """
    SEQ별 after 좌표 기준 선박 위치 분포
    """
    sample_df = df.sample(min(5000, len(df)), random_state=42)

    fig = px.scatter_mapbox(
        sample_df,
        lat="after_lat",
        lon="after_lon",
        color="mmsi",
        zoom=3,
        height=700,
        title="AIS 정제 데이터 – 선박 이동 위치 분포",
        hover_data=["mmsi", "seq", "sog"]
    )

    fig.update_layout(
        mapbox_style="open-street-map", 
        margin={"r":0, "t":40, "l":0, "b":0}
    )

    fig.show()

def plot_dwell_hist(feat_df: pd.DataFrame):
    """
    dwell_hours 분포 히스토그램
    """
    fig = px.histogram(
        feat_df,
        x="dwell_hours",
        nbins=30,
        title="선박 체류시간(dwell_hours) 분포",
        labels={"dwell_hours": "Dwell Time (hours)"},
    )
    fig.show()


def plot_cluster_scatter(feat_df: pd.DataFrame):
    """
    mean_sog vs total_distance 클러스터링 결과 산점도
    """
    fig = px.scatter(
        feat_df,
        x="mean_sog",
        y="total_distance",
        color="cluster",
        hover_data=["mmsi", "dwell_hours"],
        title="선박 동작 패턴 클러스터링 결과 (K-Means)",
        labels={
            "mean_sog": "평균 속도(knots)",
            "total_distance": "총 이동 거리(km)",
            "cluster": "클러스터",
        },
    )
    fig.show()


def plot_congestion_scatter(cong_feat_df: pd.DataFrame):
    """
    혼잡도 라벨에 따른 mean_sog vs dwell_hours 분포
    """
    fig = px.scatter(
        cong_feat_df,
        x="mean_sog",
        y="dwell_hours",
        color="congestion",
        hover_data=["mmsi", "total_distance"],
        title="혼잡도 라벨에 따른 선박 체류시간-속도 분포",
        labels={"congestion": "혼잡도(0:Low, 1:High)"},
    )
    fig.show()


def plot_cluster_confusion_matrix(cls_result: dict):
    """
    클러스터 예측 RF 모델 Confusion Matrix
    """
    y_test = cls_result["y_test"]
    pred_rf = cls_result["pred_rf"]

    cm = confusion_matrix(y_test, pred_rf)
    labels = sorted(list(set(y_test) | set(pred_rf)))

    fig = ff.create_annotated_heatmap(
        z=cm,
        x=[f"Pred {l}" for l in labels],
        y=[f"True {l}" for l in labels],
        colorscale="Blues",
        showscale=True,
    )
    fig.update_layout(
        title="클러스터 분류(RF) Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="True",
    )
    fig.show()


def plot_regression_scatter(reg_result: dict):
    """
    체류시간 실제값 vs 예측값 (RF Regression)
    """
    y_test = reg_result["y_test"]
    pred_rf = reg_result["pred_rf"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_test,
            y=pred_rf,
            mode="markers",
            name="예측값(RF)",
        )
    )

    line_min = float(min(y_test.min(), pred_rf.min()))
    line_max = float(max(y_test.max(), pred_rf.max()))
    fig.add_trace(
        go.Scatter(
            x=[line_min, line_max],
            y=[line_min, line_max],
            mode="lines",
            name="y = x",
        )
    )

    fig.update_layout(
        title="체류시간 예측 – 실제값 vs 예측값 (RandomForestRegressor)",
        xaxis_title="실제 Dwell Hours",
        yaxis_title="예측 Dwell Hours",
    )
    fig.show()


def plot_feature_importance_reg(reg_result: dict):
    """
    RF 회귀 모델 Feature 중요도
    """
    rf = reg_result["rf"]
    feature_names = ["total_distance", "mean_sog", "std_sog", "max_sog"]
    importances = rf.feature_importances_

    imp_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)

    fig = px.bar(
        imp_df,
        x="feature",
        y="importance",
        title="RandomForest Feature 중요도 (체류시간 예측)",
        labels={"importance": "Importance"},
    )
    fig.show()


def plot_feature_importance_cong(cong_result: dict):
    """
    RF 혼잡도 분류 모델 Feature 중요도
    """
    rf_cong = cong_result["rf_clf"]
    feature_names = ["total_distance", "mean_sog", "std_sog", "max_sog"]
    importances = rf_cong.feature_importances_

    imp_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)

    fig = px.bar(
        imp_df,
        x="feature",
        y="importance",
        title="RandomForest Feature 중요도 (혼잡도 예측)",
        labels={"importance": "Importance"},
    )
    fig.show()

def make_last_position_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    각 mmsi별로 마지막 seq의 after_lat/after_lon을 대표 위치로 사용
    raw_df 컬럼 예: mmsi, seq, after_lat, after_lon ...
    """
    # seq 기준으로 정렬
    raw_df_sorted = raw_df.sort_values(["mmsi", "seq"])

    # 각 mmsi의 마지막 row만 선택
    last_pos = (
        raw_df_sorted
        .groupby("mmsi")
        .tail(1)[["mmsi", "after_lat", "after_lon"]]
        .rename(columns={"after_lat": "lat", "after_lon": "lon"})
    )
    return last_pos

def export_congestion_json(feat_df: pd.DataFrame,
                           last_pos_df: pd.DataFrame,
                           out_path: str):
    """
    feat_df + last_pos_df 를 합쳐서
    지도 시각화를 위한 최종 JSON 파일 생성
    """
    # mmsi 기준으로 merge
    merged = pd.merge(feat_df, last_pos_df, on="mmsi", how="inner")

    records = []
    for _, row in merged.iterrows():
        try:
            records.append({
                "mmsi": str(row["mmsi"]),
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "cluster": int(row["cluster"]),
                "congestion": int(row["congestion"]),
                "dwell_hours": float(row["dwell_hours"]),
                "total_distance": float(row["total_distance"]),
                "mean_sog": float(row["mean_sog"]),
            })
        except Exception:
            # 타입 꼬인 애들은 스킵
            continue

    data = {"ships": records}

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[INFO] JSON 저장 완료 → {out_path}")


# ============================================================================
# 9. 메인 파이프라인 + 시각화 호출
# ============================================================================

def main():
    # 1) 데이터 로드
    df = load_all_ais_results("./ais_results")

    # 2) 거리 Feature 추가
    df = add_distance_feature(df)

    # 3) MMSI 레벨 Feature DF
    feat_df = build_mmsi_level_features(df)
    print("\n[INFO] MMSI 레벨 Feature DataFrame Preview:")
    print(feat_df.head())

    # 4) 클러스터링
    feat_df, kmeans_model, scaler = run_clustering(feat_df, n_clusters=4)

    # 5) Classification: 클러스터 예측
    cls_result = run_cluster_classification(feat_df)

    # 6) Regression: Dwell Hours 예측
    reg_result = run_dwell_regression(feat_df)

    # 7) 혼잡도 Classification
    cong_result = run_congestion_classification(feat_df)
    cong_feat_df = cong_result["feat_df"]

    # 8) 클러스터 요약 출력
    print_cluster_summary(feat_df)

    # 9) 시각화 (슬라이드용 그림들)
    plot_movement_scatter(df)
    plot_dwell_hist(feat_df)
    plot_cluster_scatter(feat_df)
    plot_congestion_scatter(cong_feat_df)
    plot_cluster_confusion_matrix(cls_result)
    plot_regression_scatter(reg_result)
    plot_feature_importance_reg(reg_result)
    plot_feature_importance_cong(cong_result)

    print("\n[INFO] 전체 파이프라인 + 시각화 완료")
    
    # 10) 각 선박의 최종 위치 + 혼잡도/클러스터 JSON Export
    last_pos_df = make_last_position_df(df)
    export_congestion_json(
        feat_df=cong_feat_df,
        last_pos_df=last_pos_df,
        out_path="./ais_results/ship_congestion_result.json"
    )


if __name__ == "__main__":
    main()