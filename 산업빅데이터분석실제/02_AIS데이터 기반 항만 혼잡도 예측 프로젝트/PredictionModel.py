import json
from pathlib import Path
from datetime import datetime

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
)

import plotly.express as px
from math import sqrt


# ============================================================================
# 1. 데이터 로드: ais_results 폴더 내 aisResult*.json → DataFrame
# ============================================================================

def load_all_ais_results(folder: str = "./ais_results") -> pd.DataFrame:
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
    R = 6371.0  # 지구 반경 (km)
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def add_distance_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    BEFORE_LAT/LON ~ AFTER_LAT/LON 사이의 이동 거리(km)를 distance_km 컬럼으로 추가
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
    """
    MMSI별로 Feature를 집계:
      - total_distance: 전체 이동 거리 합(km)
      - total_time: 전체 MOVINGTIME 합(초)
      - mean_sog, std_sog, max_sog
      - dwell_hours: total_time(초) -> 시간 단위
    """
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
# 3. Clustering: 선박 동작 패턴 K-Means 클러스터링
# ============================================================================

def run_clustering(feat_df: pd.DataFrame, n_clusters: int = 4):
    """
    K-Means 클러스터링 수행하고 cluster 레이블 반환
    """
    features = ["total_distance", "mean_sog", "std_sog", "max_sog", "dwell_hours"]

    X = feat_df[features].fillna(0.0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    clusters = kmeans.fit_predict(X_scaled)

    feat_df = feat_df.copy()
    feat_df["cluster"] = clusters

    print("[INFO] K-Means 클러스터링 완료")
    return feat_df, kmeans, scaler


# ============================================================================
# 4. Classification: 선박 클러스터 유형 예측 (Logistic vs RandomForest)
# ============================================================================

def run_cluster_classification(feat_df: pd.DataFrame):
    """
    K-Means로 얻은 cluster를 타겟으로 Classification 수행
    LogisticRegression vs RandomForestClassifier 비교
    """
    features = ["total_distance", "mean_sog", "std_sog", "max_sog", "dwell_hours"]
    X = feat_df[features].fillna(0.0)
    y = feat_df["cluster"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Logistic Regression
    log_clf = LogisticRegression(max_iter=1000, multi_class="auto")
    log_clf.fit(X_train, y_train)
    pred_log = log_clf.predict(X_test)

    # RandomForest Classifier
    rf_clf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf_clf.fit(X_train, y_train)
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
# 5. Regression: 체류시간(dwell_hours) 예측 (Linear vs RandomForestRegressor)
# ============================================================================

def run_dwell_regression(feat_df: pd.DataFrame):
    """
    dwell_hours(선박 체류시간)을 예측하는 회귀 모델
    - Linear Regression
    - RandomForest Regressor
    - RMSE를 sqrt(MSE) 로 직접 계산하여 sklearn 버전 호환 문제 방지
    """

    features = ["total_distance", "mean_sog", "std_sog", "max_sog"]
    X = feat_df[features].fillna(0.0)
    y = feat_df["dwell_hours"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 모델 선언
    lr = LinearRegression()
    rf = RandomForestRegressor(n_estimators=200, random_state=42)

    # 모델 학습
    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    # 예측
    pred_lr = lr.predict(X_test)
    pred_rf = rf.predict(X_test)

    # ===========================
    # 🔥 sklearn 구버전 대응 RMSE 계산
    # ===========================
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
        "rmse_rf": rmse_rf
    }

# ============================================================================
# 6. 혼잡도(congestion) 라벨 생성 + Classification
# ============================================================================

def run_congestion_classification(feat_df: pd.DataFrame):
    """
    dwell_hours가 상위 25% 이상이면 1(혼잡), 아니면 0(비혼잡)으로 라벨링
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
    }


# ============================================================================
# 7. Plotly 시각화
# ============================================================================

def visualize_clusters(feat_df: pd.DataFrame):
    """
    mean_sog vs total_distance 를 클러스터 색으로 시각화
    """
    fig = px.scatter(
        feat_df,
        x="mean_sog",
        y="total_distance",
        color="cluster",
        hover_data=["mmsi", "dwell_hours"],
        title="선박 동작 패턴 클러스터링 결과",
    )
    fig.show()


def visualize_dwell_distribution(feat_df: pd.DataFrame):
    """
    dwell_hours 분포 히스토그램
    """
    fig = px.histogram(
        feat_df,
        x="dwell_hours",
        nbins=30,
        title="선박 체류시간(dwell_hours) 분포",
    )
    fig.show()


# ============================================================================
# 8. 메인 파이프라인
# ============================================================================

def main():
    # 1) 데이터 로드
    df = load_all_ais_results("./ais_results")

    # 2) 거리 Feature 추가
    df = add_distance_feature(df)

    # 3) MMSI 레벨 Feature DataFrame 생성
    feat_df = build_mmsi_level_features(df)

    print("\n[INFO] MMSI 레벨 Feature DataFrame Preview:")
    print(feat_df.head())

    # 4) 클러스터링
    feat_df, kmeans_model, scaler = run_clustering(feat_df, n_clusters=4)

    # 5) 클러스터 Classification
    cls_result = run_cluster_classification(feat_df)

    # 6) Dwell Hours Regression
    reg_result = run_dwell_regression(feat_df)

    # 7) 혼잡도 Classification
    cong_result = run_congestion_classification(feat_df)

    # 8) 시각화
    visualize_clusters(feat_df)
    visualize_dwell_distribution(feat_df)

    print("\n[INFO] 전체 파이프라인 실행 완료")


if __name__ == "__main__":
    main()
