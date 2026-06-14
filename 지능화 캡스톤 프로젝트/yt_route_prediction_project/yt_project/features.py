from __future__ import annotations
import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "yt_count", "pass_count", "avg_speed_mps", "wait_sum_s", "avg_travel_time_s",
    "occupancy", "wait_norm", "speed_drop", "congestion_score",
    "length_m", "speed_limit_mps", "capacity", "base_time_s",
]


def make_future_labels(link_time: pd.DataFrame, horizon_bins: int) -> pd.DataFrame:
    df = link_time.copy().sort_values(["link_id", "bin"])
    df["y_future_count"] = df.groupby("link_id")["yt_count"].shift(-horizon_bins).fillna(0)
    df["y_future_congestion"] = df.groupby("link_id")["congestion_score"].shift(-horizon_bins).fillna(0)
    return df.sort_values(["bin", "link_id"]).reset_index(drop=True)


def standardize_train_valid_test(X_train, X_valid, X_test):
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (X_train - mean) / std, (X_valid - mean) / std, (X_test - mean) / std, mean, std


def make_tabular_dataset(link_time: pd.DataFrame, horizon_bins: int, congestion_threshold: float = 0.55):
    df = make_future_labels(link_time, horizon_bins)
    max_bin = int(df["bin"].max()) - horizon_bins
    df = df[df["bin"] <= max_bin].copy()
    X = df[FEATURE_COLUMNS].astype(float).values
    y_reg = df["y_future_count"].astype(float).values
    y_cong = (df["y_future_congestion"] >= congestion_threshold).astype(int).values
    bins = df["bin"].values
    train_mask = bins < np.quantile(bins, 0.70)
    valid_mask = (bins >= np.quantile(bins, 0.70)) & (bins < np.quantile(bins, 0.85))
    test_mask = bins >= np.quantile(bins, 0.85)
    X_train, X_valid, X_test, mean, std = standardize_train_valid_test(X[train_mask], X[valid_mask], X[test_mask])
    return {
        "df": df,
        "X_train": X_train,
        "X_valid": X_valid,
        "X_test": X_test,
        "y_train": y_reg[train_mask],
        "y_valid": y_reg[valid_mask],
        "y_test": y_reg[test_mask],
        "y_cong_train": y_cong[train_mask],
        "y_cong_valid": y_cong[valid_mask],
        "y_cong_test": y_cong[test_mask],
        "test_df": df[test_mask].copy(),
        "mean": mean,
        "std": std,
    }


def make_tensor_dataset(link_time: pd.DataFrame, link_ids: list[str], horizon_bins: int, window_size: int):
    df = make_future_labels(link_time, horizon_bins)
    link_to_idx = {lid: i for i, lid in enumerate(link_ids)}
    n_links = len(link_ids)
    n_bins = int(df["bin"].max()) + 1
    n_feat = len(FEATURE_COLUMNS)
    tensor = np.zeros((n_bins, n_links, n_feat), dtype=np.float32)
    y_count = np.zeros((n_bins, n_links), dtype=np.float32)
    y_cong = np.zeros((n_bins, n_links), dtype=np.float32)
    sub = df[df["link_id"].isin(link_ids)].copy()
    for row in sub.itertuples(index=False):
        b = int(row.bin)
        j = link_to_idx[row.link_id]
        tensor[b, j, :] = np.array([float(getattr(row, c)) for c in FEATURE_COLUMNS], dtype=np.float32)
        y_count[b, j] = float(row.y_future_count)
        y_cong[b, j] = float(row.y_future_congestion)

    # standardize features using train-time frames only.
    train_end = int(n_bins * 0.70)
    mean = tensor[:train_end].reshape(-1, n_feat).mean(axis=0, keepdims=True)
    std = tensor[:train_end].reshape(-1, n_feat).std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    tensor = (tensor - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)

    X_seq, y_seq, y_cong_seq, target_bins = [], [], [], []
    last_start = n_bins - horizon_bins
    for t in range(window_size - 1, last_start):
        X_seq.append(tensor[t - window_size + 1:t + 1])
        y_seq.append(y_count[t])
        y_cong_seq.append(y_cong[t])
        target_bins.append(t)
    X_seq = np.stack(X_seq).astype(np.float32)
    y_seq = np.stack(y_seq).astype(np.float32)
    y_cong_seq = np.stack(y_cong_seq).astype(np.float32)
    target_bins = np.array(target_bins)
    train_mask = target_bins < np.quantile(target_bins, 0.70)
    valid_mask = (target_bins >= np.quantile(target_bins, 0.70)) & (target_bins < np.quantile(target_bins, 0.85))
    test_mask = target_bins >= np.quantile(target_bins, 0.85)
    return {
        "X_train": X_seq[train_mask], "y_train": y_seq[train_mask],
        "X_valid": X_seq[valid_mask], "y_valid": y_seq[valid_mask],
        "X_test": X_seq[test_mask], "y_test": y_seq[test_mask],
        "y_cong_test": y_cong_seq[test_mask], "target_bins_test": target_bins[test_mask],
        "mean": mean, "std": std, "link_ids": link_ids,
    }
