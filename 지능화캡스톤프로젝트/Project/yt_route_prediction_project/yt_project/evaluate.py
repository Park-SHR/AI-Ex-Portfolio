from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_recall_fscore_support


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def congestion_metrics(y_true_score, y_pred_score, threshold: float = 0.55) -> dict:
    y_true = (np.asarray(y_true_score).reshape(-1) >= threshold).astype(int)
    y_pred = (np.asarray(y_pred_score).reshape(-1) >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    inter = np.logical_and(y_true == 1, y_pred == 1).sum()
    union = np.logical_or(y_true == 1, y_pred == 1).sum()
    return {
        "Congestion_Precision": float(p),
        "Congestion_Recall": float(r),
        "Congestion_F1": float(f1),
        "Heatmap_IoU": float(inter / union) if union > 0 else 0.0,
    }


def route_choice_metrics(task_routes: pd.DataFrame) -> dict:
    if task_routes.empty or "route_rank" not in task_routes:
        return {"Top1_Dijkstra": 0.0, "Top3_KShortest": 0.0, "Top5_KShortest": 0.0, "NonShortestRate": 0.0}
    return {
        "Top1_Dijkstra": float((task_routes["route_rank"] == 1).mean()),
        "Top3_KShortest": float((task_routes["route_rank"] <= 3).mean()),
        "Top5_KShortest": float((task_routes["route_rank"] <= 5).mean()),
        "NonShortestRate": float((task_routes["route_rank"] != 1).mean()),
    }
