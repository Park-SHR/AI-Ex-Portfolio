from __future__ import annotations
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from .evaluate import regression_metrics, congestion_metrics


def train_tabular_models(ds: dict, output_model_dir) -> tuple[pd.DataFrame, dict]:
    models = {
        "Persistence_CurrentCount": None,
        "RandomForest": RandomForestRegressor(n_estimators=80, max_depth=10, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "MLPRegressor": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=250, random_state=42, early_stopping=True),
    }
    rows = []
    trained = {}
    for name, model in models.items():
        if name == "Persistence_CurrentCount":
            # First standardized feature is current yt_count. Convert back approximately using saved mean/std.
            idx = 0
            pred = ds["X_test"][:, idx] * ds["std"][0, idx] + ds["mean"][0, idx]
        else:
            model.fit(ds["X_train"], ds["y_train"])
            pred = model.predict(ds["X_test"])
            trained[name] = model
            joblib.dump(model, output_model_dir / f"{name}.joblib")
        # Future count prediction can be converted to congestion-like binary by normalizing with a practical threshold.
        reg = regression_metrics(ds["y_test"], pred)
        pred_score = np.clip(pred / max(np.percentile(ds["y_train"], 90), 1.0), 0, 1)
        true_score = np.clip(ds["y_test"] / max(np.percentile(ds["y_train"], 90), 1.0), 0, 1)
        cong = congestion_metrics(true_score, pred_score, threshold=0.55)
        rows.append({"model": name, **reg, **cong})
    return pd.DataFrame(rows), trained


def train_route_choice_model(task_routes: pd.DataFrame, output_model_dir) -> tuple[pd.DataFrame, RandomForestClassifier | None]:
    if task_routes.empty or task_routes["route_rank"].nunique() < 2:
        return pd.DataFrame([{"model":"RouteChoiceRF", "Top1Acc":0.0, "Top3Acc":0.0, "note":"not enough route-rank diversity"}]), None
    df = task_routes.copy()
    df["hour"] = (df["planned_start_s"] / 3600.0) % 24
    df["is_peak"] = ((df["hour"].between(7, 10)) | (df["hour"].between(13, 16))).astype(int)
    cat = pd.get_dummies(df[["sop_type", "origin_node", "dest_node"]], sparse=False)
    num = df[["priority", "demand_multiplier", "path_length_m", "base_time_s", "hour", "is_peak"]].astype(float)
    X = pd.concat([num.reset_index(drop=True), cat.reset_index(drop=True)], axis=1)
    y = df["route_rank"].astype(int)
    order = np.argsort(df["planned_start_s"].values)
    X = X.iloc[order].reset_index(drop=True)
    y = y.iloc[order].reset_index(drop=True)
    split = int(len(X) * 0.75)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    clf = RandomForestClassifier(n_estimators=120, max_depth=12, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    proba = clf.predict_proba(X_test)
    labels = clf.classes_
    top1 = accuracy_score(y_test, pred)
    # Manual top-k because sklearn top_k_accuracy_score can be misleading when
    # k >= n_classes or when labels are not zero-based consecutive integers.
    k = min(3, len(labels))
    topk_label_matrix = labels[np.argsort(proba, axis=1)[:, -k:]]
    top3 = float(np.mean([yt in row for yt, row in zip(y_test.values, topk_label_matrix)]))
    joblib.dump({"model": clf, "columns": X.columns.tolist()}, output_model_dir / "RouteChoiceRF.joblib")
    return pd.DataFrame([{"model":"RouteChoiceRF", "Top1Acc":float(top1), "Top3Acc":float(top3), "classes":str(labels.tolist())}]), clf
