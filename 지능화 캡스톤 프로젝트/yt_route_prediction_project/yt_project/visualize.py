from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_graph(node_master: pd.DataFrame, link_master: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    plt.figure(figsize=(13, 7))
    type_markers = {"QUAY_ANCHOR": "o", "QC": "s", "ROAD_INTERSECTION": "^", "YARD_BLOCK": "D", "GATE": "*"}
    for row in link_master.itertuples(index=False):
        a = node_master[node_master.node_id == row.from_node].iloc[0]
        b = node_master[node_master.node_id == row.to_node].iloc[0]
        plt.plot([a.x, b.x], [a.y, b.y], linewidth=0.5, alpha=0.25)
    for node_type, group in node_master.groupby("node_type"):
        plt.scatter(group["x"], group["y"], label=node_type, s=35, marker=type_markers.get(node_type, "o"))
    plt.title("Generated Port Road Graph from TOS Anchors")
    plt.xlabel("TOS/DT X")
    plt.ylabel("TOS/DT Y")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_model_comparison(metrics: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    if metrics.empty or "MAE" not in metrics:
        return
    m = metrics.dropna(subset=["MAE"]).copy()
    plt.figure(figsize=(10, 5))
    plt.bar(m["model"], m["MAE"])
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Future Link Count MAE")
    plt.title("Model Comparison: Future Link Count Prediction")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_pred_actual(y_true, y_pred, output_path: str | Path, title: str = "Prediction vs Actual") -> None:
    output_path = Path(output_path)
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    sample = min(500, len(y_true))
    plt.figure(figsize=(10, 5))
    plt.plot(y_true[:sample], label="actual")
    plt.plot(y_pred[:sample], label="predicted", alpha=0.75)
    plt.title(title)
    plt.ylabel("Future Link Count")
    plt.xlabel("Sample")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_congestion_heatmap(link_time: pd.DataFrame, output_path: str | Path, max_links: int = 80) -> None:
    output_path = Path(output_path)
    pivot = link_time.pivot_table(index="link_id", columns="bin", values="congestion_score", aggfunc="mean").fillna(0)
    pivot = pivot.iloc[:max_links]
    plt.figure(figsize=(13, 7))
    plt.imshow(pivot.values, aspect="auto", origin="lower")
    plt.colorbar(label="Congestion Score")
    plt.title("Link-Time Congestion Heatmap from SOP Simulation")
    plt.xlabel("Time Bin")
    plt.ylabel("Link Index")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def export_unreal_prediction_json(link_master: pd.DataFrame, node_master: pd.DataFrame, link_time: pd.DataFrame, output_path: str | Path, top_bins: int = 5) -> None:
    """Export a simple JSON that can later be adapted to META K-Port/Unreal.

    This file includes link polylines and congestion attributes by time bin.
    """
    import json
    output_path = Path(output_path)
    node_pos = {r.node_id: {"x": float(r.x), "y": float(r.y), "z": 0.0} for r in node_master.itertuples(index=False)}
    recent_bins = sorted(link_time["bin"].unique())[:top_bins]
    attrs = link_time[link_time["bin"].isin(recent_bins)]
    attr_map = {(r.link_id, int(r.bin)): {"yt_count": float(r.yt_count), "congestion_score": float(r.congestion_score)} for r in attrs.itertuples(index=False)}
    links = []
    for row in link_master.itertuples(index=False):
        time_attrs = {str(b): attr_map.get((row.link_id, int(b)), {"yt_count": 0, "congestion_score": 0}) for b in recent_bins}
        links.append({
            "link_id": row.link_id,
            "from_node": row.from_node,
            "to_node": row.to_node,
            "polyline": [node_pos[row.from_node], node_pos[row.to_node]],
            "time_attributes": time_attrs,
        })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({"coordinate_system": "pre_meta_k_port_validation", "links": links}, f, ensure_ascii=False, indent=2)
