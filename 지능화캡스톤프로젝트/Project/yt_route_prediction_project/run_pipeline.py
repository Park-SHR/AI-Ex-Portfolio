from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from yt_project.config import ProjectConfig
from yt_project.utils import set_seed, save_json
from yt_project.tos_loader import load_tos_anchor_files, summarize_tos, make_anchor_master
from yt_project.graph_builder import build_port_graph, adjacency_from_links
from yt_project.sop import default_sop_templates, templates_to_frame, generate_task_plan
from yt_project.simulate import simulate_traces, aggregate_link_time
from yt_project.features import make_tabular_dataset, make_tensor_dataset
from yt_project.models import train_tabular_models, train_route_choice_model
from yt_project.stgnn import train_stgnn_model
from yt_project.evaluate import route_choice_metrics
from yt_project.visualize import (
    plot_graph, plot_model_comparison, plot_pred_actual, plot_congestion_heatmap, export_unreal_prediction_json,
)
from yt_project.report import write_markdown_report


def parse_args() -> ProjectConfig:
    p = argparse.ArgumentParser(description="YT Route/Traffic Prediction Validation Project")
    p.add_argument("--tos_dir", type=str, default=None, help="Directory containing TOS JSON files. If omitted, demo anchors are generated.")
    p.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--duration_minutes", type=int, default=360)
    p.add_argument("--time_bin_seconds", type=int, default=60)
    p.add_argument("--tasks_per_hour", type=int, default=85)
    p.add_argument("--k_paths", type=int, default=5)
    p.add_argument("--horizon_minutes", type=int, default=5)
    p.add_argument("--window_size", type=int, default=6)
    p.add_argument("--train_stgnn", action="store_true", help="Train Simple ST-GNN model")
    p.add_argument("--stgnn_epochs", type=int, default=12)
    p.add_argument("--max_links_for_stgnn", type=int, default=120)
    args = p.parse_args()
    return ProjectConfig(
        tos_dir=Path(args.tos_dir) if args.tos_dir else None,
        output_dir=Path(args.output_dir), seed=args.seed,
        duration_minutes=args.duration_minutes, time_bin_seconds=args.time_bin_seconds,
        tasks_per_hour=args.tasks_per_hour, k_paths=args.k_paths,
        horizon_minutes=args.horizon_minutes, window_size=args.window_size,
        train_stgnn=args.train_stgnn, stgnn_epochs=args.stgnn_epochs,
        max_links_for_stgnn=args.max_links_for_stgnn,
    )


def main() -> None:
    cfg = parse_args()
    cfg.ensure_dirs()
    set_seed(cfg.seed)
    print("[1/8] Load TOS anchors")
    tos_df = load_tos_anchor_files(cfg.tos_dir)
    tos_summary = summarize_tos(tos_df)
    anchor_master = make_anchor_master(tos_df)
    tos_df.to_csv(cfg.output_dir / "tables" / "raw_tos_records.csv", index=False, encoding="utf-8-sig")
    anchor_master.to_csv(cfg.output_dir / "tables" / "anchor_master.csv", index=False, encoding="utf-8-sig")
    save_json(tos_summary, cfg.output_dir / "tables" / "tos_summary.json")

    print("[2/8] Build port graph")
    pg = build_port_graph(anchor_master)
    pg.node_master.to_csv(cfg.output_dir / "tables" / "node_master.csv", index=False, encoding="utf-8-sig")
    pg.link_master.to_csv(cfg.output_dir / "tables" / "link_master.csv", index=False, encoding="utf-8-sig")
    A_df = adjacency_from_links(pg.link_master)
    A_df.to_csv(cfg.output_dir / "tables" / "link_adjacency.csv", encoding="utf-8-sig")
    plot_graph(pg.node_master, pg.link_master, cfg.output_dir / "plots" / "01_port_graph.png")

    print("[3/8] Generate SOP task plan and simulate traces")
    templates = default_sop_templates()
    templates_to_frame(templates).to_csv(cfg.output_dir / "tables" / "sop_template.csv", index=False, encoding="utf-8-sig")
    task_plan = generate_task_plan(pg.node_master, cfg.duration_minutes, cfg.tasks_per_hour, seed=cfg.seed)
    task_plan.to_csv(cfg.output_dir / "tables" / "task_plan.csv", index=False, encoding="utf-8-sig")
    task_routes, trace = simulate_traces(pg.graph, task_plan, k_paths=cfg.k_paths, seed=cfg.seed, delay_injected=True)
    task_routes.to_csv(cfg.output_dir / "tables" / "task_routes.csv", index=False, encoding="utf-8-sig")
    trace.to_csv(cfg.output_dir / "tables" / "synthetic_trace.csv", index=False, encoding="utf-8-sig")

    print("[4/8] Aggregate link-time samples")
    link_time = aggregate_link_time(trace, pg.link_master, time_bin_seconds=cfg.time_bin_seconds, duration_minutes=cfg.duration_minutes)
    link_time.to_csv(cfg.output_dir / "tables" / "link_time_samples.csv", index=False, encoding="utf-8-sig")
    plot_congestion_heatmap(link_time, cfg.output_dir / "plots" / "02_congestion_heatmap.png")

    print("[5/8] Train baseline and tabular ML models")
    horizon_bins = max(1, int((cfg.horizon_minutes * 60) / cfg.time_bin_seconds))
    tab_ds = make_tabular_dataset(link_time, horizon_bins=horizon_bins, congestion_threshold=cfg.congestion_threshold)
    model_metrics, trained = train_tabular_models(tab_ds, cfg.output_dir / "models")
    route_metrics = route_choice_metrics(task_routes)
    route_metric_df = pd.DataFrame([route_metrics])
    route_metric_df.to_csv(cfg.output_dir / "tables" / "route_baseline_metrics.csv", index=False, encoding="utf-8-sig")
    route_choice_df, route_model = train_route_choice_model(task_routes, cfg.output_dir / "models")
    route_choice_df.to_csv(cfg.output_dir / "tables" / "route_choice_model_metrics.csv", index=False, encoding="utf-8-sig")

    # Plot one ML model prediction sample if available.
    if trained:
        name, model = next(iter(trained.items()))
        pred = model.predict(tab_ds["X_test"])
        plot_pred_actual(tab_ds["y_test"], pred, cfg.output_dir / "plots" / "03_pred_vs_actual_tabular.png", f"{name}: Prediction vs Actual")

    all_metrics = model_metrics.copy()

    print("[6/8] Train ST-GNN model" if cfg.train_stgnn else "[6/8] Skip ST-GNN model")
    if cfg.train_stgnn:
        # Use busiest or first links to keep CPU training short.
        link_load = link_time.groupby("link_id")["pass_count"].sum().sort_values(ascending=False)
        selected_links = link_load.head(min(cfg.max_links_for_stgnn, len(link_load))).index.tolist()
        tensor_ds = make_tensor_dataset(link_time[link_time["link_id"].isin(selected_links)], selected_links, horizon_bins=horizon_bins, window_size=cfg.window_size)
        st_metrics, st_pred = train_stgnn_model(
            tensor_ds, A_df, cfg.output_dir / "models", epochs=cfg.stgnn_epochs,
            hidden=cfg.stgnn_hidden, threshold=cfg.congestion_threshold,
        )
        all_metrics = pd.concat([all_metrics, st_metrics], ignore_index=True)
        # Save ST-GNN prediction arrays instead of plotting in the same process;
        # this keeps CPU-only environments stable.
        np.save(cfg.output_dir / "tables" / "stgnn_y_test.npy", tensor_ds["y_test"])
        np.save(cfg.output_dir / "tables" / "stgnn_y_pred.npy", st_pred)

    all_metrics.to_csv(cfg.output_dir / "tables" / "model_metrics.csv", index=False, encoding="utf-8-sig")
    plot_model_comparison(all_metrics, cfg.output_dir / "plots" / "05_model_comparison.png")

    print("[7/8] Export Unreal/META K-Port sample JSON")
    export_unreal_prediction_json(pg.link_master, pg.node_master, link_time, cfg.output_dir / "unreal_prediction_sample.json")

    print("[8/8] Write report")
    report_path = write_markdown_report(cfg.output_dir, tos_summary, route_metrics, all_metrics, route_choice_df)

    print("\nDONE")
    print(f"Output dir: {cfg.output_dir.resolve()}")
    print(f"Report: {report_path.resolve()}")
    print("Key table: outputs/tables/model_metrics.csv")
    # Some torch/BLAS builds keep background worker threads alive in notebooks or
    # restricted runtimes. Flush and force clean process termination for CLI runs.
    import sys, os
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
