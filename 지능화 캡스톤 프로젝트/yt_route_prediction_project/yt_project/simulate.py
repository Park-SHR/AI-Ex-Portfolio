from __future__ import annotations
import random
import math
import pandas as pd
import numpy as np
from .graph_builder import k_shortest_paths, path_to_link_ids


def _path_stats(G, path: list[str]) -> dict:
    length = 0.0
    base_time = 0.0
    capacity_min = 999
    turns = 0
    last_type = None
    for u, v in zip(path[:-1], path[1:]):
        edge = G[u][v]
        length += edge.get("length_m", 0)
        base_time += edge.get("weight", 0)
        capacity_min = min(capacity_min, edge.get("capacity", 1))
        road_type = edge.get("road_type", "")
        if last_type is not None and road_type != last_type:
            turns += 1
        last_type = road_type
    return {"path_length_m": length, "base_time_s": base_time, "turns": turns, "min_capacity": capacity_min}


def _choose_route(paths: list[list[str]], task: pd.Series, rng: random.Random, delay_injected: bool) -> tuple[int, dict]:
    scored = []
    for rank, path in enumerate(paths, start=1):
        # The first path is shortest. Other paths can be selected when stochastic delays occur.
        base = len(path) * 0.03 + rank * 0.25
        if delay_injected:
            hour = (task["planned_start_s"] / 3600) % 24
            peak = 1.0 if 7 <= hour < 10 or 13 <= hour < 16 else 0.0
            priority_relief = max(0, 4 - int(task["priority"])) * 0.08
            # Nonlinear operational noise: simulates QC queue, yard interference, temporary blockages.
            stochastic_delay = rng.gammavariate(1.8 + peak, 0.45)
            route_randomness = rng.random() * (1.4 + 0.5 * peak)
            score = base + stochastic_delay + route_randomness - priority_relief
        else:
            score = base
        scored.append((score, rank, path))
    scored.sort(key=lambda x: x[0])
    chosen_score, chosen_rank, chosen_path = scored[0]
    return chosen_rank, {"chosen_score": chosen_score, "chosen_path": chosen_path}


def simulate_traces(G, task_plan: pd.DataFrame, k_paths: int = 5, seed: int = 42, delay_injected: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate vehicle traces from SOP task plan.

    Returns
    -------
    task_routes: one row per task with selected route metadata.
    trace: one row per traversed link interval.
    """
    rng = random.Random(seed)
    task_rows = []
    trace_rows = []
    vehicle_pool = [f"YT{i:03d}" for i in range(1, 60)]

    for _, task in task_plan.iterrows():
        paths = k_shortest_paths(G, task["origin_node"], task["dest_node"], k=k_paths)
        if not paths:
            continue
        route_rank, chosen = _choose_route(paths, task, rng, delay_injected=delay_injected)
        path = chosen["chosen_path"]
        stats = _path_stats(G, path)
        link_ids = path_to_link_ids(G, path)
        vehicle_id = rng.choice(vehicle_pool)
        start_s = float(task["planned_start_s"] + rng.uniform(0, 120))
        current_s = start_s
        total_wait = 0.0
        for seq, (u, v, link_id) in enumerate(zip(path[:-1], path[1:], link_ids)):
            edge = G[u][v]
            base_time = float(edge["weight"])
            road_type = edge.get("road_type", "ROAD")
            speed_factor = max(0.35, rng.normalvariate(1.0, 0.12)) if delay_injected else 1.0
            # Delay model: operational priority, bottleneck by road type, and peak demand.
            hour = (current_s / 3600) % 24
            peak_factor = 1.25 if 7 <= hour < 10 or 13 <= hour < 16 else 1.0
            bottleneck = {"QC_ACCESS": 1.4, "YARD_ACCESS": 1.3, "GATE_ACCESS": 1.45, "MAIN_ROAD": 1.0, "YARD_LANE": 1.2}.get(road_type, 1.0)
            wait = 0.0
            if delay_injected:
                wait = rng.expovariate(1 / (6.0 * bottleneck * peak_factor))
                if road_type in ("QC_ACCESS", "GATE_ACCESS") and rng.random() < 0.25 * peak_factor:
                    wait += rng.uniform(20, 85)
                if road_type == "YARD_ACCESS" and rng.random() < 0.18 * peak_factor:
                    wait += rng.uniform(15, 70)
            travel_time = base_time / speed_factor + wait
            end_s = current_s + travel_time
            speed = float(edge.get("length_m", 1.0)) / max(travel_time - wait * 0.4, 1.0)
            trace_rows.append({
                "task_id": task["task_id"],
                "vehicle_id": vehicle_id,
                "sop_type": task["sop_type"],
                "priority": task["priority"],
                "origin_node": task["origin_node"],
                "dest_node": task["dest_node"],
                "route_rank": route_rank,
                "seq": seq,
                "from_node": u,
                "to_node": v,
                "link_id": link_id,
                "start_s": current_s,
                "end_s": end_s,
                "travel_time_s": travel_time,
                "wait_s": wait,
                "speed_mps": speed,
                "road_type": road_type,
            })
            current_s = end_s
            total_wait += wait
        task_rows.append({
            **task.to_dict(),
            "route_rank": route_rank,
            "path_nodes": "|".join(path),
            "path_links": "|".join(link_ids),
            "path_length_m": stats["path_length_m"],
            "base_time_s": stats["base_time_s"],
            "sim_eta_s": current_s - start_s,
            "total_wait_s": total_wait,
            "delay_injected": delay_injected,
            "selected_not_shortest": int(route_rank != 1),
        })
    return pd.DataFrame(task_rows), pd.DataFrame(trace_rows)


def aggregate_link_time(trace: pd.DataFrame, link_master: pd.DataFrame, time_bin_seconds: int = 60, duration_minutes: int = 360) -> pd.DataFrame:
    link_ids = link_master["link_id"].tolist()
    n_bins = int(math.ceil(duration_minutes * 60 / time_bin_seconds))
    frames = []
    if trace.empty:
        raise ValueError("Trace is empty. Cannot aggregate link-time samples.")
    t = trace.copy()
    t["bin"] = (t["start_s"] // time_bin_seconds).astype(int)
    grouped = t.groupby(["bin", "link_id"])
    agg = grouped.agg(
        yt_count=("vehicle_id", "nunique"),
        pass_count=("task_id", "count"),
        avg_speed_mps=("speed_mps", "mean"),
        wait_sum_s=("wait_s", "sum"),
        avg_travel_time_s=("travel_time_s", "mean"),
    ).reset_index()
    static = link_master[["link_id", "length_m", "speed_limit_mps", "capacity", "base_time_s"]]
    full_index = pd.MultiIndex.from_product([range(n_bins), link_ids], names=["bin", "link_id"]).to_frame(index=False)
    out = full_index.merge(agg, on=["bin", "link_id"], how="left").fillna({
        "yt_count": 0, "pass_count": 0, "avg_speed_mps": 0, "wait_sum_s": 0, "avg_travel_time_s": 0,
    }).merge(static, on="link_id", how="left")
    out["occupancy"] = out["yt_count"] / out["capacity"].clip(lower=1)
    out["wait_norm"] = np.tanh(out["wait_sum_s"] / 120.0)
    out["speed_drop"] = 1 - (out["avg_speed_mps"] / out["speed_limit_mps"].replace(0, np.nan)).fillna(1).clip(0, 1)
    out["congestion_score"] = (0.50 * out["occupancy"].clip(0, 2) / 2 + 0.30 * out["wait_norm"] + 0.20 * out["speed_drop"]).clip(0, 1)
    return out.sort_values(["bin", "link_id"]).reset_index(drop=True)
