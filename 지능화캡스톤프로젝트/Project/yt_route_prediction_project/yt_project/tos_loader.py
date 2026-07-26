from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import numpy as np

REQUIRED_COLUMNS = ["source_file", "updateTime", "bitt", "berth", "x", "y"]


def _extract_records_from_json(obj: dict, source_file: str) -> list[dict]:
    update_time = obj.get("updateTime") or obj.get("update_time") or obj.get("timestamp") or ""
    data = obj.get("data", obj if isinstance(obj, list) else [])
    if isinstance(data, dict):
        # Some TOS payloads wrap data in another list-like value.
        for key in ("items", "records", "bitts", "berths"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    records = []
    if not isinstance(data, list):
        return records
    for row in data:
        if not isinstance(row, dict):
            continue
        if "x" not in row or "y" not in row:
            continue
        records.append({
            "source_file": source_file,
            "updateTime": row.get("updateTime") or update_time,
            "bitt": row.get("bitt") or row.get("bittNo") or row.get("id") or row.get("name"),
            "berth": row.get("berth") or row.get("berthNo") or row.get("mode") or "UNKNOWN",
            "x": float(row.get("x")),
            "y": float(row.get("y")),
        })
    return records


def load_tos_anchor_files(tos_dir: str | Path | None) -> pd.DataFrame:
    """Load TOS JSON anchor data.

    Expected shape: JSON files containing `data` rows with bitt/berth/x/y.
    If no directory or no valid records are found, a demo anchor set is generated
    so the whole validation pipeline can still run.
    """
    if tos_dir:
        tos_dir = Path(tos_dir)
        files = sorted(list(tos_dir.glob("*.json")))
    else:
        files = []

    rows: list[dict] = []
    for path in files:
        try:
            with path.open("r", encoding="utf-8") as f:
                obj = json.load(f)
            rows.extend(_extract_records_from_json(obj, path.name))
        except Exception as exc:
            print(f"[WARN] JSON read failed: {path} ({exc})")

    if rows:
        df = pd.DataFrame(rows)
        for col in REQUIRED_COLUMNS:
            if col not in df.columns:
                df[col] = None
        return df[REQUIRED_COLUMNS].sort_values(["source_file", "x", "y"]).reset_index(drop=True)

    return generate_demo_tos_anchors()


def generate_demo_tos_anchors(n_points: int = 71) -> pd.DataFrame:
    """Generate sample Bitt/Berth anchors similar to the presentation statistics.

    The user's current PPT says the original TOS-1-7 dataset contains 40 JSON
    snapshots, 71 coordinate points per file, and x range about -27.65~1196.
    This demo does not replace real TOS data. It only allows algorithm validation.
    """
    x_values = np.linspace(-27.65, 1196.0, n_points)
    rows = []
    berth_names = ["1B", "2B", "3B", "4B", "OT"]
    for file_idx in range(40):
        for i, x in enumerate(x_values):
            berth = berth_names[min(len(berth_names)-1, int(i / (n_points / len(berth_names))))]
            rows.append({
                "source_file": f"demo_TOS_1_7_{file_idx:02d}.json",
                "updateTime": f"2026-01-19T{file_idx % 24:02d}:00:00",
                "bitt": f"B{i+1:03d}",
                "berth": berth,
                "x": float(x),
                "y": 2.5 if i % 2 else 0.0,
            })
    return pd.DataFrame(rows)[REQUIRED_COLUMNS]


def summarize_tos(df: pd.DataFrame) -> dict:
    unique_points = df[["bitt", "berth", "x", "y"]].drop_duplicates()
    snapshots = df["source_file"].nunique() if "source_file" in df else 0
    return {
        "json_files_or_snapshots": int(snapshots),
        "unique_anchor_points": int(len(unique_points)),
        "total_records": int(len(df)),
        "x_min": float(df["x"].min()),
        "x_max": float(df["x"].max()),
        "y_values": sorted([float(v) for v in df["y"].dropna().unique().tolist()]),
        "coordinate_changes_by_snapshot": 0,  # Anchor snapshots are treated as static.
    }


def make_anchor_master(df: pd.DataFrame) -> pd.DataFrame:
    master = df[["bitt", "berth", "x", "y"]].drop_duplicates().sort_values("x").reset_index(drop=True)
    master.insert(0, "anchor_id", [f"A{i:03d}" for i in range(len(master))])
    return master
