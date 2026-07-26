from __future__ import annotations
from dataclasses import dataclass, asdict
import random
import pandas as pd


@dataclass
class SOPTemplate:
    sop_type: str
    origin_type: str
    dest_type: str
    priority: int
    base_service_s: float
    description: str


def default_sop_templates() -> list[SOPTemplate]:
    return [
        SOPTemplate("DISCHARGE_QC_TO_YARD", "QC", "YARD_BLOCK", 3, 120, "양하: QC 인수 후 Yard 장치"),
        SOPTemplate("LOADING_YARD_TO_QC", "YARD_BLOCK", "QC", 3, 135, "선적: Yard 반출 후 QC 인도"),
        SOPTemplate("GATE_TO_YARD", "GATE", "YARD_BLOCK", 2, 180, "반입: Gate에서 Yard로 이동"),
        SOPTemplate("YARD_TO_GATE", "YARD_BLOCK", "GATE", 2, 160, "반출: Yard에서 Gate로 이동"),
        SOPTemplate("YARD_REHANDLE", "YARD_BLOCK", "YARD_BLOCK", 1, 210, "야드 내 재배치/대기/우회"),
    ]


def templates_to_frame(templates: list[SOPTemplate]) -> pd.DataFrame:
    return pd.DataFrame([asdict(t) for t in templates])


def sample_node_by_type(node_master: pd.DataFrame, node_type: str, rng: random.Random) -> str:
    candidates = node_master[node_master["node_type"] == node_type]["node_id"].tolist()
    if not candidates:
        raise ValueError(f"No node candidates for type={node_type}")
    return rng.choice(candidates)


def demand_multiplier(minute_of_day: float) -> float:
    """Simple shift/demand pattern for pre-Meta K-Port validation."""
    hour = (minute_of_day / 60.0) % 24
    if 7 <= hour < 10:
        return 1.35
    if 13 <= hour < 16:
        return 1.20
    if 19 <= hour < 22:
        return 1.10
    if 0 <= hour < 5:
        return 0.65
    return 1.0


def generate_task_plan(node_master: pd.DataFrame, duration_minutes: int, tasks_per_hour: int, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    templates = default_sop_templates()
    rows = []
    task_id = 0
    # Poisson-like generation without scipy.
    for minute in range(duration_minutes):
        expected = tasks_per_hour / 60.0 * demand_multiplier(minute)
        n = int(expected)
        if rng.random() < expected - n:
            n += 1
        # Occasional burst at QC/Yard shift changes.
        if minute % 90 in (0, 1, 2) and rng.random() < 0.35:
            n += rng.randint(1, 3)
        for _ in range(n):
            t = rng.choices(templates, weights=[0.28, 0.25, 0.18, 0.17, 0.12], k=1)[0]
            origin = sample_node_by_type(node_master, t.origin_type, rng)
            dest = sample_node_by_type(node_master, t.dest_type, rng)
            tries = 0
            while dest == origin and tries < 10:
                dest = sample_node_by_type(node_master, t.dest_type, rng)
                tries += 1
            start_s = minute * 60 + rng.randint(0, 59)
            rows.append({
                "task_id": f"T{task_id:06d}",
                "sop_type": t.sop_type,
                "origin_node": origin,
                "dest_node": dest,
                "planned_start_s": start_s,
                "priority": t.priority,
                "base_service_s": t.base_service_s,
                "demand_multiplier": demand_multiplier(minute),
            })
            task_id += 1
    return pd.DataFrame(rows)
