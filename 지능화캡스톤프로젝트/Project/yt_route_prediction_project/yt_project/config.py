from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProjectConfig:
    tos_dir: Path | None = None
    output_dir: Path = Path("outputs")
    seed: int = 42
    duration_minutes: int = 360
    time_bin_seconds: int = 60
    vehicles: int = 35
    tasks_per_hour: int = 85
    k_paths: int = 5
    horizon_minutes: int = 5
    window_size: int = 6
    train_stgnn: bool = True
    stgnn_epochs: int = 12
    stgnn_hidden: int = 32
    max_links_for_stgnn: int = 120
    congestion_threshold: float = 0.30

    def ensure_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "tables").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "plots").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "models").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "reports").mkdir(parents=True, exist_ok=True)
