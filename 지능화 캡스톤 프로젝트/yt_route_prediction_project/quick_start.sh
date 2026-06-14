#!/usr/bin/env bash
set -e
python run_pipeline.py --output_dir outputs --duration_minutes 120 --tasks_per_hour 45 --train_stgnn --stgnn_epochs 2 --max_links_for_stgnn 30
