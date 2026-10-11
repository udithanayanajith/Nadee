"""Create a clearly labelled, synthetic 1,000-row demo dataset from the sample schema.

These records are simulated for pipeline development. They are not factory history.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "Component4_ML_Training_Dataset_Synthetic_Sample.csv"
DEFAULT_OUTPUT = HERE / "Component4_ML_Training_Dataset_Synthetic_1000.csv"
SOURCE_MARKER = "SYNTHETIC_GENERATED_NOT_REAL_HISTORY"


def generate(rows: int = 1000, seed: int = 20261011) -> pd.DataFrame:
    if rows < 500:
        raise ValueError("Generate at least 500 rows for this expanded demo.")
    rng = np.random.default_rng(seed)
    base = pd.read_csv(SOURCE)
    base["decision_snapshot_date"] = pd.to_datetime(base["decision_snapshot_date"])
    picked = base.iloc[rng.integers(0, len(base), size=rows)].reset_index(drop=True).copy()

    count_columns = [
        "operation_count", "available_JUMPER", "available_A_plus", "available_A", "available_B", "available_C",
        "assigned_elsewhere_JUMPER", "assigned_elsewhere_A_plus", "assigned_elsewhere_A",
        "assigned_elsewhere_B", "assigned_elsewhere_C", "available_machine_count", "required_machine_count",
        "shifts_per_day",
    ]
    ratio_columns = ["attendance_rate", "recent_line_efficiency_0_1"]
    continuous_columns = [
        "order_quantity", "sewing_smv_min_per_piece", "days_to_handover", "hours_per_shift",
        "planned_downtime_hours", "prior_day_output_qty",
    ]
    for column in count_columns:
        spread = max(float(base[column].std(ddof=0)) * 0.12, 0.7)
        picked[column] = np.rint(picked[column].to_numpy(float) + rng.normal(0, spread, rows)).clip(0).astype(int)
    for column in ratio_columns:
        picked[column] = np.clip(picked[column].to_numpy(float) + rng.normal(0, 0.025, rows), 0.35, 0.995)
    for column in continuous_columns:
        spread = max(float(base[column].std(ddof=0)) * 0.08, 0.15)
        picked[column] = picked[column].to_numpy(float) + rng.normal(0, spread, rows)
    picked["order_quantity"] = np.rint(picked["order_quantity"].clip(1000, 30000) / 100) * 100
    picked["sewing_smv_min_per_piece"] = picked["sewing_smv_min_per_piece"].clip(3, 25)
    picked["operation_count"] = picked["operation_count"].clip(4, 30)
    picked["days_to_handover"] = np.rint(picked["days_to_handover"].clip(2, 35)).astype(int)
    picked["hours_per_shift"] = picked["hours_per_shift"].clip(6, 10)
    picked["shifts_per_day"] = picked["shifts_per_day"].clip(1, 3)
    picked["planned_downtime_hours"] = picked["planned_downtime_hours"].clip(0, 8)
    picked["required_machine_count"] = picked["required_machine_count"].clip(1, 40)

    grade_weights = {"JUMPER": 1.4, "A_plus": 1.25, "A": 1.15, "B": 1.0, "C": 0.8}
    skilled = np.zeros(rows, dtype=float)
    assigned = np.zeros(rows, dtype=float)
    for grade, weight in grade_weights.items():
        skilled += picked[f"available_{grade}"].to_numpy(float) * weight
        assigned += picked[f"assigned_elsewhere_{grade}"].to_numpy(float) * weight
    effective_workers = np.maximum(2.0, skilled - 0.18 * assigned)
    machine_ratio = np.clip(
        picked["available_machine_count"].to_numpy(float) /
        picked["required_machine_count"].to_numpy(float), 0.45, 1.15,
    )
    hours = picked["hours_per_shift"].to_numpy(float) * picked["shifts_per_day"].to_numpy(float)
    downtime_factor = np.clip(1 - picked["planned_downtime_hours"].to_numpy(float) / np.maximum(hours, 1), 0.5, 1.0)
    daily_capacity_minutes = (
        effective_workers * hours * 60
        * picked["attendance_rate"].to_numpy(float)
        * picked["recent_line_efficiency_0_1"].to_numpy(float)
        * machine_ratio * downtime_factor
    )
    required_minutes = picked["order_quantity"].to_numpy(float) * picked["sewing_smv_min_per_piece"].to_numpy(float)
    expected_days = required_minutes / np.maximum(daily_capacity_minutes, 1)
    completion_days = np.maximum(1, np.rint(expected_days + rng.normal(0, 1.25, rows))).astype(int)
    deadline = picked["days_to_handover"].to_numpy(int)
    on_time = (completion_days <= deadline).astype(int)
    completion_fraction = np.where(
        on_time == 1,
        rng.uniform(0.94, 1.02, rows),
        rng.uniform(0.76, 0.98, rows),
    )
    completed_quantity = np.minimum(
        picked["order_quantity"].to_numpy(float) * 1.02,
        picked["order_quantity"].to_numpy(float) * completion_fraction,
    )

    start = np.datetime64("2022-01-01")
    days = rng.integers(0, 1461, size=rows)
    picked["order_id"] = [f"SYN-HIST-{i:06d}" for i in range(1, rows + 1)]
    picked["decision_snapshot_date"] = pd.to_datetime(start + days.astype("timedelta64[D]")).astype(str)
    picked["target_actual_on_time_0_1"] = on_time
    picked["target_actual_completed_quantity"] = np.rint(completed_quantity).astype(int)
    picked["target_actual_completion_days"] = completion_days
    picked["data_source_type"] = SOURCE_MARKER
    picked = picked.sort_values("decision_snapshot_date").reset_index(drop=True)
    return picked[list(base.columns)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate clearly labelled synthetic apparel order snapshots")
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    dataset = generate(args.rows, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output, index=False)
    print(f"Saved {len(dataset):,} SIMULATED rows to {args.output.resolve()}")
    print("These are not real historical factory records or evidence of model accuracy.")
    print("On-time label counts:", dataset["target_actual_on_time_0_1"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
