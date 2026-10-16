"""Feature and target construction for historical line-shift capacity training."""
import numpy as np
import pandas as pd

from model_features import STYLE_ENCODING

POSTER_OPERATORS_PER_LINE = 17
REFERENCE_SHIFT_HOURS = 8.0
CAPACITY_MODEL_FEATURES = [
    "order_quantity", "handover_days", "required_smv_total", "style_enc",
    "assigned_lines", "shifts_per_day", "shift_hours",
]


def make_capacity_features(quantity, handover_days, smv, order_value,
                           style, priority, lines, shifts, shift_hours):
    quantity = float(quantity)
    return {
        "order_quantity": quantity,
        "handover_days": float(handover_days),
        "required_smv_total": float(smv),
        "style_enc": float(STYLE_ENCODING[style]),
        "assigned_lines": float(lines),
        "shifts_per_day": float(shifts),
        "shift_hours": float(shift_hours),
    }


def build_capacity_training_data(orders):
    """Build an order-level target from recorded output and completion dates."""
    data = orders.copy()
    start = pd.to_datetime(data["order_date"], errors="coerce")
    finish = pd.to_datetime(data["actual_completion_date"], errors="coerce")
    elapsed = (finish - start).dt.days.clip(lower=1)
    numeric_columns = [
        "order_quantity", "handover_days", "required_smv_total",
        "actual_completed_qty", "n_available_workers", "n_shifts", "shift_hours",
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    # Keep the encoded feature on the training frame before dropna checks it.
    data["style_enc"] = data["style_id"].map(STYLE_ENCODING)
    data["elapsed_days"] = elapsed
    data["assigned_lines"] = data["n_available_workers"] / POSTER_OPERATORS_PER_LINE
    data["shifts_per_day"] = data["n_shifts"]
    data["target_standard_minutes_per_line_shift"] = (
        data["actual_completed_qty"] * data["required_smv_total"] /
        data["elapsed_days"] / data["assigned_lines"] /
        data["shifts_per_day"] *
        (REFERENCE_SHIFT_HOURS / data["shift_hours"])
    )
    valid = (
        data["actual_completed_qty"].gt(0) &
        data["required_smv_total"].gt(0) &
        data["assigned_lines"].gt(0) &
        data["shifts_per_day"].gt(0) &
        data["shift_hours"].gt(0) &
        data["target_standard_minutes_per_line_shift"].gt(0) &
        data["style_enc"].notna()
    )
    data = data.loc[valid].replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=CAPACITY_MODEL_FEATURES + [
        "target_standard_minutes_per_line_shift", "order_id",
    ])
    X = pd.DataFrame(
        [make_capacity_features(
            row.order_quantity, row.handover_days, row.required_smv_total,
            row.order_value, row.style_id, row.priority, row.assigned_lines,
            row.shifts_per_day, row.shift_hours,
        ) for row in data.itertuples()],
        columns=CAPACITY_MODEL_FEATURES,
    )
    y = data["target_standard_minutes_per_line_shift"].to_numpy(dtype=float)
    return X, y, data["order_id"].to_numpy(), data["style_id"].to_numpy()
