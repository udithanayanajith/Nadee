"""Features shared by line-planning model training and live predictions."""
import numpy as np
import math

from model_features import STYLE_ENCODING, STYLE_CAPACITY_FACTOR, PRIORITY_ENCODING, PRIORITY_CAPACITY_FACTOR

LINE_GRADE_MIX = {"Jumper": 5, "A+": 4, "A": 4, "B": 3, "C": 1}
LINE_OPERATORS = sum(LINE_GRADE_MIX.values())
EFFICIENCY_TARGET = 0.70
FEASIBILITY_ACCEPT_THRESHOLD = 0.70
FEASIBILITY_MODEL_TARGET = "order_status_acceptance_v2"
SHIFT_HOURS = 8
LINE_MODEL_FEATURES = [
    "order_quantity", "handover_days", "required_smv_total", "order_value",
    "style_enc", "priority_enc", "assigned_lines", "shifts_per_day",
    "line_capacity_coverage", "unit_value",
]


def make_line_features(quantity, days, smv, order_value, style, priority, lines, shifts,
                       shift_hours: float = SHIFT_HOURS):
    workload = max(float(quantity) * float(smv), 1.0)
    daily_line_capacity = (
        max(int(lines), 1) * LINE_OPERATORS * float(shift_hours) * 60
        * EFFICIENCY_TARGET * max(int(shifts), 1)
    )
    coverage = (
        daily_line_capacity * max(int(days), 1) / workload
        * STYLE_CAPACITY_FACTOR[style] * PRIORITY_CAPACITY_FACTOR[priority]
    )
    return {
        "order_quantity": float(quantity),
        "handover_days": float(days),
        "required_smv_total": float(smv),
        "order_value": float(order_value),
        "style_enc": float(STYLE_ENCODING[style]),
        "priority_enc": float(PRIORITY_ENCODING[priority]),
        "assigned_lines": float(lines),
        "shifts_per_day": float(shifts),
        "line_capacity_coverage": float(coverage),
        "unit_value": float(order_value) / max(float(quantity), 1.0),
    }


def historical_line_count(worker_count):
    """Map historical headcount to full 17-operator line equivalents."""
    return max(1, min(30, int(np.floor(float(worker_count) / LINE_OPERATORS))))


def synthetic_acceptance_probability(capacity_coverage):
    """Smooth synthetic acceptance rate used only to make sample labels realistic."""
    logit = math.log(FEASIBILITY_ACCEPT_THRESHOLD / (1 - FEASIBILITY_ACCEPT_THRESHOLD))
    logit += 0.8 * (float(capacity_coverage) - 1.0)
    return 1.0 / (1.0 + math.exp(-logit))


def acceptance_label(capacity_coverage, order_id):
    """Sample a reproducible Accept/Reject decision without changing data RNG."""
    probability = synthetic_acceptance_probability(capacity_coverage)
    draw = ((int(order_id) * 2654435761) % 4294967296) / 4294967296.0
    return "Accept" if draw < probability else "Reject"
