"""Shared feature engineering used during training and live order prediction."""
from factory_config import FACTORY_SHIFTS_PER_DAY
SCENARIO_ENCODING = {
    name: index
    for index, name in enumerate(
        ["Current", "Extra_Workers", "Hybrid", "Machine_Adjust", "Outsourcing", "Overtime"]
    )
}
STYLE_ENCODING = {"Hoodie": 0, "Jacket": 1, "Legging": 2, "Polo": 3, "T-Shirt": 4}
PRIORITY_ENCODING = {"Low": 0, "Normal": 1, "High": 2}
STYLE_CAPACITY_FACTOR = {"Polo": 1.00, "T-Shirt": 1.02, "Hoodie": 0.94,
                         "Legging": 0.98, "Jacket": 0.92}
PRIORITY_CAPACITY_FACTOR = {"Low": 1.04, "Normal": 1.00, "High": 0.90}


def build_factory_profile(orders, allocations):
    """Estimate typical order and factory values from the available records."""
    per_order = allocations.groupby("order_id").agg(
        n_conflicts=("conflict_flag", "sum"),
        avg_worker_eff=("actual_efficiency", "mean"),
    )
    return {
        "smv": float(orders["required_smv_total"].median()),
        "unit_value": float((orders["order_value"] / orders["order_quantity"]).median()),
        "typical_workers": float(orders["n_available_workers"].median()),
        "typical_shifts": float(FACTORY_SHIFTS_PER_DAY),
        "shift_hours": float(orders["shift_hours"].median()),
        "capacity_efficiency": float(orders["avg_worker_efficiency"].median()),
        "n_conflicts": float(per_order["n_conflicts"].median()),
        "avg_worker_eff": float(per_order["avg_worker_eff"].median()),
    }


def build_order_features(quantity, handover_days, smv, order_value, style, priority, profile):
    """Build matching features from quantity, days, SMV, value, style and priority."""
    quantity = float(quantity)
    handover_days = max(float(handover_days), 1.0)
    smv = float(smv)
    order_value = float(order_value)
    workload = quantity * smv
    daily_capacity = (
        profile["typical_workers"]
        * profile["shift_hours"]
        * 60
        * profile["capacity_efficiency"] / 100
        * profile["typical_shifts"]
    )
    return {
        "order_quantity": quantity,
        "handover_days": handover_days,
        "required_smv_total": smv,
        "order_value": order_value,
        "style_enc": STYLE_ENCODING[style],
        "priority_enc": PRIORITY_ENCODING[priority],
        "unit_value": order_value / quantity,
        "estimated_workload_min": workload,
        "estimated_workload_per_day": workload / handover_days,
        "capacity_ratio": (
            daily_capacity * handover_days / max(workload, 1)
            * STYLE_CAPACITY_FACTOR[style]
            * PRIORITY_CAPACITY_FACTOR[priority]
        ),
    }


def make_intervention_features(order_features, scenario, profile):
    """Append the factory-level and scenario features used by the N2 model."""
    row = {
        "scenario_enc": SCENARIO_ENCODING[scenario["scenario_type"]],
        "added_workers": float(scenario["added_workers"]),
        "overtime_hours": float(scenario["overtime_hours"]),
        "added_machines": float(scenario["added_machines"]),
        "outsourced_qty": float(scenario["outsourced_qty"]),
        "order_quantity": order_features["order_quantity"],
        "handover_days": order_features["handover_days"],
        "order_value": order_features["order_value"],
        "required_workload_min": order_features["estimated_workload_min"],
        "style_enc": order_features["style_enc"],
        "priority_enc": order_features["priority_enc"],
        "capacity_ratio": order_features["capacity_ratio"],
        "n_conflicts": profile["n_conflicts"],
        "avg_worker_eff": profile["avg_worker_eff"],
    }
    return row
