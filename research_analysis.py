"""N1 order feasibility and N2 intervention research for live order inputs."""
import math

import numpy as np
import pandas as pd

from model_features import build_order_features, make_intervention_features
from line_model_features import LINE_MODEL_FEATURES, make_line_features
from factory_config import FACTORY_SHIFTS_PER_DAY

SCENARIO_LABELS = {
    "Current": "Current resources",
    "Extra_Workers": "Add workers",
    "Hybrid": "Hybrid support",
    "Machine_Adjust": "Machine adjustment",
    "Outsourcing": "Outsource a portion",
    "Overtime": "Overtime",
}


def _positive_probability(model, features):
    probabilities = model.predict_proba(features)[0]
    classes = list(getattr(model, "classes_", [0, 1]))
    positive_index = classes.index(1) if 1 in classes else len(probabilities) - 1
    return float(probabilities[positive_index])


def predict_order_feasibility(bundle, metrics, quantity, handover_days, smv,
                              order_value, style, priority):
    """N1 probability for the form inputs and training-time factory profile."""
    if not bundle:
        return {"available": False, "message": "Delivery estimate is unavailable because the configured prediction model could not be loaded."}
    profile = bundle["profile"]
    row = build_order_features(
        quantity, handover_days, smv, order_value, style, priority, profile,
    )
    X = pd.DataFrame([row], columns=bundle["features"])
    if bundle.get("type") == "keras":
        scaled = bundle["scaler"].transform(X)
        probability = float(bundle["model"].predict(scaled, verbose=0).reshape(-1)[0])
    else:
        probability = _positive_probability(bundle["model"], X)
    model_metrics = (metrics or {}).get(bundle.get("name"), {})
    return {
        "available": True,
        "probability": probability,
        "percent": probability * 100,
        "decision": "Higher predicted on-time likelihood" if probability >= 0.5 else "Review delivery risk",
        "model_name": bundle.get("name", "N1 classifier"),
        "validation_auc": model_metrics.get("roc_auc"),
        "validation_brier": model_metrics.get("brier_score"),
        "training_signature": bundle.get("training_signature"),
        "scope_note": (
            "On-time likelihood is a model prediction based on this order's inputs and the factory profile used during model development. "
            "Check it alongside the selected line and shift capacity plan."
        ),
    }


def predict_line_feasibility(bundle, metrics, quantity, handover_days, smv,
                             order_value, style, priority, lines, shifts,
                             shift_hours):
    """Predict on-time likelihood using the lines and shifts in the plan."""
    if not bundle:
        return {
            "available": False,
            "message": "Capacity-aware delivery likelihood is unavailable because its model could not be loaded.",
        }
    row = make_line_features(
        quantity, handover_days, smv, order_value, style, priority,
        lines, shifts, shift_hours,
    )
    training_ranges = bundle.get("training_ranges", {})
    out_of_range = [
        f"{feature} ({training_ranges[feature][0]:g}–{training_ranges[feature][1]:g})"
        for feature, value in row.items()
        if feature in training_ranges and not training_ranges[feature][0] <= value <= training_ranges[feature][1]
    ]
    X = pd.DataFrame([row], columns=bundle["features"] or LINE_MODEL_FEATURES)
    probabilities = bundle["model"].predict_proba(X)[0]
    classes = list(getattr(bundle["model"], "classes_", [0, 1]))
    positive_index = classes.index(1) if 1 in classes else len(probabilities) - 1
    probability = float(probabilities[positive_index])
    model_metrics = metrics or {}
    return {
        "available": True,
        "probability": probability,
        "percent": probability * 100,
        "decision": "Higher predicted on-time likelihood" if probability >= 0.5 else "Review delivery risk",
        "model_name": bundle.get("name", "Line-aware on-time model"),
        "validation_auc": model_metrics.get("roc_auc"),
        "validation_brier": model_metrics.get("brier_score"),
        "scope_note": (
            f"Estimate uses the entered {lines} line(s), the factory's {shifts} shifts per day, and historical order outcomes. "
            "It is a learned likelihood, not a guarantee. Compare it with the measured capacity plan below."
        ),
        "extrapolation_note": (
            "This likelihood is an extrapolation because the entered capacity is outside the model's training data range for "
            + ", ".join(out_of_range)
            + ". Treat it as indicative and rely on the production capacity plan for staffing decisions."
            if out_of_range else None
        ),
    }


def _scenario_profiles(interventions, orders):
    """Derive typical intervention sizes from historical scenario records."""
    required = {
        "scenario_type", "added_workers", "overtime_hours", "added_machines",
        "outsourced_qty", "order_id",
    }
    if not required.issubset(interventions.columns):
        return {}
    order_sizes = orders[["order_id", "order_quantity"]].copy()
    if "n_shifts" in orders.columns:
        order_shifts = pd.to_numeric(orders["n_shifts"], errors="coerce")
        order_sizes = order_sizes.loc[order_shifts.le(FACTORY_SHIFTS_PER_DAY)]
    joined = interventions.merge(order_sizes, on="order_id", how="inner")
    for col in ["added_workers", "overtime_hours", "added_machines", "outsourced_qty", "order_quantity"]:
        joined[col] = pd.to_numeric(joined[col], errors="coerce")
    joined["outsourced_fraction"] = (
        joined["outsourced_qty"] / joined["order_quantity"].replace(0, np.nan)
    )
    columns = ["added_workers", "overtime_hours", "added_machines", "outsourced_fraction"]
    profiles = {}
    for scenario_type, rows in joined.groupby("scenario_type"):
        profile = rows[columns].median(numeric_only=True).fillna(0).to_dict()
        profiles[str(scenario_type)] = profile
    return profiles


def analyze_intervention_options(n2_bundle, cost_model, n2_metrics, cost_metrics,
                                 interventions, orders, quantity, handover_days,
                                 smv, order_value, style, priority):
    """Compare historical N2 intervention patterns using the fitted models.

    Scenario sizes come from per-scenario medians in recorded intervention rows.
    The N2 score means similarity to historically selected scenarios; it is not
    an estimate that the intervention will improve on-time delivery.
    """
    if not n2_bundle or cost_model is None:
        return {"available": False, "message": "Intervention estimates are unavailable because the configured models could not be loaded."}

    profiles = _scenario_profiles(interventions, orders)
    scenario_types = [
        kind for kind in n2_bundle.get("scenario_encoding", {}) if kind in profiles
    ]
    if not scenario_types:
        return {"available": False, "message": "No intervention profiles are available for the model's scenario types."}

    profile = n2_bundle["profile"]
    base = build_order_features(
        quantity, handover_days, smv, order_value, style, priority, profile,
    )
    rows = []
    feature_rows = []
    for scenario_type in scenario_types:
        pattern = profiles[scenario_type]
        scenario = {
            "scenario_type": scenario_type,
            "added_workers": float(pattern["added_workers"]),
            "overtime_hours": float(pattern["overtime_hours"]),
            "added_machines": float(pattern["added_machines"]),
            "outsourced_qty": float(min(
                quantity, math.ceil(pattern["outsourced_fraction"] * quantity),
            )),
        }
        features = make_intervention_features(base, scenario, profile)
        feature_rows.append(features)
        rows.append({
            "scenario_type": scenario_type,
            "label": SCENARIO_LABELS.get(scenario_type, scenario_type.replace("_", " ")),
            **scenario,
        })

    X = pd.DataFrame(feature_rows, columns=n2_bundle["features"])
    costs = np.maximum(0, np.asarray(cost_model.predict(X), dtype=float))
    selection = n2_bundle["model"].predict_proba(X)
    classes = list(getattr(n2_bundle["model"], "classes_", [0, 1]))
    positive_index = classes.index(1) if 1 in classes else selection.shape[1] - 1
    model_metrics = (n2_metrics or {}).get(n2_bundle.get("name"), {})
    result = []
    for row, cost, probability in zip(rows, costs, selection[:, positive_index]):
        # Current resources carry no incremental intervention cost by definition.
        estimated_cost = 0.0 if row["scenario_type"] == "Current" else float(cost)
        result.append({
            **row,
            "estimated_cost": estimated_cost,
            "cost_per_piece": estimated_cost / max(quantity, 1),
            "cost_share_order_value": estimated_cost / max(order_value, 1) * 100,
            "selection_score": float(probability),
        })
    result.sort(key=lambda row: (-row["selection_score"], row["estimated_cost"]))
    lowest_cost = min(result, key=lambda row: row["estimated_cost"])
    highest_selection = result[0]
    return {
        "available": True,
        "options": result,
        "lowest_cost": lowest_cost,
        "top_selection": highest_selection,
        "model_name": n2_bundle.get("name", "N2 classifier"),
        "cost_model_name": type(cost_model).__name__,
        "validation_auc": model_metrics.get("roc_auc"),
        "validation_brier": model_metrics.get("brier_score"),
        "cost_mae": (cost_metrics or {}).get("mae"),
        "scope_note": (
            "The model selection score indicates how closely an option matches intervention patterns learned during model development; "
            "it is not a probability of delivery improvement. Costs are estimates and should be checked against current rates."
        ),
    }
