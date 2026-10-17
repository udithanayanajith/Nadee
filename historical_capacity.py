"""Order-capacity estimates derived from historical orders with actual output."""
import math

import numpy as np
import pandas as pd
from capacity_model_features import CAPACITY_MODEL_FEATURES, make_capacity_features
from factory_config import FACTORY_SHIFTS_PER_DAY, OPERATORS_PER_LINE_SHIFT

POSTER_OPERATORS_PER_LINE = OPERATORS_PER_LINE_SHIFT
POSTER_GRADE_MIX = {"Jumper": 5, "A+": 4, "A": 4, "B": 3, "C": 1}
REFERENCE_SHIFT_HOURS = 8.0
MIN_STYLE_HISTORY = 30


def _historical_minutes_per_line_shift(orders, max_shifts=FACTORY_SHIFTS_PER_DAY):
    """Normalize realized garment minutes to an 8-hour line-shift day."""
    history = orders.copy()
    start = pd.to_datetime(history["order_date"], errors="coerce")
    finish = pd.to_datetime(history["actual_completion_date"], errors="coerce")
    elapsed_days = (finish - start).dt.days.clip(lower=1)
    numeric = [
        "actual_completed_qty", "required_smv_total", "n_available_workers",
        "n_shifts", "shift_hours",
    ]
    for column in numeric:
        history[column] = pd.to_numeric(history[column], errors="coerce")
    history = history[history["n_shifts"].le(max_shifts)].copy()
    history["elapsed_days"] = elapsed_days
    history["line_shifts_per_day"] = (
        history["n_available_workers"] / POSTER_OPERATORS_PER_LINE
    ) * history["n_shifts"].clip(lower=1)
    history["minutes_per_line_shift"] = (
        history["actual_completed_qty"] * history["required_smv_total"] /
        history["elapsed_days"] / history["line_shifts_per_day"] *
        (REFERENCE_SHIFT_HOURS / history["shift_hours"].clip(lower=1))
    )
    return history.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["minutes_per_line_shift", "style_id"]
    ).loc[lambda rows: rows["minutes_per_line_shift"].gt(0)]


def _baseline_for_style(history, style):
    style_rows = history[history["style_id"].eq(style)]
    if len(style_rows) >= MIN_STYLE_HISTORY:
        return style_rows, style
    return history, "All styles (style history below 30 orders)"


def _rate_range(history_rows, smv, shift_hours):
    values = history_rows["minutes_per_line_shift"].to_numpy(dtype=float)
    lower, upper = np.percentile(values, [5, 95])
    robust = values[(values >= lower) & (values <= upper)]
    minute_range = np.percentile(robust, [10, 50, 90])
    hours_scale = shift_hours / REFERENCE_SHIFT_HOURS
    return [float(value * hours_scale / smv) for value in minute_range]


def _make_groups(line_shifts, available_lines):
    return [
        {
            "line": index % available_lines + 1,
            "shift": f"Shift {index // available_lines + 1}",
            "line_shift_no": index + 1,
        }
        for index in range(line_shifts)
    ]


def build_historical_capacity_plan(orders, quantity, handover_days, smv,
                                   style, available_lines, max_shifts,
                                   shift_hours, order_value=1.0,
                                   priority="Normal", capacity_model=None):
    """Plan aggregate line-shift capacity; never imply a live worker roster."""
    max_shifts = FACTORY_SHIFTS_PER_DAY
    history = _historical_minutes_per_line_shift(orders, FACTORY_SHIFTS_PER_DAY)
    if history.empty:
        return {"error": "Completed orders do not contain enough capacity fields for an estimate."}

    selected_history, benchmark_style = _baseline_for_style(history, style)
    low_rate, median_rate, high_rate = _rate_range(selected_history, smv, shift_hours)
    capacity_source = "Capacity benchmark estimate (production model unavailable)"
    capacity_input_note = "Using the historical benchmark for this style and SMV."
    model_is_current = bool(
        capacity_model
        and capacity_model.get("model") is not None
        and capacity_model.get("features") == CAPACITY_MODEL_FEATURES
        and capacity_model.get("feature_ranges")
        and "p10" in capacity_model.get("residual_quantiles", {})
    )
    if model_is_current:
        features = pd.DataFrame([make_capacity_features(
            quantity, handover_days, smv, order_value, style, priority,
            available_lines, max_shifts, shift_hours,
        )], columns=capacity_model["features"])
        feature_ranges = capacity_model.get("feature_ranges", {})
        out_of_range = [
            feature for feature in features.columns
            if feature in feature_ranges and not (
                feature_ranges[feature][0] <= float(features.iloc[0][feature]) <= feature_ranges[feature][1]
            )
        ]
        if out_of_range:
            capacity_source = "Historical benchmark estimate (outside model training range)"
            capacity_input_note = (
                "The production model was not used because one or more order inputs fall outside its training range. "
                "The displayed output range comes from completed orders for the selected style (or all styles when there are fewer than 30)."
            )
        else:
            predicted_minutes = float(capacity_model["model"].predict(features)[0])
            residuals = capacity_model["residual_quantiles"]
            bounds = capacity_model.get("style_target_bounds", {}).get(style)
            if bounds:
                minimum, maximum = bounds["p05"], bounds["p95"]
            else:
                minimum, maximum = np.percentile(
                    history["minutes_per_line_shift"].to_numpy(dtype=float), [5, 95]
                )
            minutes_range = [
                float(np.clip(predicted_minutes + residuals[key], minimum, maximum))
                for key in ("p10", "p50", "p90")
            ]
            low_rate, median_rate, high_rate = [
                value * (shift_hours / REFERENCE_SHIFT_HOURS) / smv
                for value in minutes_range
            ]
            capacity_source = "Production model estimate"
            capacity_input_note = "Inputs are within the production model's training range."
    elif capacity_model and capacity_model.get("model") is not None:
        capacity_source = "Historical benchmark estimate (model needs retraining)"
        capacity_input_note = (
            "The saved production model uses an older feature set, so the planner used the historical benchmark instead. "
            "Retrain the models to enable the updated capacity predictor."
        )
    daily_target = int(math.ceil(quantity / max(handover_days, 1)))
    max_line_shifts = int(available_lines * max_shifts)
    shift_scenarios = [
        {
            "name": "One shift operating" if shift_count == 1 else "Both shifts operating",
            "active_shifts": shift_count,
            "lines": available_lines,
            "line_shifts": available_lines * shift_count,
            "required_operators": available_lines * shift_count * POSTER_OPERATORS_PER_LINE,
            "lower_output": low_rate * available_lines * shift_count,
            "expected_output": median_rate * available_lines * shift_count,
            "upper_output": high_rate * available_lines * shift_count,
            "target_output": daily_target,
        }
        for shift_count in range(1, FACTORY_SHIFTS_PER_DAY + 1)
    ]

    def make_plan(name, rate_basis, conservative=False, use_all=False):
        rate = low_rate if conservative else median_rate
        needed = int(math.ceil(daily_target / max(rate, 0.01)))
        count = max_line_shifts if use_all else min(needed, max_line_shifts)
        count = max(1, count)
        low = low_rate * count
        expected = median_rate * count
        high = high_rate * count
        basis_output = rate * count
        groups = _make_groups(count, available_lines)
        line_target = daily_target / count
        for group in groups:
            group.update({
                "grade_requirements": dict(POSTER_GRADE_MIX),
                "required_operators": POSTER_OPERATORS_PER_LINE,
                "target_pieces_per_day": line_target,
                "lower_pieces_per_day": low_rate,
                "expected_pieces_per_day": median_rate,
                "upper_pieces_per_day": high_rate,
            })
        grade_totals = {
            grade: workers * count
            for grade, workers in POSTER_GRADE_MIX.items()
        }
        shift_summary = []
        for shift_name in dict.fromkeys(group["shift"] for group in groups):
            shift_groups = [group for group in groups if group["shift"] == shift_name]
            shift_summary.append({
                "shift": shift_name,
                "lines": [group["line"] for group in shift_groups],
                "line_count": len(shift_groups),
                "required_operators": len(shift_groups) * POSTER_OPERATORS_PER_LINE,
                "target_pieces_per_day": line_target * len(shift_groups),
                "lower_pieces_per_day": low_rate * len(shift_groups),
                "expected_pieces_per_day": median_rate * len(shift_groups),
                "upper_pieces_per_day": high_rate * len(shift_groups),
            })

        # Turn the aggregate capacity recommendation into a usable daily
        # production target. Allocate whole pieces evenly across active
        # line-shifts, then roll those allocations up by shift. This is a
        # target schedule, not a claim that every line has identical measured
        # performance (the source data has no line-level output history).
        daily_schedule = []
        cumulative_target = 0
        base_day_target, extra_day_targets = divmod(quantity, max(handover_days, 1))
        for day_number in range(1, max(handover_days, 1) + 1):
            day_quantity = base_day_target + (1 if day_number <= extra_day_targets else 0)
            base_line_target, extra_line_targets = divmod(day_quantity, len(groups))
            group_targets = {
                group["line_shift_no"]: base_line_target
                + (1 if index < extra_line_targets else 0)
                for index, group in enumerate(groups)
            }
            for shift in shift_summary:
                assigned_groups = [
                    group for group in groups if group["shift"] == shift["shift"]
                ]
                shift_target = sum(
                    group_targets[group["line_shift_no"]] for group in assigned_groups
                )
                cumulative_target += shift_target
                daily_schedule.append({
                    "day": day_number,
                    "shift": shift["shift"],
                    "lines": [group["line"] for group in assigned_groups],
                    "line_count": len(assigned_groups),
                    "required_operators": len(assigned_groups) * POSTER_OPERATORS_PER_LINE,
                    "target_pieces": shift_target,
                    "cumulative_target": cumulative_target,
                })
        return {
            "name": name,
            "basis": rate_basis,
            "line_shifts": count,
            "line_shifts_required": needed,
            "line_shift_gap": max(0, needed - count),
            "groups": groups,
            "shift_summary": shift_summary,
            "daily_schedule": daily_schedule,
            "scheduled_order_quantity": quantity,
            "lines": min(available_lines, count),
            "shifts": int(math.ceil(count / max(available_lines, 1))),
            "daily_output_low": low,
            "daily_output": expected,
            "daily_output_high": high,
            "basis_daily_output": basis_output,
            "estimated_days": int(math.ceil(quantity / max(expected, 1))),
            "conservative_days": int(math.ceil(quantity / max(low, 1))),
            "meets_target": basis_output >= daily_target,
            "capacity_ratio": basis_output / max(daily_target, 1),
            "daily_target": daily_target,
            "operators_per_line_shift": POSTER_OPERATORS_PER_LINE,
            "required_operators": POSTER_OPERATORS_PER_LINE * count,
            "grade_mix_per_line_shift": dict(POSTER_GRADE_MIX),
            "grade_totals": grade_totals,
        }

    forecast_basis = (
        "Model expected output per line-shift"
        if capacity_source.startswith("Production model") else "Historical benchmark median per line-shift"
    )
    low_basis = (
        "Model lower-range output per line-shift"
        if capacity_source.startswith("Production model") else "Historical benchmark lower range"
    )
    lean = make_plan("Median-rate plan", forecast_basis)
    buffered = make_plan("Conservative buffer plan", low_basis, conservative=True)
    delivery = make_plan("Maximum available capacity", "All entered lines across both shifts", use_all=True)
    if buffered["meets_target"]:
        recommended = buffered
        decision = "CAPACITY FIT · BUFFERED"
    elif lean["meets_target"]:
        recommended = lean
        decision = "TIGHT · MEDIAN ONLY"
    else:
        recommended = delivery
        decision = "RE-PLAN CAPACITY"

    selected_dates = pd.to_datetime(selected_history["order_date"], errors="coerce")
    if selected_dates.notna().any():
        period_start = selected_dates.min().strftime("%Y-%m-%d")
        period_end = selected_dates.max().strftime("%Y-%m-%d")
    else:
        period_start = period_end = "Unknown"

    return {
        "daily_target": daily_target,
        "available_lines": available_lines,
        "max_shifts": max_shifts,
        "shift_scenarios": shift_scenarios,
        "factory_shifts_per_day": FACTORY_SHIFTS_PER_DAY,
        "max_line_shifts": max_line_shifts,
        "shift_hours": shift_hours,
        "history_orders_used": int(len(selected_history)),
        "history_period_start": period_start,
        "history_period_end": period_end,
        "benchmark_style": benchmark_style,
        "capacity_source": capacity_source,
        "capacity_input_note": capacity_input_note,
        "rate_low": low_rate,
        "rate_median": median_rate,
        "rate_high": high_rate,
        "decision": decision,
        "recommended": recommended,
        "plans": [lean, buffered, delivery],
        "staffing_basis": "Factory poster grade mix; required headcount only, not live availability.",
        "quality_target_reference": 97,
        "process_steps": [
            "Before each shift, verify the actual available headcount by grade against the required mix and confirm machine availability for the planned lines.",
            "Start the recommended line-shifts with the listed grade mix; assign named workers and operation stations using the factory supervisor's verified roster.",
            "Run first-piece approval and in-line quality checks before increasing line speed. Use 97% as the poster's quality target reference and record actual results.",
            "Track output by line at regular hourly intervals against that line-shift's daily target and its planned output range.",
            "At shift end, record actual good pieces, rejects, downtime and attendance; feed those actuals into the next capacity re-plan.",
        ],
        "planning_note": (
            "Output rates are pooled historical estimates per staffed line-shift; the source orders do not record separate line or shift results. "
            "Do not interpret equal per-line rates as equal observed line performance. The standard 17-operator grade mix is a staffing requirement, not a live attendance count."
        ),
        "machine_note": "Machine availability is not live-connected; verify required machines on the assigned lines and shifts.",
    }
