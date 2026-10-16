"""Skill-aware line/shift staffing plans for apparel orders."""
import math

import numpy as np
import pandas as pd

from workforce_model_features import (
    GRADE_RANK,
    WORKER_OPERATION_FEATURES,
    make_worker_operation_features,
)

LINE_GRADE_MIX = {"Jumper": 5, "A+": 4, "A": 4, "B": 3, "C": 1}
WORKERS_PER_LINE_SHIFT = sum(LINE_GRADE_MIX.values())
QUALITY_TARGET = 97.0
EFFICIENCY_TARGET = 70.0


def _build_performance_cache(workers, operations, model):
    """Predict each worker/operation pair once, in one vectorized model call."""
    cache = {}
    if model and workers and operations:
        rows = []
        keys = []
        for worker in workers:
            for operation in operations:
                rows.append(make_worker_operation_features(worker, operation))
                keys.append((int(worker["worker_id"]), int(operation["operation_id"])))
        predictions = model["model"].predict(
            pd.DataFrame(rows, columns=WORKER_OPERATION_FEATURES)
        )
        cache = {
            key: (float(np.clip(pred[0], 35, 110)), float(np.clip(pred[1], 50, 100)))
            for key, pred in zip(keys, predictions)
        }
    return cache


def _predict_worker_operation(worker, operation, model, performance_cache):
    key = (int(worker["worker_id"]), int(operation["operation_id"]))
    if key in performance_cache:
        return performance_cache[key]
    return (
        float(np.clip(worker.get("efficiency_avg", 70), 35, 110)),
        float(np.clip(worker.get("quality_avg", 85), 50, 100)),
    )


def _station_counts(operations, smv):
    ops = operations.to_dict(orient="records")
    if not ops:
        return [], {}
    adjusted = {
        int(op["operation_id"]): float(smv) * float(op["smv"]) /
        max(float(operations["smv"].sum()), 0.01)
        for op in ops
    }
    counts = {int(op["operation_id"]): 1 for op in ops}
    for _ in range(max(0, WORKERS_PER_LINE_SHIFT - len(counts))):
        # Add each remaining station to the operation currently limiting line output.
        bottleneck = max(counts, key=lambda op_id: adjusted[op_id] / counts[op_id])
        counts[bottleneck] += 1
    return ops, {op_id: (count, adjusted[op_id]) for op_id, count in counts.items()}


def _assign_group(line_no, shift_no, available_workers, operations, station_plan,
                  performance_model, performance_cache, objective, shift_hours,
                  machine_snapshot):
    op_by_id = {int(op["operation_id"]): op for op in operations}
    slots = []
    for op_id, (count, _) in station_plan.items():
        slots.extend([op_id] * count)
    # Fill scarce machine skills before common skills.
    def scarcity(op_id):
        machine = str(op_by_id[op_id]["machine_type_required"]).casefold()
        eligible = sum(
            machine in {x.strip().casefold() for x in str(w.get("machine_skills", "")).split(",")}
            for w in available_workers
        )
        return (eligible, -float(op_by_id[op_id]["smv"]))

    slots.sort(key=scarcity)
    grade_remaining = dict(LINE_GRADE_MIX)
    assignments = []
    remaining = list(available_workers)

    for op_id in slots:
        operation = op_by_id[op_id]
        machine = str(operation["machine_type_required"]).casefold()
        skilled = [w for w in remaining if machine in {
            x.strip().casefold() for x in str(w.get("machine_skills", "")).split(",")
        }]
        candidate_pool = skilled or remaining
        if not candidate_pool:
            break
        grade_fit = [w for w in candidate_pool if grade_remaining.get(str(w.get("grade", "C")), 0) > 0]
        if grade_fit:
            candidate_pool = grade_fit

        scored = []
        for worker in candidate_pool:
            efficiency, quality = _predict_worker_operation(
                worker, operation, performance_model, performance_cache
            )
            attendance = float(worker.get("attendance_rate", 90)) / 100
            rework = float(worker.get("rework_rate", 0))
            if objective == "quality":
                score = quality * 0.68 + efficiency * 0.32 - rework * 0.8
            elif objective == "delivery":
                score = efficiency * 0.68 + quality * 0.32 - rework * 0.4
            else:
                score = efficiency * 0.52 + quality * 0.48 - rework * 0.5
            score *= attendance
            if machine not in {x.strip().casefold() for x in str(worker.get("machine_skills", "")).split(",")}:
                score -= 35
            score -= GRADE_RANK.get(str(worker.get("grade", "C")), 0) * 0.05
            scored.append((score, worker, efficiency, quality))
        _, chosen, efficiency, quality = max(scored, key=lambda x: x[0])
        remaining.remove(chosen)
        grade = str(chosen.get("grade", "C"))
        if grade_remaining.get(grade, 0) > 0:
            grade_remaining[grade] -= 1
        skill_match = machine in {
            x.strip().casefold() for x in str(chosen.get("machine_skills", "")).split(",")
        }
        assignments.append({
            "line": line_no, "shift": shift_no,
            # Anonymous slot only; source worker IDs stay internal to matching.
            "operator_slot": len(assignments) + 1, "grade": grade,
            "operation": operation["operation_name"],
            "machine": operation["machine_type_required"],
            "efficiency": efficiency, "quality": quality,
            "attendance": float(chosen.get("attendance_rate", 90)),
            "skill_match": skill_match,
        })

    operation_results = []
    quality_weighted = 0.0
    total_slots = 0
    machine_key = (line_no, "ABC"[shift_no - 1])
    machine_info = machine_snapshot["by_line_shift"].get(machine_key)
    # Machine logs are useful context, but this historical snapshot is not a
    # live inventory. Never convert missing rows into zero production capacity.
    machine_counts = machine_info["counts"] if machine_info else {}
    machine_gaps = 0
    machine_slots_by_op = {}
    for op_id in station_plan:
        op = op_by_id[op_id]
        needed = station_plan[op_id][0]
        machine_type = str(op["machine_type_required"])
        # Counts are shown as a reference only; they are not enforced against
        # plans because the date does not establish current machine availability.
        machine_slots_by_op[op_id] = machine_counts.get(machine_type) if machine_info else None

    for op_id, (stations, adjusted_smv) in station_plan.items():
        rows = [a for a in assignments if a["operation"] == op_by_id[op_id]["operation_name"]]
        minutes = sum(
            float(r["attendance"]) / 100 * float(r["efficiency"]) / 100 *
            float(shift_hours) * 60
            for r in rows
        )
        capacity = minutes / max(adjusted_smv, 0.01)
        machine_slots = machine_slots_by_op[op_id]
        op_quality = (
            sum(r["quality"] for r in rows) / len(rows) if rows else 0.0
        )
        operation_results.append({
            "operation": op_by_id[op_id]["operation_name"],
            "machine": op_by_id[op_id]["machine_type_required"],
            "smv": adjusted_smv, "operators": len(rows),
            "shift_capacity": capacity, "predicted_quality": op_quality,
            "quality_requirement": float(op_by_id[op_id]["quality_requirement"]),
            "skill_gap": sum(not r["skill_match"] for r in rows),
            "machine_slots": machine_slots,
            "machine_shortage": None,
        })
        quality_weighted += op_quality * max(adjusted_smv, 0.01)
        total_slots += max(adjusted_smv, 0.01)

    output = min((row["shift_capacity"] for row in operation_results), default=0.0)
    quality = quality_weighted / max(total_slots, 0.01)
    efficiency = (
        sum(a["efficiency"] * a["attendance"] for a in assignments) /
        max(sum(a["attendance"] for a in assignments), 0.01)
    )
    return {
        "line": line_no, "shift": shift_no, "assignments": assignments,
        "operations": operation_results, "output": output,
        "quality": quality, "efficiency": efficiency, "operators": len(assignments),
        "grade_counts": pd.Series([a["grade"] for a in assignments]).value_counts().to_dict(),
        "skill_gaps": sum(not a["skill_match"] for a in assignments),
        "machine_gaps": machine_gaps,
        "machine_snapshot_known": machine_info is not None,
    }, remaining


def build_workforce_plans(workers, operations, quantity, handover_days, smv,
                          available_lines, max_shifts, shift_hours,
                          performance_model=None, machines=None):
    roster = workers[workers["availability_status"].eq("Available")].copy()
    roster = roster.sort_values(["attendance_rate", "experience_months"], ascending=False)
    available = roster.to_dict(orient="records")
    ops, station_plan = _station_counts(operations, smv)
    if not ops:
        return {"error": "No operation records are available for line balancing."}

    max_groups = min(available_lines * max_shifts, len(available) // WORKERS_PER_LINE_SHIFT)
    daily_target = int(math.ceil(quantity / max(handover_days, 1)))

    machine_snapshot = {"date": None, "by_line_shift": {}}
    if machines is not None and not machines.empty:
        machine_dates = pd.to_datetime(machines["date"], errors="coerce")
        if machine_dates.notna().any():
            latest = machine_dates.max()
            current = machines.loc[machine_dates.eq(latest)].copy()
            machine_snapshot["date"] = latest.strftime("%Y-%m-%d")
            for (line_id, shift_id), group in current.groupby(["line_id", "shift_id"]):
                available_rows = group[group["status"].eq("Available")]
                machine_snapshot["by_line_shift"][(int(line_id), str(shift_id))] = {
                    "counts": available_rows.groupby("machine_type").size().to_dict(),
                    "downtime": available_rows.groupby("machine_type")["downtime_hours"].mean().to_dict(),
                }

    def construct(objective):
        pool = list(available)
        groups = []
        for n in range(max_groups):
            line_no = n % available_lines + 1
            shift_no = n // available_lines + 1
            group, pool = _assign_group(
                line_no, shift_no, pool, ops, station_plan,
                performance_model, performance_cache, objective, shift_hours,
                machine_snapshot,
            )
            if group["operators"] < WORKERS_PER_LINE_SHIFT:
                break
            group["shift_output"] = group["output"]
            groups.append(group)
        return groups

    performance_cache = _build_performance_cache(available, ops, performance_model)
    candidates = {}
    for objective in ("balanced", "quality", "delivery"):
        groups = construct(objective)
        output_by_prefix = np.cumsum([g["output"] for g in groups]).tolist()
        feasible_count = next((i + 1 for i, cap in enumerate(output_by_prefix)
                               if cap >= daily_target), None)
        candidates[objective] = (groups, output_by_prefix, feasible_count)

    balanced_groups, balanced_output, balanced_count = candidates["balanced"]
    quality_groups, quality_output, quality_count = candidates["quality"]
    delivery_groups, delivery_output, _ = candidates["delivery"]

    def summarize(name, groups, prefix_output, count, objective):
        selected = groups[:count] if count else groups
        output = float(sum(g["output"] for g in selected))
        quality = (
            sum(g["quality"] * max(g["output"], 0.01) for g in selected) /
            max(output, 0.01)
        ) if selected else 0.0
        efficiency = (
            sum(g["efficiency"] * g["operators"] for g in selected) /
            max(sum(g["operators"] for g in selected), 1)
        ) if selected else 0.0
        unique_lines = len({g["line"] for g in selected})
        shifts_used = max((g["shift"] for g in selected), default=0)
        skill_gaps = sum(g["skill_gaps"] for g in selected)
        machine_gaps = sum(g["machine_gaps"] for g in selected)
        return {
            "name": name, "groups": selected,
            "line_shifts": len(selected), "lines": unique_lines,
            "shifts": shifts_used, "daily_output": output,
            "operators": sum(g["operators"] for g in selected),
            "predicted_quality": quality,
            "predicted_efficiency": efficiency,
            "quality_pass": quality >= QUALITY_TARGET,
            "skill_gaps": skill_gaps,
            "machine_gaps": machine_gaps,
            "meets_target": output >= daily_target,
            "daily_target": daily_target,
            "shortfall": max(0.0, daily_target - output),
            "objective": objective,
            "line_days": len(selected) * handover_days,
        }

    cost_plan = summarize(
        "Cost focused", balanced_groups, balanced_output, balanced_count,
        "Use the fewest line-shifts predicted to meet the daily order target.",
    )
    quality_plan = summarize(
        "Quality focused", quality_groups, quality_output, quality_count,
        "Prefer operators with stronger predicted quality while preserving machine-skill fit.",
    )
    delivery_plan = summarize(
        "Delivery focused", delivery_groups, delivery_output, len(delivery_groups),
        "Use the largest feasible line-shift team to maximize daily output.",
    )

    if quality_plan["meets_target"] and quality_plan["quality_pass"]:
        recommended = quality_plan
    elif cost_plan["meets_target"]:
        recommended = cost_plan
    else:
        recommended = delivery_plan

    supply_by_grade = roster.groupby("grade").size().to_dict()
    return {
        "daily_target": daily_target,
        "available_workers": len(available),
        "available_lines": available_lines,
        "available_by_grade": supply_by_grade,
        "max_shifts": max_shifts,
        "max_line_shifts": max_groups,
        "workers_per_line_shift": WORKERS_PER_LINE_SHIFT,
        "shift_hours": shift_hours,
        "machine_snapshot_date": machine_snapshot["date"],
        "machine_snapshot_is_current": False,
        "machine_data_note": (
            f"Machine log reference date {machine_snapshot['date']}; confirm live line/shift machine availability."
            if machine_snapshot["date"] else
            "No dated machine availability snapshot is available; confirm live line/shift machine availability."
        ),
        "required_output_smv": smv,
        "quality_target": QUALITY_TARGET,
        "efficiency_target": EFFICIENCY_TARGET,
        "performance_source": (
            "trained worker-operation model" if performance_model
            else "historical worker averages because no performance model is loaded"
        ),
        "recommended": recommended,
        "plans": [cost_plan, quality_plan, delivery_plan],
        "scenarios": [cost_plan, quality_plan, delivery_plan],
        "line_mix": LINE_GRADE_MIX,
    }
