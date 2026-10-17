"""
 Generate sample datasets for Component 4:
  - N1: Resource-conditioned probabilistic order fulfilment
  - N2: Uncertainty-aware minimum-cost intervention

 Replace these CSVs with current factory records for operational use.
"""
import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime, timedelta

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

np.random.seed(42)
os.makedirs("data", exist_ok=True)

N_ORDERS   = 3000         # larger reproducible sample for training demonstrations
N_WORKERS  = 150
N_MACHINES = 60

# ==================================================================
# 1.  ORDERS  (with realistic on_time_flag)
# ==================================================================
styles     = ["Polo", "T-Shirt", "Hoodie", "Legging", "Jacket"]
customers  = [f"C{i:03d}" for i in range(1, 21)]
priorities = ["Low", "Normal", "High"]
style_smv_ranges = {
    "Polo": (10, 16), "T-Shirt": (7, 13), "Hoodie": (18, 32),
    "Legging": (10, 18), "Jacket": (22, 40),
}
style_price_ranges_lkr = {
    "Polo": (200, 320), "T-Shirt": (150, 260), "Hoodie": (400, 800),
    "Legging": (180, 350), "Jacket": (500, 1000),
}
style_capacity_factor = {"Polo": 1.00, "T-Shirt": 1.02, "Hoodie": 0.94,
                         "Legging": 0.98, "Jacket": 0.92}
priority_capacity_factor = {"Low": 1.04, "Normal": 1.00, "High": 0.90}

orders = []
for i in range(1, N_ORDERS + 1):
    qty        = int(np.random.choice([2000, 3000, 5000, 8000, 10000, 15000],
                                       p=[0.15, 0.20, 0.25, 0.20, 0.15, 0.05]))
    style      = np.random.choice(styles)
    smv_total  = round(np.random.uniform(*style_smv_ranges[style]), 2)
    handover_d = int(np.random.choice([7, 10, 14, 20, 30],
                                       p=[0.15, 0.30, 0.30, 0.15, 0.10]))
    order_val  = round(qty * np.random.uniform(*style_price_ranges_lkr[style]), 2)
    priority   = np.random.choice(priorities, p=[0.2, 0.6, 0.2])

    # ---- Total required workload (worker-minutes) ----
    required_workload = qty * smv_total

    # ---- Available capacity depends on:
    #      worker count, avg efficiency, shift hours, handover days ----
    n_available_workers = int(np.random.randint(20, 80))          # varies per order
    avg_worker_eff      = np.random.uniform(55, 90)               # varies 55-90%
    shift_hours         = 8
    n_shifts            = np.random.choice([1, 2], p=[0.6, 0.4])
    efficiency_factor   = avg_worker_eff / 100.0

    # Effective daily capacity in worker-minutes
    daily_capacity = n_available_workers * shift_hours * 60 * efficiency_factor * n_shifts
    total_capacity = daily_capacity * handover_d

    # ---- Feasibility ratio ----
    feasibility_ratio = (
        total_capacity / max(required_workload, 1)
        * style_capacity_factor[style]
        * priority_capacity_factor[priority]
    )   # >1 means surplus after style and priority adjustments

    # ---- Probability of on-time via sigmoid centered at 1.0 ----
    #      ratio 1.0 -> 50%, ratio 1.5 -> ~88%, ratio 0.5 -> ~12%
    logit    = 5.0 * (feasibility_ratio - 1.0)
    p_ontime = 1.0 / (1.0 + np.exp(-logit))
    p_ontime = float(np.clip(p_ontime, 0.02, 0.98))

    on_time = 1 if np.random.rand() < p_ontime else 0

    # ---- Consistent completion dates ----
    start = datetime(2024, 1, 1) + timedelta(days=int(np.random.randint(0, 300)))
    handover = start + timedelta(days=handover_d)

    if on_time:
        finish_offset = int(np.random.randint(max(1, handover_d - 4), handover_d + 1))
    else:
        finish_offset = int(np.random.randint(handover_d + 1, handover_d + 12))
    actual_completion = start + timedelta(days=finish_offset)

    # If late, completed qty tends to be smaller
    completion_ratio = np.random.uniform(0.85, 1.02) if on_time else np.random.uniform(0.60, 0.95)
    actual_completed_qty = int(qty * completion_ratio)

    orders.append({
        "order_id": i,
        "order_date": start.date(),
        "customer_id": np.random.choice(customers),
        "style_id": style,
        "garment_type": style,
        "order_quantity": qty,
        "handover_date": handover.date(),
        "handover_days": handover_d,
        "order_value": order_val,
        "priority": priority,
        "required_smv_total": smv_total,
        "required_workload_min": round(required_workload, 2),
        "n_available_workers": n_available_workers,
        "avg_worker_efficiency": round(avg_worker_eff, 2),
        "n_shifts": n_shifts,
        "shift_hours": shift_hours,
        "feasibility_ratio": round(feasibility_ratio, 4),
        "actual_completion_date": actual_completion.date(),
        "actual_completed_qty": actual_completed_qty,
        "on_time_flag": on_time,
        "order_status": "Accepted"
    })

df_orders = pd.DataFrame(orders)
df_orders.to_csv("data/orders.csv", index=False)
print(f"✅ orders.csv            — {len(df_orders)} rows  "
      f"(on-time rate = {df_orders['on_time_flag'].mean():.2%})")

# ==================================================================
# 2.  WORKERS / SKILL MATRIX
# ==================================================================
grades         = ["C", "B", "A", "A+", "Jumper"]
machine_skills = ["SNLS", "Overlock", "Flatlock", "Bartack", "Buttonhole", "Kansai"]

workers = []
for w in range(1, N_WORKERS + 1):
    grade = np.random.choice(grades, p=[0.15, 0.30, 0.30, 0.15, 0.10])
    exp   = int(np.random.randint(3, 180))

    # grade correlates with efficiency & quality
    base_eff = {"C": 62, "B": 72, "A": 80, "A+": 87, "Jumper": 92}[grade]
    eff  = round(np.clip(np.random.normal(base_eff, 5), 40, 100), 2)
    qual = round(np.clip(np.random.normal(base_eff + 8, 4), 70, 100), 2)
    att  = round(np.clip(np.random.normal(93, 5), 70, 100), 2)
    skills = ", ".join(np.random.choice(machine_skills,
                                         size=np.random.randint(1, 4), replace=False))
    workers.append({
        "worker_id": w,
        "grade": grade,
        "experience_months": exp,
        "training_status": np.random.choice(["Trained", "Untrained"], p=[0.8, 0.2]),
        "attendance_rate": att,
        "availability_status": np.random.choice(
            ["Available", "Allocated", "Leave"], p=[0.70, 0.25, 0.05]),
        "machine_skills": skills,
        "efficiency_avg": eff,
        "quality_avg": qual,
        "rework_rate": round(np.clip(np.random.normal(3, 1.5), 0.5, 10), 2),
        "previous_output": int(np.random.randint(50, 400))
    })

df_workers = pd.DataFrame(workers)
df_workers.to_csv("data/workers.csv", index=False)
print(f"✅ workers.csv           — {len(df_workers)} rows")

# ==================================================================
# 3.  OPERATIONS
# ==================================================================
op_names = ["Collar attach", "Collar topst", "Sleeve attach", "Sleeve hem",
            "Side seam", "Bottom hem", "Cuff attach", "Placket attach",
            "Pocket attach", "Label attach", "Shoulder attach", "Neck piping"]
op_cats  = ["Basic", "Intermediate", "Critical"]
m_types  = ["SNLS", "Overlock", "Flatlock", "Bartack", "Buttonhole", "Kansai"]

ops = []
for i, name in enumerate(op_names, 1):
    ops.append({
        "operation_id": i,
        "operation_name": name,
        "operation_category": np.random.choice(op_cats, p=[0.4, 0.4, 0.2]),
        "machine_type_required": np.random.choice(m_types),
        "smv": round(np.random.uniform(0.5, 3.5), 2),
        "sequence_no": i,
        "complexity_score": round(np.random.uniform(1, 10), 2),
        "quality_requirement": round(np.random.uniform(88, 98), 2)
    })

df_ops = pd.DataFrame(ops)
df_ops.to_csv("data/operations.csv", index=False)
print(f"✅ operations.csv        — {len(df_ops)} rows")

# ==================================================================
# 4.  ALLOCATIONS  (worker efficiency now correlates with actual output)
# ==================================================================
allocs = []
alloc_id = 1
for _, o in df_orders.iterrows():
    n_ops = np.random.randint(4, 10)
    # number of workers engaged in this order
    n_engaged = int(np.random.randint(5, min(25, N_WORKERS) + 1))
    engaged_workers = df_workers.sample(n_engaged).reset_index(drop=True)

    for k in range(n_ops):
        op = df_ops.sample(1).iloc[0]
        w  = engaged_workers.sample(1).iloc[0]
        start = datetime.strptime(str(o["order_date"]), "%Y-%m-%d")
        dur   = int(np.random.randint(3, 15))

        # ----- Actual efficiency now depends on worker's avg efficiency + noise -----
        eff  = round(np.clip(np.random.normal(w["efficiency_avg"], 4), 30, 100), 2)
        qual = round(np.clip(np.random.normal(w["quality_avg"],     3), 60, 100), 2)

        # ----- Output scales with worker efficiency -----
        allocated_workload = int(np.random.randint(200, 3000))
        output = int(allocated_workload * (eff / 100.0) * np.random.uniform(0.9, 1.05))
        rework = int(output * (100 - qual) / 100.0 * 0.1)

        allocs.append({
            "allocation_id": alloc_id,
            "order_id": o["order_id"],
            "worker_id": w["worker_id"],
            "operation_id": op["operation_id"],
            "line_id": int(np.random.randint(1, 8)),
            "machine_id": int(np.random.randint(1, N_MACHINES + 1)),
            "assigned_start": start.date(),
            "assigned_end": (start + timedelta(days=dur)).date(),
            "allocated_workload": allocated_workload,
            "allocation_status": np.random.choice(["Active", "Completed"], p=[0.3, 0.7]),
            "assigned_by": np.random.choice(["IE01", "IE02", "IE03"]),
            "conflict_flag": np.random.choice([0, 1], p=[0.85, 0.15]),
            "actual_efficiency": eff,
            "actual_quality": qual,
            "actual_output": output,
            "rework_qty": rework
        })
        alloc_id += 1

df_allocs = pd.DataFrame(allocs)
df_allocs.to_csv("data/allocations.csv", index=False)
print(f"✅ allocations.csv       — {len(df_allocs)} rows")

# ==================================================================
# 5. MACHINES / SHIFTS
# ==================================================================
rows = []
for m in range(1, N_MACHINES + 1):
    for d in range(30):
        shift_id = np.random.choice(["A", "B"])
        shift_start, shift_end = ("06:00", "14:00") if shift_id == "A" else ("14:00", "22:00")
        rows.append({
            "machine_id": m,
            "machine_type": np.random.choice(m_types),
            "line_id": int(np.random.randint(1, 8)),
            "date": (datetime(2024, 1, 1) + timedelta(days=d)).date(),
            "status": np.random.choice(["Available", "Down"], p=[0.92, 0.08]),
            "downtime_hours": round(np.random.choice([0, 0, 0, 1, 2, 4]), 2),
            "shift_id": shift_id,
            "start_time": shift_start,
            "end_time":   shift_end,
            "working_hours": 8,
            "overtime_allowed": np.random.choice([0, 1], p=[0.4, 0.6])
        })
df_ms = pd.DataFrame(rows)
df_ms.to_csv("data/machines_shifts.csv", index=False)
print(f"✅ machines_shifts.csv   — {len(df_ms)} rows")

# ==================================================================
# 6. INTERVENTIONS  (scenario-level, with realistic cost & P(On-Time))
# ==================================================================
scenarios = []
sid = 1
for _, o in df_orders.iterrows():
    req_workload = o["required_workload_min"]
    base_cap     = req_workload / max(o["feasibility_ratio"], 0.05)   # back out capacity

    for scen in ["Current", "Overtime", "Extra_Workers",
                 "Machine_Adjust", "Outsourcing", "Hybrid"]:

        # --- Define parameters per scenario ---
        if scen == "Current":
            added_w, ot_h, add_m, outs_q = 0, 0, 0, 0
            cap_mult = 1.00
        elif scen == "Overtime":
            added_w, ot_h, add_m, outs_q = 0, np.random.randint(1, 4), 0, 0
            cap_mult = 1.0 + 0.08 * ot_h
        elif scen == "Extra_Workers":
            added_w = np.random.randint(3, 15)
            ot_h, add_m, outs_q = 0, 0, 0
            cap_mult = 1.0 + 0.02 * added_w
        elif scen == "Machine_Adjust":
            add_m   = np.random.randint(1, 5)
            added_w, ot_h, outs_q = 0, 0, 0
            cap_mult = 1.0 + 0.03 * add_m
        elif scen == "Outsourcing":
            outs_q  = int(o["order_quantity"] * np.random.uniform(0.2, 0.5))
            added_w, ot_h, add_m = 0, 0, 0
            cap_mult = 1.0 + (outs_q / o["order_quantity"]) * 0.9
        else:  # Hybrid
            added_w = np.random.randint(2, 8)
            ot_h    = np.random.randint(1, 3)
            add_m   = np.random.randint(1, 3)
            outs_q  = int(o["order_quantity"] * np.random.uniform(0.1, 0.3))
            cap_mult = 1.0 + 0.02 * added_w + 0.08 * ot_h + 0.03 * add_m \
                       + (outs_q / o["order_quantity"]) * 0.9

        # --- P(On-Time) driven by capacity multiplier ---
        new_ratio = o["feasibility_ratio"] * cap_mult
        p_on = 1.0 / (1.0 + np.exp(-5.0 * (new_ratio - 1.0)))
        p_on = float(np.clip(p_on + np.random.normal(0, 0.03), 0.02, 0.99))

        # --- Costs ---
        labour_c  = added_w * np.random.uniform(1200, 1800) * 10
        ot_cost   = ot_h * 5000
        hire_cost = added_w * 3000
        mach_cost = add_m * 2500
        unit_value = o["order_value"] / o["order_quantity"]
        out_cost  = outs_q * unit_value * np.random.uniform(0.45, 0.75)
        total_c   = labour_c + ot_cost + hire_cost + mach_cost + out_cost

        scenarios.append({
            "scenario_id": sid,
            "order_id": o["order_id"],
            "scenario_type": scen,
            "added_workers": added_w,
            "overtime_hours": ot_h,
            "added_machines": add_m,
            "outsourced_qty": outs_q,
            "labour_cost": round(labour_c, 2),
            "overtime_cost": round(ot_cost, 2),
            "hiring_cost": round(hire_cost, 2),
            "machine_adjustment_cost": round(mach_cost, 2),
            "outsourcing_cost": round(out_cost, 2),
            "total_cost": round(total_c, 2),
            "expected_P_OnTime": round(p_on, 3),
            "expected_profit": round(o["order_value"] - total_c, 2),
            "selected_flag": 0,
            "decision": "Review"
        })
        sid += 1

df_scen = pd.DataFrame(scenarios)

# ==================================================================
# 7. MARK SELECTED SCENARIO PER ORDER (top-2 min-cost feasible)
# ==================================================================
print("→ Marking selected scenarios (top-2 min-cost with P>=0.80)…")

# Priority: prefer "Current" if it already meets criterion, else min-cost intervention
for oid in df_scen["order_id"].unique():
    sub = df_scen[df_scen["order_id"] == oid]
    feasible = sub[sub["expected_P_OnTime"] >= 0.80]

    if len(feasible) == 0:
        # No feasible scenario → mark highest-P scenario as "Risky" but not selected
        continue

    # Prefer Current if feasible
    current = feasible[feasible["scenario_type"] == "Current"]
    if len(current) > 0:
        idx = current.index[0]
        df_scen.loc[idx, "selected_flag"] = 1
        df_scen.loc[idx, "decision"]      = "Accept"
    else:
        # Mark top-2 cheapest feasible scenarios as selected (helps ML learn)
        top2 = feasible.nsmallest(2, "total_cost").index
        for idx in top2:
            df_scen.loc[idx, "selected_flag"] = 1
            df_scen.loc[idx, "decision"]      = "Accept with Intervention"

df_scen.to_csv("data/interventions.csv", index=False)
print(f"✅ interventions.csv     — {len(df_scen)} rows  "
      f"(selected = {df_scen['selected_flag'].mean():.2%})")

# ==================================================================
# 8. SUMMARY
# ==================================================================
print("\n" + "="*60)
print("📊 DATASET SUMMARY")
print("="*60)
print(f"Orders        : {len(df_orders)}")
print(f"Workers       : {len(df_workers)}")
print(f"Operations    : {len(df_ops)}")
print(f"Allocations   : {len(df_allocs)}")
print(f"Machines×days : {len(df_ms)}")
print(f"Scenarios     : {len(df_scen)}")
print(f"\nOn-time rate  : {df_orders['on_time_flag'].mean():.2%}")
print(f"Selected rate : {df_scen['selected_flag'].mean():.2%}  (N2 target)")
print("\nFiles written to: data/")
for f in sorted(os.listdir("data")):
    print("  data/" + f)
