"""
 Flask app: loads the auto-selected best models from the comparison report.
"""
from flask import Flask, render_template, request
import os
import numpy as np
import pandas as pd, joblib, json
from model_features import (
    SCENARIO_ENCODING,
    STYLE_CAPACITY_FACTOR,
    PRIORITY_CAPACITY_FACTOR,
    build_factory_profile,
    build_order_features,
    make_intervention_features,
)

app = Flask(__name__)

# The datasets are the app's current factory profile. Allocation values are
# summarized per order so the model receives features on the same scale as it
# saw during training, while the user only has to enter the new order details.
orders_data = pd.read_csv("data/orders.csv")
allocations_data = pd.read_csv("data/allocations.csv")

with open("models/model_comparison.json") as f:
    REPORT = json.load(f)

def load_model_bundle(path):
    bundle = joblib.load(path)
    if bundle.get("type") == "keras" and bundle.get("model_path"):
        import tensorflow as tf

        model_path = bundle["model_path"]
        if not os.path.isabs(model_path):
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path)
        bundle["model"] = tf.keras.models.load_model(model_path)
    return bundle


best_n1_bundle = load_model_bundle("models/best_n1_model.pkl")
best_n2_bundle = load_model_bundle("models/best_n2_model.pkl")
best_cost      = joblib.load("models/best_cost_model.pkl")
FACTORY = best_n1_bundle.get("profile") or build_factory_profile(orders_data, allocations_data)
N1_FEATURES = [
    "order_quantity", "handover_days", "required_smv_total", "order_value",
    "style_enc", "priority_enc", "unit_value", "estimated_workload_min",
    "estimated_workload_per_day", "capacity_ratio",
]
n1_model_ready = (
    best_n1_bundle.get("training_signature") == "six-order-inputs-v1"
    and best_n1_bundle.get("features") == N1_FEATURES
)
n2_model_ready = best_n2_bundle.get("training_signature") == "six-order-inputs-v1"
best_n1_name = best_n1_bundle["name"] if n1_model_ready else "Capacity profile fallback"
best_n2_name = best_n2_bundle["name"] if n2_model_ready else "Capacity-based fallback"
n1_features = N1_FEATURES
n2_features = best_n2_bundle["features"]

def predict_proba(bundle, X):
    """Works for both keras & sklearn bundles."""
    model = bundle["model"]
    if bundle["type"] == "keras":
        scaled = bundle["scaler"].transform(np.asarray(X, dtype=np.float32))
        return model.predict(scaled, verbose=0).ravel()
    return model.predict_proba(X)[:, 1]

SCENARIO_MAP = SCENARIO_ENCODING

def evaluate_scenarios(base_input):
    results = []
    for sname in SCENARIO_MAP:
        aw = {"Current":0,"Overtime":0,"Extra_Workers":5,"Machine_Adjust":0,"Outsourcing":0,"Hybrid":3}[sname]
        ot = {"Current":0,"Overtime":3,"Extra_Workers":0,"Machine_Adjust":0,"Outsourcing":0,"Hybrid":2}[sname]
        am = {"Current":0,"Overtime":0,"Extra_Workers":0,"Machine_Adjust":3,"Outsourcing":0,"Hybrid":2}[sname]
        oq = {"Current":0,"Overtime":0,"Extra_Workers":0,"Machine_Adjust":0,
              "Outsourcing":int(base_input["order_quantity"]*0.3),
              "Hybrid":int(base_input["order_quantity"]*0.2)}[sname]
        scenario = {"scenario_type": sname, "added_workers": aw,
                    "overtime_hours": ot, "added_machines": am,
                    "outsourced_qty": oq}
        inferred = build_order_features(
            base_input["order_quantity"], base_input["handover_days"],
            base_input["required_smv_total"], base_input["order_value"],
            base_input["style"], base_input["priority"], FACTORY,
        )
        row = make_intervention_features(inferred, scenario, FACTORY)
        vector = pd.DataFrame([row], columns=n2_features)
        if n2_model_ready:
            p = float(predict_proba(best_n2_bundle, vector)[0])
            cost = max(0.0, float(best_cost.predict(vector)[0]))
        else:
            capacity_ratio = inferred["capacity_ratio"]
            capacity_multiplier = {
                "Current": 1.0,
                "Overtime": 1 + 0.08 * ot,
                "Extra_Workers": 1 + 0.02 * aw,
                "Machine_Adjust": 1 + 0.03 * am,
                "Outsourcing": 1 + (oq / max(base_input["order_quantity"], 1)) * 0.9,
                "Hybrid": 1 + 0.02 * aw + 0.08 * ot + 0.03 * am + (oq / max(base_input["order_quantity"], 1)) * 0.9,
            }[sname]
            p = float(np.clip(1 / (1 + np.exp(-np.clip(5 * (capacity_ratio * capacity_multiplier - 1), -60, 60))), 0.02, 0.99))
            unit_value = base_input["order_value"] / max(base_input["order_quantity"], 1)
            cost = aw * 15000 + ot * 5000 + am * 2500 + oq * unit_value * 0.60
        results.append({"scenario": sname, "p_selected": round(p, 3),
                        "cost": round(cost, 2), "added_workers": aw,
                        "overtime_hours": ot, "added_machines": am,
                        "outsourced_qty": oq})
    return results

#  Home page: load the dashboard and comparison metrics.
@app.route("/", methods=["GET"])
def home():
    n1_metric = REPORT.get("N1_models", {}).get(best_n1_name, {})
    n2_metric = REPORT.get("N2_models", {}).get(best_n2_name, {})
    return render_template("index.html",
                           quantity="10000", handover_days="10",
                           smv="12.5", order_value="2500000",
                           style="Polo", priority="Normal",
                           best_n1=best_n1_name, best_n2=best_n2_name,
                           n1_metric=n1_metric, n2_metric=n2_metric,
                           cost_best=REPORT.get("cost_best", ""))

#  Prediction endpoint: evaluate the order and present the recommended action.
@app.route("/predict", methods=["POST"])
def predict():
    form = request.form
    try:
        qty = int(form.get("order_quantity", ""))
        days = int(form.get("handover_days", ""))
        smv = float(form.get("required_smv_total", ""))
        order_value = float(form.get("order_value", ""))
        style = form.get("style", "")
        priority = form.get("priority", "")
    except (TypeError, ValueError):
        return render_template("index.html", quantity=form.get("order_quantity", "10000"),
                               handover_days=form.get("handover_days", "10"),
                               smv=form.get("required_smv_total", "12.5"),
                               order_value=form.get("order_value", "2500000"),
                               style=form.get("style", "Polo"), priority=form.get("priority", "Normal"),
                               error="Enter valid numeric values for quantity, days, SMV and order value."), 400

    if qty <= 0 or days <= 0 or smv <= 0 or order_value <= 0:
        return render_template("index.html", quantity=qty, handover_days=days,
                               smv=smv, order_value=order_value, style=style, priority=priority,
                               error="Quantity, handover days, SMV and order value must be greater than zero."), 400
    if style not in {"Polo", "T-Shirt", "Hoodie", "Legging", "Jacket"} or priority not in {"Low", "Normal", "High"}:
        return render_template("index.html", quantity=qty, handover_days=days,
                               smv=smv, order_value=order_value, style="Polo", priority="Normal",
                               error="Choose a valid style and priority."), 400

    order_features = build_order_features(qty, days, smv, order_value, style, priority, FACTORY)
    n1_vector = pd.DataFrame([order_features], columns=n1_features)
    capacity_ratio = order_features["capacity_ratio"]
    if n1_model_ready:
        p_ontime = float(predict_proba(best_n1_bundle, n1_vector)[0])
    else:
        daily_capacity = (
            orders_data["n_available_workers"].to_numpy()
            * orders_data["shift_hours"].to_numpy()
            * 60
            * orders_data["avg_worker_efficiency"].to_numpy() / 100
            * orders_data["n_shifts"].to_numpy()
        )
        ratios = (
            daily_capacity * days / max(order_features["estimated_workload_min"], 1)
            * STYLE_CAPACITY_FACTOR[style] * PRIORITY_CAPACITY_FACTOR[priority]
        )
        p_ontime = float(np.clip(1 / (1 + np.exp(-np.clip(5 * (ratios - 1), -60, 60))), 0.02, 0.98).mean())

    base = {"order_quantity": qty,
            "handover_days": days,
            "order_value": order_value,
            "required_workload_min": order_features["estimated_workload_min"],
            "required_smv_total": smv,
            "style": style,
            "priority": priority,
            "n_conflicts": FACTORY["n_conflicts"],
            "avg_worker_eff": FACTORY["avg_worker_eff"]}
    scen = evaluate_scenarios(base)
    min_scenario_score = 0.5 if n2_model_ready else 0.8
    feas = [s for s in scen if s["p_selected"] >= min_scenario_score]
    best = min(feas, key=lambda s: s["cost"]) if feas else None

    if p_ontime >= 0.85 and (not best or best["scenario"] == "Current"):
        decision = "ACCEPT"
    elif best and best["scenario"] != "Current":
        decision = "ACCEPT WITH INTERVENTION"
    elif p_ontime < 0.50:
        decision = "REJECT / RE-PLAN"
    else:
        decision = "RISKY / REVIEW"

    explanation = [f"Estimated capacity covers {capacity_ratio:.2f}x the required workload.",
                   f"Intervention ranking model: {best_n2_name}.",
                   "The current training records are generated demo data; replace them with real factory history to get production estimates."]
    if n1_model_ready:
        n1_metric = REPORT["N1_models"][best_n1_name]
        explanation.insert(1, f"On-time model: {best_n1_name} (5-fold CV ROC-AUC {n1_metric['roc_auc']:.3f}; Brier score {n1_metric['brier_score']:.3f}).")
    else:
        explanation.insert(1, "The retrained six-input model is not present yet; this result uses the capacity-profile fallback.")
    if not n2_model_ready:
        explanation.append("Intervention options use the capacity estimate and demo cost rates until the new models are trained.")
    if FACTORY["n_conflicts"] > 0:
        explanation.append(f"Typical factory workload includes about {FACTORY['n_conflicts']:.0f} allocation conflict(s) per order.")
    if p_ontime < 0.7:
        explanation.append("Low on-time probability — resource or deadline risk.")
    if best and best["scenario"] != "Current":
        explanation.append(f"Recommended: {best['scenario']} "
                           f"(+{best['added_workers']} workers, {best['overtime_hours']}h OT, "
                           f"+{best['added_machines']} machines, outsource {best['outsourced_qty']} pcs).")

    return render_template("result.html",
                           p_ontime=round(p_ontime*100, 2),
                           order_quantity=qty,
                           handover_days=days,
                           estimated_smv=smv,
                           estimated_order_value=base["order_value"],
                           estimated_workload=order_features["estimated_workload_min"],
                           capacity_ratio=capacity_ratio,
                           style=style,
                           priority=priority,
                           typical_workers=FACTORY["typical_workers"],
                           estimated_efficiency=FACTORY["capacity_efficiency"],
                           scenario_score_label="Selection score" if n2_model_ready else "Estimated on-time",
                           decision=decision, scenarios=scen, best=best,
                           explanation=explanation,
                           best_n2=best_n2_name)

#  Start the Flask app for local prediction testing.
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
