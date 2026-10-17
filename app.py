"""Flask app for historical order-capacity planning."""
import json
import os

from flask import Flask, render_template, request
import joblib
import pandas as pd
from historical_capacity import build_historical_capacity_plan
from line_model_features import (
    LINE_MODEL_FEATURES,
    historical_line_count,
    make_line_features,
)
from research_analysis import (
    analyze_intervention_options,
    predict_line_feasibility,
)
from factory_config import FACTORY_SHIFTS_PER_DAY

app = Flask(__name__)

# Completed orders are the source for the historical throughput benchmark.
orders_data = pd.read_csv("data/orders.csv")
interventions_data = pd.read_csv("data/interventions.csv")
shift_hours = float(pd.to_numeric(orders_data["shift_hours"], errors="coerce").median())


def _load_bundle(path):
    if not os.path.exists(path):
        return None
    try:
        bundle = joblib.load(path)
        if bundle.get("type") == "keras":
            import tensorflow as tf
            bundle["model"] = tf.keras.models.load_model(
                bundle["model_path"], compile=False,
            )
        return bundle
    except Exception as exc:
        print(f"Could not load {path}: {exc}")
        return None


capacity_model = _load_bundle("models/line_shift_capacity_model.pkl")
line_model = _load_bundle("models/line_feasibility_model.pkl")
if line_model:
    training_orders = orders_data[
        pd.to_numeric(orders_data["n_shifts"], errors="coerce").le(FACTORY_SHIFTS_PER_DAY)
    ]
    if not training_orders.empty:
        line_features = pd.DataFrame([
            make_line_features(
                row.order_quantity, row.handover_days, row.required_smv_total,
                row.order_value, row.style_id, row.priority,
                historical_line_count(row.n_available_workers), row.n_shifts,
                row.shift_hours,
            )
            for row in training_orders.itertuples()
        ], columns=LINE_MODEL_FEATURES)
        fallback_ranges = {
            feature: [float(line_features[feature].min()), float(line_features[feature].max())]
            for feature in LINE_MODEL_FEATURES
        }
        for feature, limits in fallback_ranges.items():
            line_model.setdefault("training_ranges", {}).setdefault(feature, limits)
n2_bundle = _load_bundle("models/best_n2_model.pkl")
cost_model = None
if os.path.exists("models/best_cost_model.pkl"):
    try:
        cost_model = joblib.load("models/best_cost_model.pkl")
    except Exception as exc:
        print(f"Could not load models/best_cost_model.pkl: {exc}")

training_report = {}
try:
    with open("models/model_comparison.json", encoding="utf-8") as report_file:
        training_report = json.load(report_file)
except (OSError, json.JSONDecodeError):
    pass


#  Home page: load the dashboard and comparison metrics.
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html",
                           quantity="10000", handover_days="10",
                           smv="12.5", order_value="2500000",
                           available_lines="4",
                           style="Polo", priority="Normal")

#  Prediction endpoint: evaluate the order and present the recommended action.
@app.route("/predict", methods=["POST"])
def predict():
    form = request.form
    try:
        qty = int(form.get("order_quantity", ""))
        days = int(form.get("handover_days", ""))
        available_lines = int(form.get("available_lines", ""))
        # This factory runs two shifts; shift count is fixed and not user-entered.
        available_shifts = FACTORY_SHIFTS_PER_DAY
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
                               available_lines=form.get("available_lines", "4"),
                               available_shifts=FACTORY_SHIFTS_PER_DAY,
                               error="Enter valid numeric values for quantity, days, lines, SMV and order value."), 400

    if qty <= 0 or days <= 0 or smv <= 0 or order_value <= 0:
        return render_template("index.html", quantity=qty, handover_days=days,
                               smv=smv, order_value=order_value, style=style, priority=priority,
                               available_lines=available_lines,
                               available_shifts=available_shifts,
                               error="Quantity, handover days, SMV and order value must be greater than zero."), 400
    if available_lines < 1 or available_lines > 30:
        return render_template("index.html", quantity=qty, handover_days=days,
                               smv=smv, order_value=order_value, style=style, priority=priority,
                               available_lines=available_lines,
                               available_shifts=available_shifts,
                               error="Available sewing lines must be from 1 to 30."), 400
    if style not in {"Polo", "T-Shirt", "Hoodie", "Legging", "Jacket"} or priority not in {"Low", "Normal", "High"}:
        return render_template("index.html", quantity=qty, handover_days=days,
                               smv=smv, order_value=order_value, style="Polo", priority="Normal",
                                available_lines=available_lines,
                                available_shifts=available_shifts,
                                error="Choose a valid style and priority."), 400

    production_plan = build_historical_capacity_plan(
        orders_data, qty, days, smv, style, available_lines,
        available_shifts, shift_hours, order_value=order_value,
        priority=priority, capacity_model=capacity_model,
    )
    if production_plan.get("error"):
        return render_template("index.html", quantity=qty, handover_days=days,
                               smv=smv, order_value=order_value, style=style, priority=priority,
                               available_lines=available_lines, available_shifts=available_shifts,
                               error=production_plan["error"]), 400
    n1_analysis = predict_line_feasibility(
        line_model, training_report.get("line_model_metrics"),
        qty, days, smv, order_value, style, priority,
        available_lines, available_shifts, shift_hours,
    )
    cost_metrics = training_report.get("cost_models", {}).get(
        training_report.get("cost_best"), {},
    )
    n2_analysis = analyze_intervention_options(
        n2_bundle, cost_model,
        training_report.get("N2_models"), cost_metrics,
        interventions_data, orders_data, qty, days, smv,
        order_value, style, priority,
    )
    recommended = production_plan["recommended"]
    capacity_ratio = recommended["capacity_ratio"]
    decision = production_plan["decision"]
    explanation = [
        f"{production_plan['capacity_source']}: estimated {production_plan['rate_median']:.1f} pieces per line-shift per day at this order's SMV.",
        f"The median plan needs {production_plan['plans'][0]['line_shifts_required']} line-shifts; the conservative plan needs {production_plan['plans'][1]['line_shifts_required']}. Maximum capacity is {production_plan['max_line_shifts']} line-shifts across {available_lines} lines and {available_shifts} shifts.",
        f"Estimated completion: {recommended['estimated_days']} days at the expected rate; conservative estimate: {recommended['conservative_days']} days.",
        production_plan["planning_note"],
        production_plan["machine_note"],
    ]

    return render_template("result.html",
                           order_quantity=qty,
                           handover_days=days,
                           estimated_smv=smv,
                           estimated_order_value=order_value,
                           estimated_workload=qty * smv,
                           capacity_ratio=capacity_ratio,
                           style=style,
                           priority=priority,
                           production_plan=production_plan,
                           n1_analysis=n1_analysis,
                           n2_analysis=n2_analysis,
                           decision=decision, explanation=explanation)

#  Start the Flask app for local prediction testing.
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
