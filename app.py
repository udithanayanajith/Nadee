"""
 Flask app: loads the auto-selected best models from the comparison report.
"""
from flask import Flask, render_template, request
import numpy as np, pandas as pd, joblib, json
from tensorflow.keras.models import load_model

app = Flask(__name__)

#  Load the comparison report to know which model won each task.
with open("models/model_comparison.json") as f:
    REPORT = json.load(f)

scaler_n1   = joblib.load("models/scaler_n1.pkl")
scaler_n2   = joblib.load("models/scaler_n2.pkl")
n1_features = joblib.load("models/n1_features.pkl")
n2_features = joblib.load("models/n2_features.pkl")

def load_any(path):
    """Load either a .h5 Keras model or a .pkl sklearn model."""
    if path.endswith(".h5"):
        return load_model(path, compile=False)
    return joblib.load(path)

best_n1_name = REPORT["N1_best"]
best_n2_name = REPORT["N2_best"]

#  Try to load the best models. If a .pkl file is available, use it;
# otherwise, fall back to the saved .h5 model for Keras-based solutions.
def load_best(name, task):
    pkl = f"models/{task}_best_{name.replace(' ','_')}.pkl"
    if joblib.os.path.exists(pkl):
        return joblib.load(pkl)
    # fallback: load generic saved files
    if name == "DNN" or name == "GANN":
        return load_model(f"models/model_{task}.h5", compile=False)
    return joblib.load(f"models/model_{task}_rf.pkl")

#  Retrain-time save ensures each bundle contains the selected model and metadata.
best_n1_bundle = joblib.load("models/best_n1_model.pkl")
best_n2_bundle = joblib.load("models/best_n2_model.pkl")
best_cost      = joblib.load("models/best_cost_model.pkl")

def predict_proba(bundle, X):
    """Works for both keras & sklearn bundles."""
    model = bundle["model"]
    if bundle["type"] == "keras":
        return model.predict(X, verbose=0).ravel()
    return model.predict_proba(X)[:, 1]

#  Scenario mapping used to compare intervention options.
SCENARIO_MAP = {"Current": 0, "Overtime": 1, "Extra_Workers": 2,
                "Machine_Adjust": 3, "Outsourcing": 4, "Hybrid": 5}

def build_n1_vector(form):
    qty = float(form["order_quantity"]); days = float(form["handover_days"])
    smv = float(form["required_smv_total"]); workload = qty * smv
    row = {
        "order_quantity": qty, "handover_days": days,
        "order_value": float(form["order_value"]),
        "required_smv_total": smv, "required_workload_min": workload,
        "workload_per_day": workload / days,
        "priority_enc": {"Low": 0, "Normal": 1, "High": 2}.get(form["priority"], 1),
        "style_enc": {"Polo": 0, "T-Shirt": 1, "Hoodie": 2, "Legging": 3, "Jacket": 4}.get(form["style"], 0),
        "total_assigned_workload": float(form.get("total_assigned_workload", 0)),
        "n_allocations": float(form.get("n_allocations", 0)),
        "n_conflicts": float(form.get("n_conflicts", 0)),
        "avg_worker_eff": float(form.get("avg_worker_eff", 70)),
        "avg_worker_qual": float(form.get("avg_worker_qual", 92)),
        "distinct_workers": float(form.get("distinct_workers", 10)),
        "distinct_machines": float(form.get("distinct_machines", 8)),
        "active_allocs": float(form.get("active_allocs", 3)),
        "availability_rate": float(form.get("availability_rate", 0.7)),
    }
    return np.array([[row[f] for f in n1_features]])

def evaluate_scenarios(base_input):
    results = []
    for sname, senc in SCENARIO_MAP.items():
        aw = {"Current":0,"Overtime":0,"Extra_Workers":5,"Machine_Adjust":0,"Outsourcing":0,"Hybrid":3}[sname]
        ot = {"Current":0,"Overtime":3,"Extra_Workers":0,"Machine_Adjust":0,"Outsourcing":0,"Hybrid":2}[sname]
        am = {"Current":0,"Overtime":0,"Extra_Workers":0,"Machine_Adjust":3,"Outsourcing":0,"Hybrid":2}[sname]
        oq = {"Current":0,"Overtime":0,"Extra_Workers":0,"Machine_Adjust":0,
              "Outsourcing":int(base_input["order_quantity"]*0.3),
              "Hybrid":int(base_input["order_quantity"]*0.2)}[sname]
        row = {"scenario_enc": senc, "added_workers": aw, "overtime_hours": ot,
               "added_machines": am, "outsourced_qty": oq,
               "order_quantity": base_input["order_quantity"],
               "handover_days": base_input["handover_days"],
               "order_value": base_input["order_value"],
               "required_workload_min": base_input["required_workload_min"],
               "n_conflicts": base_input["n_conflicts"],
               "avg_worker_eff": base_input["avg_worker_eff"]}
        vec = scaler_n2.transform(np.array([[row[f] for f in n2_features]]))
        p   = float(predict_proba(best_n2_bundle, vec)[0])
        cost = float(best_cost.predict(vec)[0])
        results.append({"scenario": sname, "p_selected": round(p, 3),
                        "cost": round(cost, 2), "added_workers": aw,
                        "overtime_hours": ot, "added_machines": am,
                        "outsourced_qty": oq})
    return results

#  Home page: load the dashboard and comparison metrics.
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html",
                           best_n1=best_n1_name, best_n2=best_n2_name,
                           n1_metrics=REPORT["N1_models"],
                           n2_metrics=REPORT["N2_models"])

#  Prediction endpoint: evaluate the order and present the recommended action.
@app.route("/predict", methods=["POST"])
def predict():
    form = request.form
    x1 = scaler_n1.transform(build_n1_vector(form))
    p_ontime = float(predict_proba(best_n1_bundle, x1)[0])

    base = {"order_quantity": float(form["order_quantity"]),
            "handover_days": float(form["handover_days"]),
            "order_value": float(form["order_value"]),
            "required_workload_min": float(form["order_quantity"]) * float(form["required_smv_total"]),
            "n_conflicts": float(form.get("n_conflicts", 0)),
            "avg_worker_eff": float(form.get("avg_worker_eff", 70))}
    scen = evaluate_scenarios(base)
    feas = [s for s in scen if s["p_selected"] >= 0.5]
    best = min(feas, key=lambda s: s["cost"]) if feas else None

    if p_ontime >= 0.85 and best and best["scenario"] == "Current":
        decision = "ACCEPT"
    elif p_ontime < 0.50 and not feas:
        decision = "REJECT / RE-PLAN"
    elif p_ontime < 0.85 and best:
        decision = "ACCEPT WITH INTERVENTION"
    else:
        decision = "RISKY / REVIEW"

    explanation = [f"Best N1 model used: {best_n1_name}",
                   f"Best N2 model used: {best_n2_name}"]
    if float(form.get("n_conflicts", 0)) > 0:
        explanation.append(f"{int(float(form.get('n_conflicts',0)))} allocation conflict(s) detected.")
    if p_ontime < 0.7:
        explanation.append("Low on-time probability — resource or deadline risk.")
    if best and best["scenario"] != "Current":
        explanation.append(f"Recommended: {best['scenario']} "
                           f"(+{best['added_workers']} workers, {best['overtime_hours']}h OT, "
                           f"+{best['added_machines']} machines, outsource {best['outsourced_qty']} pcs).")

    return render_template("result.html",
                           p_ontime=round(p_ontime*100, 2),
                           decision=decision, scenarios=scen, best=best,
                           explanation=explanation,
                           best_n1=best_n1_name, best_n2=best_n2_name,
                           n1_metrics=REPORT["N1_models"],
                           n2_metrics=REPORT["N2_models"])

#  Start the Flask app for local prediction testing.
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)