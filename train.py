import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from neural_models import (
    GANNClassifier,
    GraphNeuralNetworkClassifier,
    TensorFlowDNNClassifier,
    tf,
)
from model_features import (
    SCENARIO_ENCODING,
    build_factory_profile,
    build_order_features,
    make_intervention_features,
)
from line_model_features import (
    FEASIBILITY_ACCEPT_THRESHOLD,
    FEASIBILITY_MODEL_TARGET,
    LINE_MODEL_FEATURES,
    historical_line_count,
    make_line_features,
)
from workforce_model_features import (
    WORKER_OPERATION_FEATURES,
    make_worker_operation_features,
)
from capacity_model_features import build_capacity_training_data
from factory_config import FACTORY_SHIFTS_PER_DAY

SEED = 42
os.makedirs("models", exist_ok=True)
warnings.filterwarnings("ignore", category=UserWarning)

orders = pd.read_csv("data/orders.csv")
allocations = pd.read_csv("data/allocations.csv")
scenarios = pd.read_csv("data/interventions.csv")
workers = pd.read_csv("data/workers.csv")
operations = pd.read_csv("data/operations.csv")
profile = build_factory_profile(orders, allocations)

N1_FEATURES = [
    "order_quantity", "handover_days", "required_smv_total", "order_value",
    "style_enc", "priority_enc", "unit_value", "estimated_workload_min",
    "estimated_workload_per_day", "capacity_ratio",
]


def classification_metrics(y_true, probabilities, threshold=0.5):
    predictions = (probabilities >= threshold).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y_true, predictions)), 4),
        "precision": round(float(precision_score(y_true, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, predictions, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 4),
        "brier_score": round(float(brier_score_loss(y_true, probabilities)), 4),
        "confusion_matrix": confusion_matrix(
            y_true, predictions, labels=[0, 1]
        ).tolist(),
    }


def compare_classifiers(name, candidates, X, y, cv, groups=None):
    scores = {}
    for model_name, model in candidates.items():
        print(f"  Training {model_name}...", flush=True)
        kwargs = {"groups": groups} if groups is not None else {}
        probabilities = cross_val_predict(
            model, X, y, cv=cv, method="predict_proba", n_jobs=1, **kwargs
        )[:, 1]
        scores[model_name] = classification_metrics(y, probabilities)
    # Brier score rewards useful, well-calibrated probabilities; AUC breaks ties.
    winner = min(scores, key=lambda key: (scores[key]["brier_score"], -scores[key]["roc_auc"]))
    final_model = candidates[winner].fit(X, y)
    scores = dict(sorted(
        scores.items(), key=lambda item: (item[1]["brier_score"], -item[1]["roc_auc"])
    ))
    print(f"\n{name} comparison (sorted by Brier score, then ROC-AUC):")
    for model_name, metrics in scores.items():
        marker = "  <-- BEST" if model_name == winner else ""
        print(
            f"  {model_name:<30} Brier={metrics['brier_score']:.4f}  "
            f"ROC-AUC={metrics['roc_auc']:.4f}  F1={metrics['f1']:.4f}{marker}"
        )
    return winner, final_model, scores


def save_classifier_bundle(name, model, features, profile, output_path,
                           keras_path, extra=None):
    """Save sklearn models in joblib and Keras networks in native .keras format."""
    bundle = {
        "name": name,
        "features": features,
        "profile": profile,
        "training_signature": "six-order-inputs-v1",
    }
    if extra:
        bundle.update(extra)
    if isinstance(model, TensorFlowDNNClassifier):
        model.keras_model_.save(keras_path)
        bundle.update({
            "model": None,
            "type": "keras",
            "model_path": keras_path,
            "scaler": model.scaler_,
        })
    else:
        bundle.update({"model": model, "type": "sklearn"})
    joblib.dump(bundle, output_path)


# N1: use the six order details available in the form, plus workload and
# capacity features derived from those inputs and the factory profile.
X1 = pd.DataFrame(
    [
        build_order_features(
            row.order_quantity, row.handover_days, row.required_smv_total,
            row.order_value, row.style_id, row.priority, profile,
        )
        for row in orders.itertuples()
    ],
    columns=N1_FEATURES,
)
y1 = orders["on_time_flag"].astype(int).to_numpy()

n1_candidates = {
    "Logistic Regression": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED)
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=160, max_depth=10, min_samples_leaf=8,
        n_jobs=-1, random_state=SEED,
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=160, max_depth=12, min_samples_leaf=6,
        n_jobs=-1, random_state=SEED,
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=8, min_samples_leaf=12, class_weight="balanced", random_state=SEED,
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100, max_depth=2, learning_rate=0.06, random_state=SEED,
    ),
    "Histogram Gradient Boosting": HistGradientBoostingClassifier(
        max_iter=140, max_leaf_nodes=15, l2_regularization=2.0,
        learning_rate=0.07, random_state=SEED,
    ),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, learning_rate=0.05, random_state=SEED),
    "K-Nearest Neighbors": make_pipeline(
        StandardScaler(), KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1)
    ),
    "Genetic Algorithm Neural Network (GANN)": make_pipeline(
        StandardScaler(), GANNClassifier(population_size=4, generations=2,
                                         max_iter=100, random_state=SEED)
    ),
    "Graph Neural Network (GNN)": GraphNeuralNetworkClassifier(random_state=SEED),
    "Gaussian Naive Bayes": make_pipeline(StandardScaler(), GaussianNB()),
}
if tf is not None:
    n1_candidates["DNN (TensorFlow/Keras)"] = TensorFlowDNNClassifier(random_state=SEED)
n1_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
n1_name, n1_model, n1_metrics = compare_classifiers(
    "On-time model", n1_candidates, X1, y1, n1_cv
)
save_classifier_bundle(
    n1_name, n1_model, N1_FEATURES, profile,
    "models/best_n1_model.pkl", "models/best_n1_model.keras",
    extra={"user_inputs": ["order_quantity", "handover_days", "required_smv_total", "order_value", "style_id", "priority"]},
)

# N2: preserve scenario labels and costs, but build the order-level fields with
# the same inferred profile used at prediction time. Grouped folds keep all six
# scenarios for an order on the same side of each validation split.
order_features = {
    int(float(row.order_id)): build_order_features(
        row.order_quantity, row.handover_days, row.required_smv_total,
        row.order_value, row.style_id, row.priority, profile,
    )
    for row in orders.itertuples()
}
two_shift_order_ids = set(
    orders.loc[
        pd.to_numeric(orders["n_shifts"], errors="coerce").le(FACTORY_SHIFTS_PER_DAY),
        "order_id",
    ].astype(int)
)
scenario_training = scenarios[
    scenarios["order_id"].astype(int).isin(two_shift_order_ids)
].copy()
n2_feature_rows = []
for scenario in scenario_training.to_dict(orient="records"):
    inferred = order_features[int(scenario["order_id"])]
    n2_feature_rows.append(make_intervention_features(inferred, scenario, profile))

N2_FEATURES = [
    "scenario_enc", "added_workers", "overtime_hours", "added_machines",
    "outsourced_qty", "order_quantity", "handover_days", "order_value",
    "required_workload_min", "style_enc", "priority_enc", "capacity_ratio",
    "n_conflicts", "avg_worker_eff",
]
X2 = pd.DataFrame(n2_feature_rows, columns=N2_FEATURES)
y2 = scenario_training["selected_flag"].astype(int).to_numpy()
groups = scenario_training["order_id"].to_numpy()
n2_cv = GroupKFold(n_splits=5)
n2_candidates = {
    "Logistic Regression": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED)
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=120, max_depth=14, min_samples_leaf=5,
        n_jobs=-1, random_state=SEED,
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=140, max_depth=16, min_samples_leaf=4,
        n_jobs=-1, random_state=SEED,
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=10, min_samples_leaf=10, class_weight="balanced", random_state=SEED,
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100, max_depth=2, learning_rate=0.06, random_state=SEED,
    ),
    "Histogram Gradient Boosting": HistGradientBoostingClassifier(
        max_iter=140, max_leaf_nodes=19, l2_regularization=2.0,
        learning_rate=0.07, random_state=SEED,
    ),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, learning_rate=0.05, random_state=SEED),
    "K-Nearest Neighbors": make_pipeline(
        StandardScaler(), KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1)
    ),
    "Genetic Algorithm Neural Network (GANN)": make_pipeline(
        StandardScaler(), GANNClassifier(population_size=4, generations=2,
                                         max_iter=100, random_state=SEED)
    ),
    "Graph Neural Network (GNN)": GraphNeuralNetworkClassifier(random_state=SEED),
    "Gaussian Naive Bayes": make_pipeline(StandardScaler(), GaussianNB()),
}
if tf is not None:
    n2_candidates["DNN (TensorFlow/Keras)"] = TensorFlowDNNClassifier(random_state=SEED)
n2_name, n2_model, n2_metrics = compare_classifiers(
    "Intervention model", n2_candidates, X2, y2, n2_cv, groups
)
save_classifier_bundle(
    n2_name, n2_model, N2_FEATURES, profile,
    "models/best_n2_model.pkl", "models/best_n2_model.keras",
    extra={"scenario_encoding": SCENARIO_ENCODING},
)

# The cost regressor is trained and evaluated with the same order-grouped folds.
cost_candidates = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(
        max_depth=12, min_samples_leaf=8, random_state=SEED,
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=120, max_depth=16, min_samples_leaf=4,
        n_jobs=-1, random_state=SEED,
    ),
    "Extra Trees": ExtraTreesRegressor(
        n_estimators=120, max_depth=16, min_samples_leaf=4,
        n_jobs=-1, random_state=SEED,
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100, max_depth=2, learning_rate=0.06, loss="huber", random_state=SEED,
    ),
    "Histogram Gradient Boosting": HistGradientBoostingRegressor(
        max_iter=160, max_leaf_nodes=19, l2_regularization=2.0,
        learning_rate=0.07, random_state=SEED,
    ),
    "AdaBoost": AdaBoostRegressor(n_estimators=100, learning_rate=0.05, random_state=SEED),
    "K-Nearest Neighbors": make_pipeline(
        StandardScaler(), KNeighborsRegressor(n_neighbors=15, weights="distance", n_jobs=-1)
    ),
}
cost_scores = {}
cost_models = {}
for model_name, model in cost_candidates.items():
    print(f"  Training cost model {model_name}...", flush=True)
    oof_cost = cross_val_predict(model, X2, scenario_training["total_cost"].to_numpy(),
                                 cv=n2_cv, groups=groups, n_jobs=1)
    cost_scores[model_name] = {
        "mae": round(float(mean_absolute_error(scenario_training["total_cost"], oof_cost)), 2),
        "r2": round(float(r2_score(scenario_training["total_cost"], oof_cost)), 4),
    }
    cost_models[model_name] = model
cost_name = min(cost_scores, key=lambda key: cost_scores[key]["mae"])
cost_scores = dict(sorted(cost_scores.items(), key=lambda item: item[1]["mae"]))
print("\nCost model comparison (sorted by MAE):")
for model_name, metrics in cost_scores.items():
    marker = "  <-- BEST" if model_name == cost_name else ""
    print(f"  {model_name:<30} MAE={metrics['mae']:.2f}  R²={metrics['r2']:.4f}{marker}")
cost_model = cost_models[cost_name].fit(X2, scenario_training["total_cost"].to_numpy())
joblib.dump(cost_model, "models/best_cost_model.pkl")

# N3: line-aware accept/reject feasibility classifier. Historical worker
# counts are mapped to 17-operator line equivalents using the factory standard.
line_orders = orders[
    pd.to_numeric(orders["n_shifts"], errors="coerce").le(FACTORY_SHIFTS_PER_DAY)
    & orders["order_status"].isin(["Accept", "Reject"])
]
line_feature_rows = []
for row in line_orders.itertuples():
    line_count = historical_line_count(row.n_available_workers)
    line_feature_rows.append(make_line_features(
        row.order_quantity, row.handover_days, row.required_smv_total,
        row.order_value, row.style_id, row.priority, line_count, int(float(row.n_shifts)),
        float(row.shift_hours),
    ))
X3 = pd.DataFrame(line_feature_rows, columns=LINE_MODEL_FEATURES)
y3 = line_orders["order_status"].eq("Accept").astype(int).to_numpy()
line_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(C=0.8, max_iter=2000, random_state=SEED),
)
line_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
line_oof = cross_val_predict(
    line_model, X3, y3, cv=line_cv, method="predict_proba", n_jobs=1
)[:, 1]
line_metrics = classification_metrics(y3, line_oof, FEASIBILITY_ACCEPT_THRESHOLD)
line_model.fit(X3, y3)
joblib.dump({
    "name": "Feasibility Logistic Regression",
    "target": FEASIBILITY_MODEL_TARGET,
    "accept_threshold": FEASIBILITY_ACCEPT_THRESHOLD,
    "model": line_model,
    "features": LINE_MODEL_FEATURES,
    "training_ranges": {
        feature: [float(X3[feature].min()), float(X3[feature].max())]
        for feature in LINE_MODEL_FEATURES
    },
}, "models/line_feasibility_model.pkl")
print(
    "\nLine-aware model validation: "
    f"Brier={line_metrics['brier_score']:.4f}  "
    f"ROC-AUC={line_metrics['roc_auc']:.4f}"
)

# N4: predict operator efficiency and quality for each required operation.
performance_rows = allocations.merge(workers, on="worker_id").merge(operations, on="operation_id")
performance_rows = performance_rows[
    performance_rows["allocation_status"].eq("Completed")
].dropna(subset=["actual_efficiency", "actual_quality"])
X4 = pd.DataFrame(
    [make_worker_operation_features(row, row) for row in performance_rows.to_dict(orient="records")],
    columns=WORKER_OPERATION_FEATURES,
)
y4 = performance_rows[["actual_efficiency", "actual_quality"]].astype(float).to_numpy()
workforce_candidates = {
    "Extra Trees": ExtraTreesRegressor(
        n_estimators=100, max_depth=16, min_samples_leaf=8,
        n_jobs=-1, random_state=SEED,
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=100, max_depth=16, min_samples_leaf=8,
        n_jobs=-1, random_state=SEED,
    ),
}
performance_cv = GroupKFold(n_splits=5)
workforce_scores = {}
workforce_groups = performance_rows["order_id"].to_numpy()
for model_name, candidate in workforce_candidates.items():
    workforce_oof = cross_val_predict(
        candidate, X4, y4, cv=performance_cv,
        groups=workforce_groups, n_jobs=1,
    )
    workforce_scores[model_name] = {
        "efficiency_mae": round(float(mean_absolute_error(y4[:, 0], workforce_oof[:, 0])), 3),
        "efficiency_r2": round(float(r2_score(y4[:, 0], workforce_oof[:, 0])), 4),
        "quality_mae": round(float(mean_absolute_error(y4[:, 1], workforce_oof[:, 1])), 3),
        "quality_r2": round(float(r2_score(y4[:, 1], workforce_oof[:, 1])), 4),
    }
workforce_best = min(
    workforce_scores,
    key=lambda name: (workforce_scores[name]["efficiency_mae"] + workforce_scores[name]["quality_mae"]) / 2,
)
performance_metrics = workforce_scores[workforce_best]
performance_model = workforce_candidates[workforce_best]
performance_model.fit(X4, y4)
joblib.dump({
    "name": f"Worker-operation {workforce_best} Regressor",
    "model": performance_model,
    "features": WORKER_OPERATION_FEATURES,
}, "models/workforce_performance_model.pkl")
print(
    f"\nWorker-operation model validation ({workforce_best}): "
    f"Efficiency MAE={performance_metrics['efficiency_mae']:.3f}, "
    f"Quality MAE={performance_metrics['quality_mae']:.3f}"
)

# N5: predict aggregate completed garment minutes per standardized line-shift.
# The target is built only from order rows with recorded completion quantity,
# start/completion dates, historical headcount, shift count and shift length.
X5, y5, capacity_groups, capacity_styles = build_capacity_training_data(orders)
if len(X5) < 20 or len(np.unique(capacity_groups)) < 5:
    raise ValueError(
        "Capacity model requires at least 20 valid completed-order records "
        "across at least 5 distinct orders. Check data/orders.csv."
    )
capacity_cv = GroupKFold(n_splits=min(5, len(np.unique(capacity_groups))))
capacity_candidates = {
    "Extra Trees": ExtraTreesRegressor(
        n_estimators=180, max_depth=18, min_samples_leaf=5,
        n_jobs=-1, random_state=SEED,
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=180, max_depth=16, min_samples_leaf=5,
        n_jobs=-1, random_state=SEED,
    ),
    "Gradient Boosting (Huber)": GradientBoostingRegressor(
        n_estimators=140, max_depth=2, min_samples_leaf=8,
        learning_rate=0.05, loss="huber", random_state=SEED,
    ),
}
capacity_scores = {}
capacity_oof_predictions = {}
for model_name, candidate in capacity_candidates.items():
    print(f"  Training capacity model {model_name}...", flush=True)
    oof_prediction = cross_val_predict(
        candidate, X5, y5, cv=capacity_cv, groups=capacity_groups, n_jobs=1,
    )
    capacity_oof_predictions[model_name] = oof_prediction
    capacity_scores[model_name] = {
        "mae_standard_minutes_per_line_shift": round(
            float(mean_absolute_error(y5, oof_prediction)), 3
        ),
        "r2": round(float(r2_score(y5, oof_prediction)), 4),
    }
capacity_best = min(
    capacity_scores,
    key=lambda name: capacity_scores[name]["mae_standard_minutes_per_line_shift"],
)
capacity_model = capacity_candidates[capacity_best].fit(X5, y5)
capacity_residuals = y5 - capacity_oof_predictions[capacity_best]
residual_quantiles = np.quantile(capacity_residuals, [0.10, 0.50, 0.90])
style_target_bounds = {}
for style_name in sorted(set(capacity_styles)):
    style_values = y5[capacity_styles == style_name]
    style_target_bounds[style_name] = {
        "p05": float(np.quantile(style_values, 0.05)),
        "p95": float(np.quantile(style_values, 0.95)),
        "rows": int(len(style_values)),
    }
capacity_bundle = {
    "name": capacity_best,
    "model": capacity_model,
    "features": list(X5.columns),
    "feature_ranges": {
        feature: [float(X5[feature].min()), float(X5[feature].max())]
        for feature in X5.columns
    },
    "residual_quantiles": {
        "p10": float(residual_quantiles[0]),
        "p50": float(residual_quantiles[1]),
        "p90": float(residual_quantiles[2]),
    },
    "style_target_bounds": style_target_bounds,
    "training_rows": int(len(X5)),
    "training_orders": int(len(np.unique(capacity_groups))),
    "target": "recorded completed quantity * SMV / elapsed days / historical line equivalents / shifts, normalized to 8 hours",
    "validation": "5-fold order-grouped out-of-fold MAE and R2",
}
joblib.dump(capacity_bundle, "models/line_shift_capacity_model.pkl")
capacity_metrics = capacity_scores[capacity_best]
capacity_scores = dict(sorted(
    capacity_scores.items(),
    key=lambda item: item[1]["mae_standard_minutes_per_line_shift"],
))
print(
    f"\nLine-shift capacity model validation ({capacity_best}): "
    f"MAE={capacity_metrics['mae_standard_minutes_per_line_shift']:.3f} "
    f"standard minutes/line-shift, R²={capacity_metrics['r2']:.4f}; "
    f"trained on {len(X5)} completed orders."
)

report = {
    "data_note": "Checked-in records are generated sample data; replace and validate against actual factory history before operational use.",
    "training_orders": int(len(orders)),
    "N1_user_inputs": ["order_quantity", "handover_days", "required_smv_total", "order_value", "style_id", "priority"],
    "training_signature": "six-order-inputs-v1",
    "N1_features": N1_FEATURES,
    "N1_best": n1_name,
    "N1_selection_metric": "Lowest 5-fold cross-validated Brier score; ROC-AUC breaks ties",
    "N1_models": n1_metrics,
    "N2_best": n2_name,
    "N2_selection_metric": "Lowest order-grouped 5-fold cross-validated Brier score; ROC-AUC breaks ties",
    "N2_models": n2_metrics,
    "N2_training_orders": int(scenario_training["order_id"].nunique()),
    "cost_best": cost_name,
    "cost_models": cost_scores,
    "line_model": "Line-aware Logistic Regression",
    "factory_shifts_per_day": FACTORY_SHIFTS_PER_DAY,
    "line_model_metrics": line_metrics,
    "workforce_model": f"Worker-operation {workforce_best} Regressor",
    "workforce_model_metrics": performance_metrics,
    "workforce_models": workforce_scores,
    "workforce_training_allocations": int(len(performance_rows)),
    "workforce_training_orders": int(performance_rows["order_id"].nunique()),
    "capacity_model": capacity_best,
    "capacity_model_metrics": capacity_metrics,
    "capacity_model_candidates": capacity_scores,
    "capacity_model_training_rows": int(len(X5)),
    "capacity_model_training_orders": int(len(np.unique(capacity_groups))),
    "capacity_model_features": list(X5.columns),
    "capacity_model_target": capacity_bundle["target"],
    "capacity_model_interval": "P10/P50/P90 from order-grouped out-of-fold residuals; planning range, not guarantee",
}
with open("models/model_comparison.json", "w", encoding="utf-8") as report_file:
    json.dump(report, report_file, indent=2)
with open("models/metrics.json", "w", encoding="utf-8") as metrics_file:
    json.dump({"N1": n1_metrics, "N2": n2_metrics, "cost": cost_scores,
               "line_feasibility": line_metrics,
               "workforce_performance": workforce_scores,
               "line_shift_capacity": capacity_scores}, metrics_file, indent=2)

print("Saved trained models and cross-validation metrics under models/.")
