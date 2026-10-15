import json
import os
import warnings

import joblib
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
)
from model_features import (
    SCENARIO_ENCODING,
    build_factory_profile,
    build_order_features,
    make_intervention_features,
)

SEED = 42
os.makedirs("models", exist_ok=True)
warnings.filterwarnings("ignore", category=UserWarning)

orders = pd.read_csv("data/orders.csv")
allocations = pd.read_csv("data/allocations.csv")
scenarios = pd.read_csv("data/interventions.csv")
profile = build_factory_profile(orders, allocations)

N1_FEATURES = [
    "order_quantity", "handover_days", "required_smv_total", "order_value",
    "style_enc", "priority_enc", "unit_value", "estimated_workload_min",
    "estimated_workload_per_day", "capacity_ratio",
]


def classification_metrics(y_true, probabilities):
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y_true, predictions)), 4),
        "precision": round(float(precision_score(y_true, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, predictions, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 4),
        "brier_score": round(float(brier_score_loss(y_true, probabilities)), 4),
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
    "DNN (TensorFlow/Keras)": TensorFlowDNNClassifier(random_state=SEED),
    "Genetic Algorithm Neural Network (GANN)": make_pipeline(
        StandardScaler(), GANNClassifier(population_size=4, generations=2,
                                         max_iter=100, random_state=SEED)
    ),
    "Graph Neural Network (GNN)": GraphNeuralNetworkClassifier(random_state=SEED),
    "Gaussian Naive Bayes": make_pipeline(StandardScaler(), GaussianNB()),
}
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
    int(row.order_id): build_order_features(
        row.order_quantity, row.handover_days, row.required_smv_total,
        row.order_value, row.style_id, row.priority, profile,
    )
    for row in orders.itertuples()
}
n2_feature_rows = []
for scenario in scenarios.to_dict(orient="records"):
    inferred = order_features[int(scenario["order_id"])]
    n2_feature_rows.append(make_intervention_features(inferred, scenario, profile))

N2_FEATURES = [
    "scenario_enc", "added_workers", "overtime_hours", "added_machines",
    "outsourced_qty", "order_quantity", "handover_days", "order_value",
    "required_workload_min", "style_enc", "priority_enc", "capacity_ratio",
    "n_conflicts", "avg_worker_eff",
]
X2 = pd.DataFrame(n2_feature_rows, columns=N2_FEATURES)
y2 = scenarios["selected_flag"].astype(int).to_numpy()
groups = scenarios["order_id"].to_numpy()
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
    "DNN (TensorFlow/Keras)": TensorFlowDNNClassifier(random_state=SEED),
    "Genetic Algorithm Neural Network (GANN)": make_pipeline(
        StandardScaler(), GANNClassifier(population_size=4, generations=2,
                                         max_iter=100, random_state=SEED)
    ),
    "Graph Neural Network (GNN)": GraphNeuralNetworkClassifier(random_state=SEED),
    "Gaussian Naive Bayes": make_pipeline(StandardScaler(), GaussianNB()),
}
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
    oof_cost = cross_val_predict(model, X2, scenarios["total_cost"].to_numpy(),
                                 cv=n2_cv, groups=groups, n_jobs=1)
    cost_scores[model_name] = {
        "mae": round(float(mean_absolute_error(scenarios["total_cost"], oof_cost)), 2),
        "r2": round(float(r2_score(scenarios["total_cost"], oof_cost)), 4),
    }
    cost_models[model_name] = model
cost_name = min(cost_scores, key=lambda key: cost_scores[key]["mae"])
cost_scores = dict(sorted(cost_scores.items(), key=lambda item: item[1]["mae"]))
print("\nCost model comparison (sorted by MAE):")
for model_name, metrics in cost_scores.items():
    marker = "  <-- BEST" if model_name == cost_name else ""
    print(f"  {model_name:<30} MAE={metrics['mae']:.2f}  R²={metrics['r2']:.4f}{marker}")
cost_model = cost_models[cost_name].fit(X2, scenarios["total_cost"].to_numpy())
joblib.dump(cost_model, "models/best_cost_model.pkl")

report = {
    "data_note": "Generated demo data; replace with real factory history for operational decisions.",
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
    "cost_best": cost_name,
    "cost_models": cost_scores,
}
with open("models/model_comparison.json", "w", encoding="utf-8") as report_file:
    json.dump(report, report_file, indent=2)
with open("models/metrics.json", "w", encoding="utf-8") as metrics_file:
    json.dump({"N1": n1_metrics, "N2": n2_metrics, "cost": cost_scores}, metrics_file, indent=2)

print("Saved trained models and cross-validation metrics under models/.")
