from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier,
                              GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor,
                              VotingClassifier)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, f1_score, log_loss, precision_score,
                             recall_score, roc_auc_score, mean_absolute_error, mean_squared_error, r2_score)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from WORKING_GRAPH.features import add_graph_context_features
from WORKING_KG.features import add_knowledge_graph_features
from plot_results import create_plots

TARGET = "target_actual_on_time_0_1"
OUTCOME_COLUMNS = [TARGET, "target_actual_completed_quantity", "target_actual_completion_days"]
NON_FEATURE_COLUMNS = ["order_id", "decision_snapshot_date", "data_source_type"]
ALL_FAMILIES = ["tabular", "knn", "graph", "knowledge_graph", "ensemble"]
FAMILY_FOLDERS = {
    "tabular": "WORKING_ML",
    "knn": "WORKING_KNN",
    "graph": "WORKING_GRAPH",
    "knowledge_graph": "WORKING_KG",
    "ensemble": "WORKING_ENSEMBLE",
}


def load_data(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {TARGET, "decision_snapshot_date"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    frame["decision_snapshot_date"] = pd.to_datetime(frame["decision_snapshot_date"], errors="coerce")
    frame = frame.dropna(subset=[TARGET, "decision_snapshot_date"]).copy()
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="coerce")
    frame = frame.dropna(subset=[TARGET])
    frame[TARGET] = frame[TARGET].astype(int)
    invalid = sorted(set(frame[TARGET].unique()) - {0, 1})
    if invalid:
        raise ValueError(f"Target must be binary 0/1; found {invalid}")
    return frame.sort_values("decision_snapshot_date").reset_index(drop=True)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = set(NON_FEATURE_COLUMNS + OUTCOME_COLUMNS)
    return [c for c in frame.columns if c not in excluded]


def preprocessor(frame: pd.DataFrame, columns: list[str], scaled: bool = False, dense: bool = False) -> ColumnTransformer:
    categorical = [c for c in columns if not pd.api.types.is_numeric_dtype(frame[c])]
    numerical = [c for c in columns if c not in categorical]
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scaled:
        num_steps.append(("scale", StandardScaler()))
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense))])
    return ColumnTransformer([
        ("num", Pipeline(num_steps), numerical),
        ("cat", cat, categorical),
    ], remainder="drop")


def candidates(frame: pd.DataFrame, columns: list[str], family: str) -> dict[str, Pipeline]:
    if family == "tabular":
        estimators = {
            "logistic_regression": LogisticRegression(max_iter=5000, solver="liblinear", class_weight="balanced"),
            "decision_tree": DecisionTreeClassifier(max_depth=4, min_samples_leaf=2, class_weight="balanced", random_state=41),
            "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=41),
            "extra_trees": ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=41),
            "gradient_boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.04, max_depth=2, min_samples_leaf=2, random_state=41),
        }
        return {name: Pipeline([("prep", preprocessor(frame, columns, scaled=(name == "logistic_regression"), dense=(name == "gradient_boosting"))), ("model", model)]) for name, model in estimators.items()}
    if family == "knn":
        n = max(1, min(5, len(frame) - 1))
        return {"knn_distance_weighted": Pipeline([
            ("prep", preprocessor(frame, columns, scaled=True)),
            ("model", KNeighborsClassifier(n_neighbors=n, weights="distance", metric="minkowski")),
        ])}
    if family == "ensemble":
        members = []
        for name, model in [
            ("lr", LogisticRegression(max_iter=5000, solver="liblinear", class_weight="balanced")),
            ("rf", RandomForestClassifier(n_estimators=250, min_samples_leaf=2, class_weight="balanced", random_state=41)),
            ("et", ExtraTreesClassifier(n_estimators=250, min_samples_leaf=2, class_weight="balanced", random_state=41)),
        ]:
            members.append((name, Pipeline([("prep", preprocessor(frame, columns)), ("model", model)])))
        return {"soft_voting_lr_rf_et": VotingClassifier(estimators=members, voting="soft")}
    if family in {"graph", "knowledge_graph"}:
        # make_family_frame already adds the family-specific feature columns.
        return candidates(frame, columns, "tabular")
    raise ValueError(f"Unknown model family: {family}")


def make_family_frame(family: str, train: pd.DataFrame, valid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if family == "graph":
        return add_graph_context_features(train, train), add_graph_context_features(valid, train)
    if family == "knowledge_graph":
        return add_knowledge_graph_features(train, train), add_knowledge_graph_features(valid, train)
    return train.copy(), valid.copy()


def metric_row(y: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    pred = (proba >= 0.5).astype(int)
    result = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "brier": float(brier_score_loss(y, proba)),
        "log_loss": float(log_loss(y, np.column_stack([1 - proba, proba]), labels=[0, 1])),
    }
    result["roc_auc"] = float(roc_auc_score(y, proba)) if len(np.unique(y)) == 2 else float("nan")
    result["average_precision"] = float(average_precision_score(y, proba)) if np.any(y == 1) else float("nan")
    return result


def temporal_splits(frame: pd.DataFrame, min_train: int = 12, valid_size: int = 8, max_splits: int = 3):
    n = len(frame)
    if n <= min_train:
        raise ValueError(f"Need more than {min_train} time-ordered examples; found {n}.")
    starts = list(range(max(min_train, n - max_splits * valid_size), n, valid_size))
    splits = []
    for start in starts:
        end = min(start + valid_size, n)
        if end - start >= 3 and start >= min_train:
            splits.append((np.arange(0, start), np.arange(start, end)))
    return splits


def benchmark(frame: pd.DataFrame, columns: list[str], families: list[str] | None = None) -> tuple[pd.DataFrame, dict[str, list[dict]]]:
    scores: list[dict] = []
    fold_predictions: dict[str, list[dict]] = {}
    families = families or ALL_FAMILIES
    for fold_id, (train_idx, valid_idx) in enumerate(temporal_splits(frame), start=1):
        raw_train, raw_valid = frame.iloc[train_idx].copy(), frame.iloc[valid_idx].copy()
        if raw_train[TARGET].nunique() < 2:
            continue
        for family in families:
            train, valid = make_family_frame(family, raw_train, raw_valid)
            cols = columns + [c for c in train.columns if c.startswith(("graph_", "kg_"))]
            # De-duplicate for graph and KG engineered contexts.
            cols = list(dict.fromkeys(cols))
            for name, estimator in candidates(train, cols, family).items():
                y_train = train[TARGET].to_numpy()
                y_valid = valid[TARGET].to_numpy()
                if len(np.unique(y_valid)) < 2:
                    # Keep classification and probability metrics that do not require both classes.
                    pass
                try:
                    fitted = clone(estimator).fit(train[cols], y_train)
                    p = fitted.predict_proba(valid[cols])[:, 1]
                    key = f"{family}::{name}"
                    scores.append({"family": family, "model": name, "fold": len(scores), "fold_id": fold_id, **metric_row(y_valid, p)})
                    fold_predictions.setdefault(key, []).append({"fold_id": fold_id, "y": y_valid.tolist(), "p": p.tolist()})
                except Exception as exc:
                    warnings.warn(f"Skipping {family}/{name}: {exc}")
    table = pd.DataFrame(scores)
    if table.empty:
        raise ValueError("No temporal validation folds could be trained. Check sample size and target classes.")
    # Brier score is the primary ordering criterion for a probability-based decision. AUC/F1 are reported alongside it.
    metrics = ["brier", "log_loss", "balanced_accuracy", "f1", "roc_auc", "average_precision", "accuracy", "precision", "recall"]
    aggregations = {m: (m, "mean") for m in metrics}
    aggregations.update({"brier_std": ("brier", "std"), "fold_count": ("fold", "count")})
    leaderboard = (table.groupby(["family", "model"], as_index=False)
                   .agg(**aggregations)
                   .sort_values(["brier", "log_loss"], ascending=True, na_position="last"))
    return leaderboard, fold_predictions


def empirical_residual_radius(y_true: np.ndarray, y_pred: np.ndarray, coverage: float = 0.9) -> tuple[float, float]:
    """Empirical validation residual quantile; not a formal coverage guarantee."""
    residuals = np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))
    if len(residuals) == 0:
        return float("nan"), float("nan")
    q = min(1.0, np.ceil((len(residuals) + 1) * coverage) / len(residuals))
    radius = float(np.quantile(residuals, q, method="higher"))
    return radius, coverage


def benchmark_completion_days(frame: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, float, pd.DataFrame]:
    """Secondary capacity target; report chronological error and a demo conformal radius."""
    target = "target_actual_completion_days"
    if target not in frame:
        return pd.DataFrame(), float("nan"), pd.DataFrame()
    data = frame.copy()
    data[target] = pd.to_numeric(data[target], errors="coerce")
    data = data.dropna(subset=[target]).reset_index(drop=True)
    if len(data) <= 12:
        return pd.DataFrame(), float("nan"), pd.DataFrame()
    estimators = {
        "random_forest_regressor": RandomForestRegressor(n_estimators=250, min_samples_leaf=2, random_state=41),
        "extra_trees_regressor": ExtraTreesRegressor(n_estimators=250, min_samples_leaf=2, random_state=41),
        "gradient_boosting_regressor": GradientBoostingRegressor(n_estimators=100, learning_rate=0.04, max_depth=2, min_samples_leaf=2, random_state=41),
        "knn_regressor": KNeighborsRegressor(n_neighbors=5, weights="distance"),
    }
    results, residual_y, residual_p = [], [], []
    for fold_id, (train_idx, valid_idx) in enumerate(temporal_splits(data), start=1):
        train, valid = data.iloc[train_idx], data.iloc[valid_idx]
        for name, model in estimators.items():
            dense = name == "gradient_boosting_regressor"
            scaled = name == "knn_regressor"
            estimator = Pipeline([("prep", preprocessor(train, columns, scaled=scaled, dense=dense)), ("model", clone(model))])
            try:
                estimator.fit(train[columns], train[target])
                y, p = valid[target].to_numpy(dtype=float), estimator.predict(valid[columns])
                results.append({"fold_id": fold_id, "model": name, "mae_days": float(mean_absolute_error(y, p)),
                                "rmse_days": float(np.sqrt(mean_squared_error(y, p))),
                                "r2": float(r2_score(y, p)) if len(y) > 1 else float("nan")})
                if name == "random_forest_regressor":
                    residual_y.extend(y.tolist())
                    residual_p.extend(p.tolist())
            except Exception as exc:
                warnings.warn(f"Skipping completion-days model {name}: {exc}")
    table = pd.DataFrame(results)
    if table.empty:
        return table, float("nan"), table
    radius, _ = empirical_residual_radius(np.asarray(residual_y), np.asarray(residual_p), coverage=0.9)
    summary = (table.groupby("model", as_index=False).mean(numeric_only=True)
               .drop(columns=["fold_id"], errors="ignore").sort_values("mae_days"))
    return summary, radius, table


def main() -> None:
    parser = argparse.ArgumentParser(description="Chronological multimodel benchmark for apparel order acceptance")
    project_dir = Path(__file__).resolve().parent
    parser.add_argument("--family", choices=["all"] + ALL_FAMILIES, default="all",
                        help="Train every runnable family, or only one family folder")
    parser.add_argument("--data", type=Path,
                        default=project_dir / "dataset" / "Component4_ML_Training_Dataset_Synthetic_Sample.csv",
                        help="Model-ready CSV; defaults to dataset/Component4_ML_Training_Dataset_Synthetic_Sample.csv")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    active_families = ALL_FAMILIES if args.family == "all" else [args.family]
    if args.output is None:
        args.output = project_dir / "artifacts" if args.family == "all" else project_dir / FAMILY_FOLDERS[args.family]
    args.output.mkdir(parents=True, exist_ok=True)
    data = load_data(args.data)
    columns = feature_columns(data)
    leaderboard, fold_predictions = benchmark(data, columns, active_families)
    leaderboard.to_csv(args.output / "leaderboard.csv", index=False)
    regression_board, conformal_radius, regression_folds = benchmark_completion_days(data, columns)
    if not regression_board.empty:
        regression_board.to_csv(args.output / "completion_days_leaderboard.csv", index=False)
        regression_folds.to_csv(args.output / "completion_days_fold_metrics.csv", index=False)

    best_family = str(leaderboard.iloc[0]["family"])
    best_model = str(leaderboard.iloc[0]["model"])
    selected = None
    engineered = None
    final_columns: list[str] = []
    synthetic_only = bool(data.get("data_source_type", pd.Series(dtype=str)).astype(str).str.contains("SYNTHETIC", case=False).all())
    for family in active_families:
        family_data, _ = make_family_frame(family, data, data)
        family_columns = columns + [c for c in family_data.columns if c.startswith(("graph_", "kg_"))]
        family_columns = list(dict.fromkeys(family_columns))
        folder = project_dir / FAMILY_FOLDERS[family]
        model_dir = folder / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        family_board = leaderboard[leaderboard["family"] == family].copy()
        family_board.to_csv(folder / "leaderboard.csv", index=False)
        for model_name, candidate in candidates(family_data, family_columns, family).items():
            fitted = clone(candidate).fit(family_data[family_columns], data[TARGET])
            package = {"estimator": fitted, "family": family, "model": model_name,
                       "features": family_columns, "target": TARGET, "synthetic_only": synthetic_only}
            joblib.dump(package, model_dir / f"{model_name}.joblib")
            if family == best_family and model_name == best_model:
                selected, engineered, final_columns = fitted, family_data, family_columns
                joblib.dump(package, model_dir / "selected_model.joblib")
    if selected is None or engineered is None:
        raise RuntimeError("Could not fit the selected model candidate.")
    joblib.dump({"estimator": selected, "family": best_family, "model": best_model,
                 "features": final_columns, "target": TARGET, "synthetic_only": synthetic_only},
                args.output / "selected_model.joblib")

    full_prob = selected.predict_proba(engineered[final_columns])[:, 1]
    pred = data[[c for c in ["order_id", "decision_snapshot_date", "data_source_type"] if c in data]].copy()
    pred["estimated_p_on_time_in_sample"] = full_prob
    pred["warning"] = "In-sample demo output; never use for factory decisions"
    pred.to_csv(args.output / "demo_predictions.csv", index=False)

    # Explanations are measured on the latest chronological holdout, never on training rows.
    imp_train_raw, imp_valid_raw = data.iloc[:-max(3, min(8, len(data) // 5))].copy(), data.iloc[-max(3, min(8, len(data) // 5)):].copy()
    imp_train, imp_valid = make_family_frame(best_family, imp_train_raw, imp_valid_raw)
    if imp_train[TARGET].nunique() > 1:
        imp_cols = columns + [c for c in imp_train.columns if c.startswith(("graph_", "kg_"))]
        imp_cols = list(dict.fromkeys(imp_cols))
        diagnostic = clone(candidates(imp_train, imp_cols, best_family)[best_model]).fit(imp_train[imp_cols], imp_train[TARGET])
        importance = permutation_importance(diagnostic, imp_valid[imp_cols], imp_valid[TARGET],
                                            n_repeats=8, random_state=41, scoring="balanced_accuracy")
        pd.DataFrame({"feature": imp_cols, "importance_mean": importance.importances_mean,
                      "importance_std": importance.importances_std}).sort_values("importance_mean", ascending=False).to_csv(
                          args.output / "permutation_importance_latest_holdout.csv", index=False)

    synthetic = bool("data_source_type" in data and data["data_source_type"].astype(str).str.contains("SYNTHETIC", case=False).all())
    summary = {
        "rows": int(len(data)), "date_start": str(data.decision_snapshot_date.min().date()),
        "date_end": str(data.decision_snapshot_date.max().date()), "class_counts": data[TARGET].value_counts().sort_index().to_dict(),
        "feature_count": len(columns), "excluded_outcomes_and_identifiers": NON_FEATURE_COLUMNS + OUTCOME_COLUMNS,
        "evaluation": "chronological expanding-window validation; small-sample diagnostic only",
        "selection_rule": "lowest mean Brier score, then log loss; inspect all metrics and fold variance before choosing",
        "selected_candidate": {"family": best_family, "model": best_model},
        "completion_days_model_leaderboard": "completion_days_leaderboard.csv" if not regression_board.empty else None,
        "completion_days_demo_90pct_validation_residual_radius_days": conformal_radius,
        "uncertainty_note": "Empirical residual radius pooled from temporal validation folds; small synthetic sample, not a formal conformal coverage guarantee.",
        "synthetic_only": synthetic,
        "operational_use_allowed": False,
        "data_limitations": ["Synthetic sample; no factory validation", "Grade-aggregate workforce, not individual operator assignments", "Small sample makes model ranks unstable"],
    }
    (args.output / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    plot_paths = create_plots(args.output)
    from plot_results import create_family_plots
    for family in active_families:
        folder = project_dir / FAMILY_FOLDERS[family]
        create_family_plots(leaderboard[leaderboard["family"] == family], folder / "plots",
                            fold_predictions=fold_predictions)
    print(leaderboard.to_string(index=False))
    print(f"\nSaved diagnostics to {args.output.resolve()}")
    print(f"Created {len(plot_paths)} comparison plots in { (args.output / 'plots').resolve() }")
    if synthetic:
        print("SYNTHETIC DEMO ONLY: these results are not real-world accuracy estimates.")


if __name__ == "__main__":
    main()
