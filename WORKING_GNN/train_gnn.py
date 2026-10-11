from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from train import TARGET, feature_columns, load_data, temporal_splits


def build_temporal_graph(frame: pd.DataFrame, history_per_relation: int = 8) -> sparse.csr_matrix:
    """Make a causal order graph: each order sees itself and recent earlier orders on its line/style."""
    n = len(frame)
    row_idx: list[int] = []
    col_idx: list[int] = []
    values: list[float] = []
    previous: dict[str, dict[object, list[int]]] = {
        "production_line": {}, "style_category": {}
    }
    line_values = frame["production_line"].to_numpy()
    style_values = frame["style_category"].to_numpy()
    for current in range(n):
        neighbors: set[int] = {current}
        for column, key in (("production_line", line_values[current]),
                            ("style_category", style_values[current])):
            neighbors.update(previous[column].get(key, [])[-history_per_relation:])
        for neighbor in neighbors:
            row_idx.append(current)
            col_idx.append(neighbor)
            values.append(1.0)
        for column, key in (("production_line", line_values[current]),
                            ("style_category", style_values[current])):
            previous[column].setdefault(key, []).append(current)
    adjacency = sparse.csr_matrix((values, (row_idx, col_idx)), shape=(n, n), dtype=np.float32)
    row_sums = np.asarray(adjacency.sum(axis=1)).ravel()
    return sparse.diags(1.0 / np.maximum(row_sums, 1.0)).dot(adjacency).tocsr()


def encode_features(frame: pd.DataFrame, columns: list[str], train_idx: np.ndarray) -> tuple[sparse.csr_matrix, dict]:
    categorical = [c for c in columns if not pd.api.types.is_numeric_dtype(frame[c])]
    numerical = [c for c in columns if c not in categorical]
    pieces = []
    state = {}
    if numerical:
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        train_values = imputer.fit_transform(frame.iloc[train_idx][numerical])
        all_values = imputer.transform(frame[numerical])
        scaler.fit(train_values)
        pieces.append(sparse.csr_matrix(scaler.transform(all_values), dtype=np.float32))
        state["numeric_imputer"] = imputer
        state["numeric_scaler"] = scaler
    if categorical:
        imputer = SimpleImputer(strategy="most_frequent")
        train_values = imputer.fit_transform(frame.iloc[train_idx][categorical]).astype(str)
        all_values = imputer.transform(frame[categorical]).astype(str)
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)
        encoder.fit(train_values)
        pieces.append(encoder.transform(all_values).tocsr())
        state["categorical_imputer"] = imputer
        state["categorical_encoder"] = encoder
    return sparse.hstack(pieces, format="csr", dtype=np.float32), state


def _init_params(input_dim: int, hidden_dim: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0, np.sqrt(2 / max(input_dim, 1)), (input_dim, hidden_dim)).astype(np.float32),
        "b1": np.zeros(hidden_dim, dtype=np.float32),
        "Wc": rng.normal(0, np.sqrt(2 / hidden_dim), (hidden_dim, 2)).astype(np.float32),
        "bc": np.zeros(2, dtype=np.float32),
        "Wr": rng.normal(0, np.sqrt(1 / hidden_dim), (hidden_dim, 1)).astype(np.float32),
        "br": np.zeros(1, dtype=np.float32),
    }


def _forward(features_aggregated: sparse.csr_matrix, adjacency: sparse.csr_matrix,
             params: dict[str, np.ndarray]):
    pre = np.asarray(features_aggregated @ params["W1"] + params["b1"], dtype=np.float32)
    hidden = np.maximum(pre, 0)
    neighbor_hidden = np.asarray(adjacency @ hidden, dtype=np.float32)
    logits = neighbor_hidden @ params["Wc"] + params["bc"]
    regression_scaled = (neighbor_hidden @ params["Wr"] + params["br"]).ravel()
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(np.clip(logits, -40, 40))
    probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    return pre, hidden, neighbor_hidden, probabilities, regression_scaled


def train_gcn(features: sparse.csr_matrix, adjacency: sparse.csr_matrix, train_idx: np.ndarray,
              valid_idx: np.ndarray, y_class: np.ndarray, y_days_scaled: np.ndarray,
              y_days: np.ndarray, days_mean: float, days_std: float, epochs: int, seed: int,
              hidden_dim: int = 32, learning_rate: float = 0.01):
    params = _init_params(features.shape[1], hidden_dim, seed)
    first_agg = adjacency.dot(features).tocsr()
    adjacency_t = adjacency.transpose().tocsr()
    first_moment = {name: np.zeros_like(value) for name, value in params.items()}
    second_moment = {name: np.zeros_like(value) for name, value in params.items()}
    history = {"train_loss": [], "validation_accuracy": [], "validation_mae_days": []}
    y_onehot = np.eye(2, dtype=np.float32)[y_class]

    for step in range(1, epochs + 1):
        pre, hidden, neighbor_hidden, probabilities, regression_scaled = _forward(first_agg, adjacency, params)
        p = np.clip(probabilities[train_idx], 1e-8, 1.0)
        class_loss = -np.mean(np.sum(y_onehot[train_idx] * np.log(p), axis=1))
        difference = regression_scaled[train_idx] - y_days_scaled[train_idx]
        absolute = np.abs(difference)
        huber = np.where(absolute <= 1, 0.5 * difference ** 2, absolute - 0.5)
        regression_loss = float(np.mean(huber))

        d_logits = np.zeros_like(probabilities)
        d_logits[train_idx] = (probabilities[train_idx] - y_onehot[train_idx]) / len(train_idx)
        d_regression = np.zeros(len(y_class), dtype=np.float32)
        d_regression[train_idx] = np.clip(difference, -1, 1) / len(train_idx)
        d_hidden_neighbors = d_logits @ params["Wc"].T + 0.2 * d_regression[:, None] @ params["Wr"].T
        d_hidden = np.asarray(adjacency_t @ d_hidden_neighbors, dtype=np.float32)
        d_pre = d_hidden * (pre > 0)
        gradients = {
            "W1": np.asarray(first_agg.T @ d_pre, dtype=np.float32),
            "b1": d_pre.sum(axis=0),
            "Wc": neighbor_hidden.T @ d_logits,
            "bc": d_logits.sum(axis=0),
            "Wr": 0.2 * neighbor_hidden.T @ d_regression[:, None],
            "br": 0.2 * d_regression.sum(keepdims=True),
        }
        for name, grad in gradients.items():
            grad = np.asarray(grad, dtype=np.float32)
            first_moment[name] = 0.9 * first_moment[name] + 0.1 * grad
            second_moment[name] = 0.999 * second_moment[name] + 0.001 * grad ** 2
            m_hat = first_moment[name] / (1 - 0.9 ** step)
            v_hat = second_moment[name] / (1 - 0.999 ** step)
            params[name] -= learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)

        _, _, _, validation_probabilities, validation_days_scaled = _forward(first_agg, adjacency, params)
        validation_pred = np.argmax(validation_probabilities[valid_idx], axis=1)
        validation_days = validation_days_scaled[valid_idx] * days_std + days_mean
        history["train_loss"].append(float(class_loss + 0.2 * regression_loss))
        history["validation_accuracy"].append(float(np.mean(validation_pred == y_class[valid_idx])))
        history["validation_mae_days"].append(float(np.mean(np.abs(validation_days - y_days[valid_idx]))))
    return params, history


def plot_curves(curves: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray,
                output_dir: Path, synthetic_only: bool) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    plt.style.use("seaborn-v0_8-whitegrid")
    note = "Synthetic demo data; chronological validation folds"
    if synthetic_only:
        note += " (not factory evidence)"

    for metric, title, ylabel, filename in [
        ("train_loss", "GNN — Training Loss", "Training loss", "gnn_loss.png"),
        ("validation_accuracy", "GNN — On-time Validation Accuracy", "Accuracy", "gnn_accuracy.png"),
        ("validation_mae_days", "GNN — Completion-day Validation MAE", "MAE (days)", "gnn_regression_mae.png"),
    ]:
        grouped = curves.groupby("epoch")[metric]
        mean = grouped.mean()
        std = grouped.std().fillna(0)
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        ax.plot(mean.index, mean.values, color="#2474b5", linewidth=2, label=f"Mean across {curves['fold'].nunique()} folds")
        if curves["fold"].nunique() > 1:
            ax.fill_between(mean.index, mean.values - std.values, mean.values + std.values,
                            color="#2474b5", alpha=0.16, label="±1 fold standard deviation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(frameon=True)
        ax.text(0.01, -0.17, note, transform=ax.transAxes, fontsize=8, color="#555555")
        path = output_dir / filename
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    matrix = np.zeros((2, 2), dtype=int)
    for actual, predicted in zip(y_true.astype(int), y_pred.astype(int)):
        matrix[actual, predicted] += 1
    fig, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    im = ax.imshow(matrix, cmap="Greens")
    ax.set_xticks([0, 1], ["Late (0)", "On time (1)"])
    ax.set_yticks([0, 1], ["Late (0)", "On time (1)"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("GNN On-time Confusion Matrix\nChronological out-of-fold predictions")
    for r in range(2):
        for c in range(2):
            ax.text(c, r, str(matrix[r, c]), ha="center", va="center",
                    color="white" if matrix[r, c] > matrix.max() / 2 else "#222222", fontsize=12)
    fig.colorbar(im, ax=ax, shrink=0.82, label="Order count")
    path = output_dir / "gnn_confusion_on_time.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)
    return saved


def run_gnn(data_path: Path, epochs: int = 150, seed: int = 41) -> list[Path]:
    project_dir = Path(__file__).resolve().parents[1]
    output_dir = Path(__file__).resolve().parent
    plots_dir = output_dir / "plots"
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    frame = load_data(data_path)
    columns = feature_columns(frame)
    y_class = frame[TARGET].to_numpy(dtype=int)
    y_days = pd.to_numeric(frame["target_actual_completion_days"], errors="coerce").to_numpy(dtype=float)
    if np.isnan(y_days).any():
        raise ValueError("GNN regression plot needs complete target_actual_completion_days values.")
    adjacency = build_temporal_graph(frame)
    folds = temporal_splits(frame)
    fold_curves = []
    oof_y, oof_pred = [], []
    final_params = None
    final_state = None

    for fold_id, (train_idx, valid_idx) in enumerate(folds, start=1):
        features, state = encode_features(frame, columns, train_idx)
        days_mean = float(y_days[train_idx].mean())
        days_std = float(y_days[train_idx].std()) or 1.0
        days_scaled = (y_days - days_mean) / days_std
        params, history = train_gcn(features, adjacency, train_idx, valid_idx, y_class, days_scaled,
                                    y_days, days_mean, days_std, epochs=epochs, seed=seed + fold_id)
        _, _, _, probabilities, predicted_days_scaled = _forward(adjacency.dot(features).tocsr(), adjacency, params)
        oof_y.extend(y_class[valid_idx].tolist())
        oof_pred.extend(np.argmax(probabilities[valid_idx], axis=1).tolist())
        fold_curves.append(pd.DataFrame({
            "fold": fold_id,
            "epoch": np.arange(1, epochs + 1),
            "train_loss": history["train_loss"],
            "validation_accuracy": history["validation_accuracy"],
            "validation_mae_days": history["validation_mae_days"],
        }))
        final_params = params
        final_state = {**state, "days_mean": days_mean, "days_std": days_std, "columns": columns,
                       "feature_count": int(features.shape[1]), "epochs": epochs, "graph_relations": ["production_line", "style_category"],
                       "history_per_relation": 8}

    curves = pd.concat(fold_curves, ignore_index=True)
    curves.to_csv(plots_dir / "gnn_training_curves.csv", index=False)
    synthetic_only = bool(frame.get("data_source_type", pd.Series(dtype=str)).astype(str).str.contains("SYNTHETIC", case=False).all())
    saved = plot_curves(curves, np.asarray(oof_y), np.asarray(oof_pred), plots_dir, synthetic_only)
    all_indices = np.arange(len(frame))
    full_features, full_state = encode_features(frame, columns, all_indices)
    full_days_mean = float(y_days.mean())
    full_days_std = float(y_days.std()) or 1.0
    full_days_scaled = (y_days - full_days_mean) / full_days_std
    final_params, _ = train_gcn(full_features, adjacency, all_indices, all_indices, y_class,
                                full_days_scaled, y_days, full_days_mean, full_days_std,
                                epochs=epochs, seed=seed)
    final_state = {**full_state, "days_mean": full_days_mean, "days_std": full_days_std,
                   "columns": columns, "feature_count": int(full_features.shape[1]),
                   "epochs": epochs, "graph_relations": ["production_line", "style_category"],
                   "history_per_relation": 8, "model_note": "Final all-row demo model; use OOF plots for evaluation."}
    np.savez_compressed(models_dir / "gnn_model.npz", **final_params)
    joblib.dump(final_state, models_dir / "gnn_preprocessing.joblib")
    summary = {"rows": len(frame), "validation_folds": len(folds), "epochs": epochs,
               "target_classification": TARGET, "target_regression": "target_actual_completion_days",
               "graph_edges": "each order aggregates itself and up to 8 previous orders sharing production_line or style_category",
               "data_is_synthetic": synthetic_only,
               "research_warning": f"{len(frame):,}-row synthetic demo; metrics are diagnostic only" if synthetic_only else "Validate on held-out factory periods before use"}
    (output_dir / "gnn_run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"GNN diagnostics saved under {plots_dir.resolve()}")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Temporal graph neural network benchmark for order outcomes")
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] / "dataset" / "Component4_ML_Training_Dataset_Synthetic_1000.csv")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()
    run_gnn(args.data, epochs=args.epochs, seed=args.seed)


if __name__ == "__main__":
    main()
