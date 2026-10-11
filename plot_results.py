from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

# Keep Matplotlib's font cache out of restricted profile directories.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "nadee_research_mplconfig"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {
    "tabular": "#2667a8",
    "knn": "#db7b18",
    "graph": "#31915c",
    "knowledge_graph": "#8955a1",
    "ensemble": "#bd4451",
}


def create_plots(artifact_dir: Path) -> list[Path]:
    plots_dir = artifact_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    board_path = artifact_dir / "leaderboard.csv"
    if not board_path.exists():
        raise FileNotFoundError(f"Run train.py first; not found: {board_path}")
    board = pd.read_csv(board_path)
    saved: list[Path] = []
    plt.style.use("seaborn-v0_8-whitegrid")

    # Accuracy vs. log loss with short point IDs; a sidecar maps every ID to its model.
    ranked = board.sort_values("brier").reset_index(drop=True).copy()
    ranked["plot_id"] = [f"{i:02d}" for i in range(1, len(ranked) + 1)]
    ranked[["plot_id", "family", "model", "accuracy", "log_loss", "brier"]].to_csv(
        plots_dir / "model_code_map.csv", index=False)
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    for family, group in ranked.groupby("family", sort=False):
        ax.scatter(group["accuracy"], group["log_loss"], s=78, alpha=0.9,
                   color=COLORS.get(family, "#666666"), label=family.replace("_", " "),
                   edgecolors="white", linewidths=0.7)
    for i, row in ranked.iterrows():
        offsets = [(5, 4), (5, -8), (-18, 7), (-18, -9), (6, 11)]
        ax.annotate(row["plot_id"], (row["accuracy"], row["log_loss"]),
                    xytext=offsets[i % len(offsets)], textcoords="offset points",
                    fontsize=7, weight="bold", alpha=0.9)
    ax.set_yscale("log")
    ax.set_title("Model comparison: validation accuracy vs. log loss", weight="bold")
    ax.set_xlabel("Mean validation accuracy (higher is better)")
    ax.set_ylabel("Mean validation log loss (lower is better)")
    ax.legend(title="Model family", frameon=True)
    ax.text(0.01, -0.16, "Chronological folds; synthetic demo data. Lower-right is preferable. Point IDs map to plots/model_code_map.csv.",
            transform=ax.transAxes, fontsize=8, color="#444444")
    path = plots_dir / "accuracy_vs_log_loss.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    # Classification metrics for the ten candidates with the best mean Brier score.
    top = board.sort_values("brier").head(10).sort_values("brier", ascending=False).copy()
    labels = [f"{r.family} / {r.model}" for r in top.itertuples()]
    y = np.arange(len(top))
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    height = 0.2
    for i, metric in enumerate(["accuracy", "balanced_accuracy", "f1", "recall"]):
        ax.barh(y + (i - 1.5) * height, top[metric], height=height, label=metric.replace("_", " "))
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Mean validation score")
    ax.set_title("Classification metrics: top 10 by Brier score", weight="bold")
    ax.legend(ncol=4, loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    path = plots_dir / "classification_metrics_top10.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    # Brier and log loss are shown in separate panels because their scales differ.
    ranked = board.sort_values("brier", ascending=True).reset_index(drop=True)
    labels = [f"{r.family} / {r.model}" for r in ranked.itertuples()]
    y = np.arange(len(ranked))
    fig, axes = plt.subplots(1, 2, figsize=(14, max(7, 0.37 * len(ranked))), sharey=True, constrained_layout=True)
    axes[0].barh(y, ranked["brier"], xerr=ranked.get("brier_std"), color="#3976a8", alpha=0.88, capsize=2)
    axes[0].set_title("Brier score (lower is better)")
    axes[0].set_xlabel("Mean score ± fold standard deviation")
    axes[1].barh(y, ranked["log_loss"], color="#c97730", alpha=0.88)
    axes[1].set_title("Log loss (lower is better)")
    axes[1].set_xlabel("Mean validation score")
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    for ax in axes:
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Probability quality across all candidate models", weight="bold")
    path = plots_dir / "probability_scores_all_models.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    regression_path = artifact_dir / "completion_days_leaderboard.csv"
    if regression_path.exists():
        reg = pd.read_csv(regression_path).sort_values("mae_days", ascending=False)
        labels = [str(m).replace("_", " ") for m in reg["model"]]
        y = np.arange(len(reg))
        fig, axes = plt.subplots(1, 2, figsize=(12, max(4, 0.8 * len(reg))), sharey=True, constrained_layout=True)
        axes[0].barh(y, reg["mae_days"], color="#3976a8")
        axes[0].set_title("MAE in days (lower is better)")
        axes[1].barh(y, reg["rmse_days"], color="#c97730")
        axes[1].set_title("RMSE in days (lower is better)")
        axes[0].set_yticks(y, labels)
        axes[0].invert_yaxis()
        for ax in axes:
            ax.grid(axis="x", alpha=0.25)
            ax.set_xlabel("Mean chronological validation error")
        fig.suptitle("Completion-time regression model comparison", weight="bold")
        path = plots_dir / "completion_days_errors.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    regression_folds_path = artifact_dir / "completion_days_fold_metrics.csv"
    if regression_folds_path.exists():
        folds = pd.read_csv(regression_folds_path)
        if not folds.empty:
            fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
            for model, group in folds.groupby("model", sort=False):
                ax.plot(group["fold_id"], group["mae_days"], marker="o", linewidth=2,
                        label=str(model).replace("_", " "))
            ax.set_xticks(sorted(folds["fold_id"].unique()))
            ax.set_xlabel("Chronological validation fold")
            ax.set_ylabel("Mean absolute error (days; lower is better)")
            ax.set_title("Completion-time regression: validation MAE by fold", weight="bold")
            ax.legend(frameon=True)
            ax.grid(alpha=0.3)
            path = plots_dir / "completion_days_mae_by_fold.png"
            fig.savefig(path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            saved.append(path)
    return saved


def create_family_plots(board: pd.DataFrame, plots_dir: Path,
                        fold_predictions: dict[str, list[dict]] | None = None) -> list[Path]:
    """Write concise, family-only plots into WORKING_<FAMILY>/plots."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    if board.empty:
        return []
    board = board.sort_values("brier").reset_index(drop=True)
    family = str(board.iloc[0]["family"])
    board[["family", "model", "accuracy", "balanced_accuracy", "f1", "brier", "log_loss", "brier_std"]].to_csv(
        plots_dir / "metrics_used_for_plots.csv", index=False)
    saved: list[Path] = []

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.scatter(board["accuracy"], board["log_loss"], s=90, color=COLORS.get(family, "#555555"), edgecolors="white")
    for _, row in board.iterrows():
        ax.annotate(str(row["model"]).replace("_", " "), (row["accuracy"], row["log_loss"]),
                    xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("Mean chronological validation accuracy (higher is better)")
    ax.set_ylabel("Mean chronological validation log loss (lower is better)")
    ax.set_title(f"{family.replace('_', ' ').title()}: accuracy vs. log loss", weight="bold")
    path = plots_dir / "accuracy_vs_log_loss.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(10, max(4, 0.55 * len(board))), sharey=True, constrained_layout=True)
    y = np.arange(len(board))
    labels = [str(name).replace("_", " ") for name in board["model"]]
    axes[0].barh(y, board["brier"], xerr=board["brier_std"], color="#3976a8", capsize=2)
    axes[0].set_title("Brier score (lower is better)")
    axes[0].set_xlabel("Mean ± fold standard deviation")
    axes[1].barh(y, board["log_loss"], color="#c97730")
    axes[1].set_title("Log loss (lower is better)")
    axes[1].set_xlabel("Mean score")
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    path = plots_dir / "probability_scores.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    # Fold-wise metrics are plotted instead of epochs: these models do not all train iteratively.
    fold_predictions = fold_predictions or {}
    family_models = board["model"].astype(str).tolist()
    accuracy_rows, loss_rows = [], []
    for model in family_models:
        key = f"{family}::{model}"
        for result in fold_predictions.get(key, []):
            y_true = np.asarray(result["y"], dtype=int)
            probability = np.clip(np.asarray(result["p"], dtype=float), 1e-12, 1 - 1e-12)
            y_pred = (probability >= 0.5).astype(int)
            accuracy_rows.append({"model": model, "fold_id": result["fold_id"],
                                  "accuracy": float(np.mean(y_true == y_pred))})
            loss = -np.mean(y_true * np.log(probability) + (1 - y_true) * np.log(1 - probability))
            loss_rows.append({"model": model, "fold_id": result["fold_id"], "log_loss": float(loss)})

    if accuracy_rows:
        accuracy = pd.DataFrame(accuracy_rows)
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        for model, group in accuracy.groupby("model", sort=False):
            ax.plot(group["fold_id"], group["accuracy"], marker="o", linewidth=1.8,
                    label=str(model).replace("_", " "))
        ax.set_xticks(sorted(accuracy["fold_id"].unique()))
        ax.set_ylim(0, 1)
        ax.set_xlabel("Chronological validation fold")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{family.replace('_', ' ').title()}: validation accuracy by fold", weight="bold")
        ax.legend(fontsize=8, frameon=True)
        ax.grid(alpha=0.3)
        path = plots_dir / "validation_accuracy_by_fold.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

        losses = pd.DataFrame(loss_rows)
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        for model, group in losses.groupby("model", sort=False):
            ax.plot(group["fold_id"], group["log_loss"], marker="o", linewidth=1.8,
                    label=str(model).replace("_", " "))
        ax.set_xticks(sorted(losses["fold_id"].unique()))
        ax.set_xlabel("Chronological validation fold")
        ax.set_ylabel("Log loss (lower is better)")
        ax.set_title(f"{family.replace('_', ' ').title()}: validation log loss by fold", weight="bold")
        ax.legend(fontsize=8, frameon=True)
        ax.grid(alpha=0.3)
        path = plots_dir / "validation_log_loss_by_fold.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

        # Every classifier gets its own out-of-fold confusion matrix.
        for model in family_models:
            heldout = fold_predictions.get(f"{family}::{model}", [])
            if not heldout:
                continue
            matrix = np.zeros((2, 2), dtype=int)
            for result in heldout:
                y_true = np.asarray(result["y"], dtype=int)
                y_pred = (np.asarray(result["p"], dtype=float) >= 0.5).astype(int)
                for actual, predicted in zip(y_true, y_pred):
                    matrix[actual, predicted] += 1
            fig, ax = plt.subplots(figsize=(6.4, 5.4), constrained_layout=True)
            image = ax.imshow(matrix, cmap="Greens")
            ax.set_xticks([0, 1], ["Late (0)", "On time (1)"])
            ax.set_yticks([0, 1], ["Late (0)", "On time (1)"])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(f"{family.replace('_', ' ').title()} — on-time confusion matrix\n"
                         f"Chronological out-of-fold; {model.replace('_', ' ')}")
            threshold = matrix.max() / 2 if matrix.size else 0
            for row in range(2):
                for col in range(2):
                    ax.text(col, row, str(matrix[row, col]), ha="center", va="center",
                            color="white" if matrix[row, col] > threshold else "#222222", fontsize=12)
            fig.colorbar(image, ax=ax, shrink=0.82, label="Order count")
            safe_model = str(model).replace("/", "_").replace("\\", "_")
            path = plots_dir / f"confusion_matrix__{safe_model}.png"
            fig.savefig(path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            saved.append(path)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Create plots from a completed multimodel training run")
    parser.add_argument("--artifacts", type=Path, default=Path(__file__).resolve().parent / "artifacts")
    args = parser.parse_args()
    for path in create_plots(args.artifacts):
        print(path.resolve())


if __name__ == "__main__":
    main()
