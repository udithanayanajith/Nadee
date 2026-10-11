# Smart Apparel-LK: order acceptance and workforce allocation research

Research prototype for the supplied Component 4 proposal. It predicts order-level on-time completion from information available at the decision snapshot, benchmarks several model families, estimates uncertainty, and reports an auditable model comparison. The operator-allocation documents inform the feature design and the next data-collection stage.

## Read this before interpreting results

The original sample CSV contains 40 synthetic order snapshots. The default training CSV contains 1,000 simulated rows generated from that sample's schema and plausible capacity rules; every row is labelled `SYNTHETIC_GENERATED_NOT_REAL_HISTORY`. Neither file contains genuine factory history. These datasets can demonstrate a pipeline, but cannot establish real factory accuracy, validate an allocation policy, or support individual-worker conflict detection.

The primary target is `target_actual_on_time_0_1`. Completion quantity and completion days are future outcomes and are excluded from predictors. IDs, snapshot date, and data-source markers are also excluded. Evaluation uses chronological holdouts; performance on this small synthetic sample is diagnostic only. Do not use the output for operational order acceptance.

## Model families

- `WORKING_ML`: logistic regression, decision tree, random forest, extra trees, and gradient boosting.
- `WORKING_KNN`: scaled nearest-neighbour classifier.
- `WORKING_GRAPH`: relational context features built from shared style/line contexts. This is a graph-aware baseline, not a trained GNN.
- `WORKING_GNN`: temporal GNN over order nodes, linked to up to eight prior orders sharing a line or style; predicts on-time completion and completion days.
- `WORKING_KG`: interpretable style, line, machine, and workforce-context facts converted into decision-time graph features.
- `WORKING_ENSEMBLE`: soft-voting blend of tabular models.
- `WORKING_KG/aggregate_conflicts.py`: aggregate grade-level capacity conflict check; does not fabricate named operators.

The current GNN models order-to-order context; it does not represent individual worker-operation assignments. Pseudonymous operator IDs, operation-to-machine mappings, time-windowed assignments, and sufficient outcomes are still needed for workforce allocation research. KNN is a classical ML baseline, not a deep-learning method.

## Simple folder layout

Runnable families follow the `WORKING_*` layout of the reference project. Each family folder keeps its own `models/`, `plots/`, training entrypoint, and leaderboard.

```text
research_project/
├── dataset/                 # default training CSV
├── WORKING_KNN/             # KNN model, plots, one-click family runner
├── WORKING_KG/              # knowledge-graph context models and plots
│   └── aggregate_conflicts.py
├── WORKING_GRAPH/           # graph-context baseline and plots
├── WORKING_ML/              # logistic/tree/forest/boosting models and plots
├── WORKING_ENSEMBLE/        # soft-voting model and plots
├── WORKING_GNN/             # temporal order graph GNN and plots
├── train.py                 # train all runnable families including GNN
└── requirements.txt
```

## Explainability and uncertainty

The benchmark reports permutation importance on the latest chronological holdout and an empirical validation-residual radius for the continuous completion-day target. The residual radius is not a formal conformal coverage guarantee. Probability quality is assessed with Brier score and log loss; it is not assumed to be solved by an explanation method. Explainability does not improve predictive accuracy by itself. A follow-on XAI study should compare SHAP, accumulated local effects, and counterfactuals for fidelity, stability, and usefulness to IE experts. LIME/SHAP are explanations, not more or less accurate predictors.

## Run

Python 3.10+ is recommended.

```powershell
cd D:\Freelance\Nadee\research_project_simple
python -m venv .venv_local
.\.venv_local\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train.py
```

To regenerate the 1,000-row simulated CSV, run `python dataset/generate_synthetic_dataset.py`. To train one family, run its entrypoint, for example `python WORKING_KNN/training_with_plots_KNN.py`, `python WORKING_KG/training_with_plots_KG.py`, or `python train.py --family gnn`.

The 1,000-row synthetic CSV in `dataset/` is selected automatically. Run `python train.py` to train all runnable families, including the GNN. Each `WORKING_*` folder receives that family's fitted models and plots; combined comparisons are saved under `artifacts/`. Use `python train.py --data dataset/Component4_ML_Training_Dataset_Synthetic_Sample.csv` to run the original 40-row sample instead.

### Training visualizations

- `WORKING_*/plots/confusion_matrix__<model>.png`: one chronological out-of-fold confusion matrix per classifier.
- `WORKING_*/plots/validation_accuracy_by_fold.png` and `validation_log_loss_by_fold.png`: validation trends for every classical classifier in that family.
- `WORKING_GNN/plots/gnn_loss.png`, `gnn_accuracy.png`, `gnn_regression_mae.png`, and `gnn_confusion_on_time.png`: GNN epoch history plus its out-of-fold confusion matrix.
- `artifacts/plots/completion_days_mae_by_fold.png`: completion-time regression MAE by chronological fold for each regressor.
- `artifacts/plots/accuracy_vs_log_loss.png`, `classification_metrics_top10.png`, `probability_scores_all_models.png`, and `completion_days_errors.png`: overall model comparisons.

Classical models use chronological validation folds rather than training epochs; the GNN has an epoch-based loss and metric history. The dataset has no quality-grade or crop target, so the project produces on-time confusion matrices. The 1,000-row file is simulated and must not be described as historical factory data or used as research evidence. Replace it with de-identified real production records before making research claims; never include names or direct identifiers.

## Data required for the real study

Store one row per decision snapshot, with only fields known at that time: order/style/SMV/quantity/deadline; present, absent, and already-allocated suitably skilled workforce; machine compatibility and downtime; shifts; line context; and prior production history. Outcomes should include actual completion date, completed quantity, quality/rework, and intervention used/cost. For operator suitability, add pseudonymous worker ID, operation ID, required machine/method, assignment start/end, and post-assignment efficiency/quality/output. Record expert overrides and subsequent outcomes. Maintain a time-based split by order or production period to prevent leakage.

## Scope

This is a research scaffold, not a production decision system. It does not invent individual operator records, validate sample costs, optimize live assignments, or claim causal effects. Before operational use, validate target definitions, data capture, cost assumptions, constraints, probability calibration, subgroup performance, and decision thresholds with factory IE/Quality/Finance stakeholders.

