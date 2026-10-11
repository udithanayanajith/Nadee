# Smart Apparel-LK: order acceptance and workforce allocation research

Research prototype for the supplied Component 4 proposal. It predicts order-level on-time completion from information available at the decision snapshot, benchmarks several model families, estimates uncertainty, and reports an auditable model comparison. The operator-allocation documents inform the feature design and the next data-collection stage.

## Read this before interpreting results

The provided CSV and workbook samples are labelled `SYNTHETIC_DEMO_ONLY`. They contain roughly 40 order snapshots, aggregate operator counts by grade, and no stable worker-operation history. They can demonstrate a pipeline, but cannot establish real factory accuracy, validate an allocation policy, or support individual-worker conflict detection. The project refuses to present a synthetic run as factory evidence. The placeholder intervention costs and grade rules in the workbook are not used as validated facts.

The primary target is `target_actual_on_time_0_1`. Completion quantity and completion days are future outcomes and are excluded from predictors. IDs, snapshot date, and data-source markers are also excluded. Evaluation uses chronological holdouts; performance on this small synthetic sample is diagnostic only. Do not use the output for operational order acceptance.

## Model families

- `WORKING_ML`: logistic regression, decision tree, random forest, extra trees, and gradient boosting.
- `WORKING_KNN`: scaled nearest-neighbour classifier.
- `WORKING_GRAPH`: relational context features built from shared style/line contexts. This is a graph-aware baseline, not a trained GNN.
- `WORKING_KG`: interpretable style, line, machine, and workforce-context facts converted into decision-time graph features.
- `WORKING_ENSEMBLE`: soft-voting blend of tabular models.
- `WORKING_KG/aggregate_conflicts.py`: aggregate grade-level capacity conflict check; does not fabricate named operators.

With pseudonymous operator IDs, operation-to-machine mappings, time-windowed assignments, and sufficient historical outcomes, the graph track can be extended to a heterogeneous GNN. The current aggregate grade counts do not encode operator-operation edges, so reporting a trained GNN here would be misleading. KNN is a classical ML baseline, not a deep-learning method.

## Simple folder layout

Runnable families follow the `WORKING_*` layout of the reference project. Each family folder keeps its own `models/`, `plots/`, training entrypoint, and leaderboard. `WORKING_GNN/` is reserved with an explanation and empty output folders until real graph relationships are collected.

```text
research_project/
├── dataset/                 # default training CSV
├── WORKING_KNN/             # KNN model, plots, one-click family runner
├── WORKING_KG/              # knowledge-graph context models and plots
│   └── aggregate_conflicts.py
├── WORKING_GRAPH/           # graph-context baseline and plots
├── WORKING_ML/              # logistic/tree/forest/boosting models and plots
├── WORKING_ENSEMBLE/        # soft-voting model and plots
├── WORKING_GNN/             # reserved; needs operator-level graph data
├── train.py                 # train all runnable families
└── requirements.txt
```

## Explainability and uncertainty

The benchmark reports permutation importance on the latest chronological holdout and an empirical validation-residual radius for the continuous completion-day target. The residual radius is not a formal conformal coverage guarantee. Probability quality is assessed with Brier score and log loss; it is not assumed to be solved by an explanation method. Explainability does not improve predictive accuracy by itself. A follow-on XAI study should compare SHAP, accumulated local effects, and counterfactuals for fidelity, stability, and usefulness to IE experts. LIME/SHAP are explanations, not more or less accurate predictors.

## Run

Python 3.10+ is recommended.

```powershell
cd D:\Freelance\Nadee\research_project
python -m venv .venv_local
.\.venv_local\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train.py
```

To train just one family, run its entrypoint, for example `python WORKING_KNN/training_with_plots_KNN.py` or `python WORKING_KG/training_with_plots_KG.py`.

The sample CSV is stored in `dataset/` and is selected automatically. Run `python train.py` to train all runnable families. Each `WORKING_*` folder receives that family's fitted models, leaderboard, and plots; the combined leaderboard and overall comparison plots are saved under `artifacts/`. The plots compare validation accuracy against log loss, probability scores, classification metrics, and completion-day errors. No acceptance threshold is applied. To use another CSV, pass `--data path\to\file.csv`. Replace the sample with de-identified real historical data before making research claims; never include names or direct identifiers.

## Data required for the real study

Store one row per decision snapshot, with only fields known at that time: order/style/SMV/quantity/deadline; present, absent, and already-allocated suitably skilled workforce; machine compatibility and downtime; shifts; line context; and prior production history. Outcomes should include actual completion date, completed quantity, quality/rework, and intervention used/cost. For operator suitability, add pseudonymous worker ID, operation ID, required machine/method, assignment start/end, and post-assignment efficiency/quality/output. Record expert overrides and subsequent outcomes. Maintain a time-based split by order or production period to prevent leakage.

## Scope

This is a research scaffold, not a production decision system. It does not invent individual operator records, validate sample costs, optimize live assignments, or claim causal effects. Before operational use, validate target definitions, data capture, cost assumptions, constraints, probability calibration, subgroup performance, and decision thresholds with factory IE/Quality/Finance stakeholders.
"# Nadee" 
