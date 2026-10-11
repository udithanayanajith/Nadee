# Research design and decision rules

## Research question

How do decision-time order requirements, available skilled workforce, existing grade-level commitments, machine capacity, shifts, and production history affect the probability and uncertainty of on-time sewing-order completion? When real individual-level data become available, how does context-aware operator-operation suitability ranking affect feasible line performance and order feasibility?

## Planned claims

The contribution is the measured pathway from resource-conditioned capacity and uncertainty to an order-level completion probability and minimum-cost intervention decision. The project does not claim a new ML algorithm, new explainability algorithm, or novelty in order admission, worker scheduling, or skill matrices individually.

## Current executable study

The primary task is binary on-time completion classification. Secondary task is completion-days regression. Decision-time features are restricted to the CSV fields before outcomes. Model families include logistic/tree/forest/boosting, KNN, graph-context, knowledge-graph-derived context, and soft voting. Candidate selection sorts by mean Brier score, with log loss and discrimination metrics shown alongside it. Chronological expanding-window validation is used; the available 40-row synthetic sample is too small for reliable rankings. Final demo probabilities are trained on all rows and clearly labelled in-sample.

An empirical completion-time residual radius is computed from chronological validation predictions. Its coverage is not guaranteed on this tiny, synthetic, temporally drifting sample. Validate coverage on a substantially larger untouched real production period before using it as uncertainty evidence.

## Next study with factory records

1. Validate labels, snapshot timing, costs, grade eligibility, and machine/shift constraints with IE, Quality, Production, and Finance.
2. Build operator-operation-style-machine assignment records using pseudonymous workers and time windows; retain output, efficiency, quality/rework, and expert overrides.
3. Benchmark grade rules, logistic/linear baselines, tree ensembles, KNN, and ranking methods before neural models.
4. Use a heterogeneous graph only after entity links and enough repeated observations exist. Compare GNN against feature-based baselines under the same time split.
5. Tune probability calibration and conformal intervals using separate chronological calibration windows. Report Brier/log loss, AUROC/AUPRC, sensitivity, false-accept rate, MAE/RMSE, interval coverage/width, and subgroup or production-regime results.
6. Compare explanation methods using held-out fidelity, stability under small perturbations, and IE expert review. Explanations are not scored as predictive accuracy.
7. Evaluate intervention scenarios with stakeholder-approved cost functions and hard feasibility constraints. Assess whether uncertainty or allocation conflicts change the recommended action; do not infer causal benefit from observational predictions alone.
