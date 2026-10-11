# WORKING_GNN

This folder contains a temporal graph neural network for the available order-level targets: on-time completion and completion days. Each order snapshot is a node; it receives messages from itself and up to eight earlier orders sharing its production line or style. Edges use only those decision-time context fields.

Run `python WORKING_GNN/training_with_plots_GNN.py` or `python train.py --family gnn`. The default is 150 epochs. It writes `gnn_loss.png`, `gnn_accuracy.png`, `gnn_regression_mae.png`, and `gnn_confusion_on_time.png` under `plots/`. The loss and epoch curves are reported across chronological folds; the confusion matrix uses out-of-fold predictions. `models/` stores the final demo weights and preprocessing state.

This GNN models similarities between orders; it does not represent individual workers or optimize worker-operation assignments. The default 1,000-row CSV is simulated from the 40-row sample schema, so its metrics are only a pipeline demonstration. It has no crop or quality target; do not create crop/quality confusion matrices from this dataset.
