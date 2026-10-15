# Training the order models

The order form takes these six values:

- Order quantity
- Handover days
- SMV per garment
- Total order value in LKR
- Style
- Priority

The app derives total workload, workload per day, unit value, and capacity ratio from those values and a factory profile. Training and live prediction use the same feature builder in `model_features.py`.

## Train

```powershell
python -m pip install -r requirements.txt
# Optional: recreate the sample CSV files
python generate_data.py
```
Open `compare_models.ipynb` and run its cells from top to bottom to train and compare the models. Then start the web app with `python app.py`.

`generate_data.py` recreates the CSVs with 3,000 generated sample orders. Skip that command when training on real factory CSVs. The notebook trains and compares twelve classifier models for N1 and N2, including DNN, GANN, and GNN. It compares model probabilities using cross-validated Brier score with ROC-AUC as the tie-breaker. N2 folds group all scenarios for one order together. Eight cost regressors are compared separately by MAE. The score tables and selection criteria are saved to `models/model_comparison.json`; winning models are saved under `models/`.

The notebook displays the model comparisons, accuracy-versus-loss charts, confusion matrices, DNN training curves, and dataset overview plots inline.

`compare_models.ipynb` is also a standalone, single-notebook version of the feature engineering, neural models, training, and chart code. Run its cells from top to bottom; it reads the CSV inputs from `data/` and writes the trained model bundles and comparison reports under `models/`. Its training and plotting code is contained in the notebook.

The DNN uses TensorFlow/Keras Dense and Dropout layers. Its trained model is saved in native `.keras` format and loaded by the app when selected as the winner. The GNN treats each training order/scenario row as a graph node and links it to its nearest feature-similar training rows. It aggregates neighbour features as a message-passing step, then predicts with a neural network. The GANN searches a small population of neural architectures and training settings with a genetic algorithm, then fits the best configuration on the fold's training rows. GNN and GANN use scikit-learn and NumPy; DNN requires TensorFlow, included in `requirements.txt`.

## Dataset notes

- `orders.csv` contains the six form values and `on_time_flag`, the fulfilment target.
- `interventions.csv` contains scenario selection and estimated cost targets for N2.
- `allocations.csv` supplies typical conflict and worker-efficiency context. Its order-specific assignments are made after an order is accepted, so those fields are not used as live N1 inputs.
- `workers.csv`, `machines_shifts.csv`, and `operations.csv` do not currently have a reliable order/date join to the fulfilment target. The current generated data therefore does not treat them as direct N1 predictors.

The current CSVs are synthetic demonstration data. Cross-validation measures how well the model reproduces those generated labels; it is not evidence of accuracy in a real factory. Replace the sample data with completed orders, actual handover outcomes, dated capacity snapshots, and actual intervention costs before using predictions for customer commitments.


