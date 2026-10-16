# Training the order models

The order form takes these values:

- Order quantity
- Handover days
- Available sewing lines
- Available shifts per line (1–3)
- SMV per garment
- Total order value in LKR
- Style
- Priority

The live result combines N1 order feasibility, N2 intervention/cost comparisons, and the N5 line-shift production plan. N1 uses the six order-form details and the training-time factory profile. N2 compares intervention patterns whose sizes are taken from historical scenario medians; its selection score estimates similarity to historically selected scenarios, not a delivery improvement probability. The N2 cost model estimates intervention cost from the recorded scenario cost target. N5 supplies the line-shift output range and grade-based required staffing template.

## Train

```powershell
python -m pip install -r requirements.txt
```
Run `python train.py` to compare and train all models in one pipeline. N5 is the aggregate line-shift capacity predictor used by the live planning page. Then start the web app with `python app.py`.

N5 learns standard garment minutes completed per 8-hour line-shift from order records containing actual completed quantity, order/completion dates, historical headcount, shift count and shift length. Its input features are order quantity, handover days, SMV, garment style, entered line count, shift count and shift length. Order value and priority do not influence throughput predictions. It compares Extra Trees, Random Forest and Huber Gradient Boosting with order-grouped out-of-fold validation, selects the lowest MAE, and saves `models/line_shift_capacity_model.pkl`. The lower/expected/upper output estimates are based on out-of-fold model residuals and are bounded by observed historical target values. They are planning ranges, not probability guarantees. `models/model_comparison.json` records the selected model, validation metrics, target definition, features and training record counts.

`train.py` trains and compares the order and intervention models, and also evaluates line-feasibility and worker-operation models. The live result keeps the N1 order-level likelihood separate from N5's line/shift capacity decision, so the on-time score is not presented as proof that a selected line plan can meet the deadline.

The line-aware model uses historical worker headcount mapped to 17-operator line equivalents, shifts per day, order details, and the on-time outcome. It is evaluated in the training report but its probability is not presented as a live capacity decision.

The N4 worker-operation regressor learns efficiency and quality from completed worker-operation allocations joined to worker and operation records. It uses order-grouped cross-validation to keep an order's allocations together. The model is saved to `models/workforce_performance_model.pkl` for evaluation and future use with a verified current roster. The live capacity page does not turn historical employee rows into current worker assignments.

The live result page does not show worker-level assignments or grade availability because the project has no verified current workforce roster.

`historical_capacity.py` calculates realized output minutes per line-shift from recorded actual order quantities, order SMV, order-to-completion days, historical headcount and shifts. It normalizes to an 8-hour shift, removes the outer 5% of observations, uses the selected style's data when at least 30 orders are available, and reports the lower quartile, median and upper quartile. The 17-operator normalization is a historical line-equivalent assumption from the factory poster; the result does not claim any current worker count or grade availability.

The available order data does not support current worker allocation, machine constraints, actual LKR labor costing, or a reliable order-specific quality forecast. Confirm live staffing and machine capacity before release.

The notebook displays model comparisons, confusion matrices, cost and workforce error charts, and DNN training curves inline.

`compare_models.ipynb` contains the model comparison and training workflow, including N3 line feasibility, N4 worker-operation performance and N5 line-shift capacity. Run its cells from top to bottom from the project root to reproduce the model artifacts and comparison charts.

The DNN uses TensorFlow/Keras Dense and Dropout layers. Its trained model is saved in native `.keras` format and loaded by the app when selected as the winner. The GNN treats each training order/scenario row as a graph node and links it to its nearest feature-similar training rows. It aggregates neighbour features as a message-passing step, then predicts with a neural network. The GANN searches a small population of neural architectures and training settings with a genetic algorithm, then fits the best configuration on the fold's training rows. GNN and GANN use scikit-learn and NumPy; DNN requires TensorFlow, included in `requirements.txt`.

## Dataset notes

- `orders.csv` contains order details and `on_time_flag`, the fulfilment target. The training pipeline maps worker headcount to 17-operator line equivalents for line-model inputs.
- `interventions.csv` contains scenario selection and estimated cost targets for N2.
- `allocations.csv` supplies typical conflict and worker-efficiency context. Its order-specific assignments are made after an order is accepted, so those fields are not used as live N1 inputs.
- `workers.csv`, `machines_shifts.csv`, and `operations.csv` do not currently have a reliable order/date join to the fulfilment target, so they are not direct line-model inputs.

Validate model performance with completed orders, actual handover outcomes, dated capacity snapshots, and actual intervention costs before using predictions for customer commitments.


