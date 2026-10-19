# Training the order models

The order form takes these values:

- Order quantity
- Handover days
- Available sewing lines
- Factory shift count: fixed at two; it is not entered in the form
- SMV per garment
- Total order value in LKR
- Style
- Priority

The live result uses a logistic regression classifier trained on `order_status` Accept/Reject decisions, N2 intervention/cost comparisons, and the N5 line-shift production plan. Synthetic `order_status` values are sampled from a smooth acceptance rate based on the same planned line-capacity features, so the sample classifier has uncertainty instead of learning a hard 0%/100% rule. `on_time_flag` remains the separate observed delivery outcome. The live recommendation uses a 70% predicted acceptance-probability cutoff; the production plan separately estimates output and required line-shifts. Replace the generated decisions with actual factory accept/reject records before relying on probabilities.

## Train

```powershell
python -m pip install -r requirements.txt
```
Run `python train.py` to compare and train all models in one pipeline. N5 is the aggregate line-shift capacity predictor used by the live planning page. Then start the web app with `python app.py`.

N5 learns standard garment minutes completed per 8-hour line-shift from completed order records. It filters training rows to one or two shifts to match the factory configuration. Its input features are order quantity, handover days, SMV, style and shift length; total line count and number of shifts are not used to predict the rate for one line-shift. Extra Trees, Random Forest and Huber Gradient Boosting are compared with order-grouped out-of-fold validation, and the lowest-MAE model is saved as `models/line_shift_capacity_model.pkl`. The lower/expected/upper range comes from the P10/P50/P90 out-of-fold residuals, bounded by observed target values. It is a planning range, not a probability guarantee. The app scales this pooled rate by available lines and one or both shifts.

`train.py` trains and compares the order and intervention models, and also evaluates line-feasibility and worker-operation models. The live on-time likelihood uses submitted line count and two shifts; the capacity plan separately estimates actual output against the deadline. Neither score guarantees an order outcome.

The feasibility model uses historical worker headcount mapped to 17-operator line equivalents, shifts per day, order details, and `order_status` (`Accept`/`Reject`). The app withholds its recommendation if the saved model was trained against the old on-time target; run `python train.py` to create a compatible model. The capacity plan remains available independently.

The N4 worker-operation regressor learns efficiency and quality from completed worker-operation allocations joined to worker and operation records. It uses order-grouped cross-validation to keep an order's allocations together. It is trained for research but is not used to assign lines in the live page because the checked-in worker and operation records are generated sample data, not a verified current roster.

The live result page does not show worker-level assignments or grade availability because the project has no verified current workforce roster.

`historical_capacity.py` calculates realized output minutes per line-shift from recorded completed quantities, SMV, order-to-completion days, historical headcount and shifts. It normalizes to an 8-hour shift, removes the outer 5% of observations, uses the selected style's data when at least 30 orders are available, and reports lower, expected and upper ranges. The 17-operator normalization comes from the factory poster. Because the order data has no line IDs or shift-level actual output, the app does not claim different measured rates for each line or shift.

The checked-in CSVs are reproducible generated sample data (`generate_data.py`). Their metrics and LKR estimates demonstrate the pipeline, not real factory performance or costs. Replace them with validated factory records before operational use. The available order data does not support current worker allocation, machine constraints, actual LKR labor costing, or a reliable order-specific quality forecast. Confirm live staffing and machine capacity before release.

The notebook displays model comparisons, confusion matrices, cost and workforce error charts, and DNN training curves inline.

`train.py` is the canonical training pipeline. `compare_models.ipynb` is an exploratory notebook with saved research outputs; use the script to generate artifacts matching the current app feature definitions.

The optional DNN uses TensorFlow/Keras Dense and Dropout layers. The GNN and GANN candidates use scikit-learn and NumPy. If TensorFlow is unavailable, the training script skips the DNN and compares the remaining models.

## Dataset notes

- `orders.csv` contains order details, `order_status` (`Accept`/`Reject`, the feasibility classification target), and `on_time_flag` (observed on-time outcome). The training pipeline maps worker headcount to 17-operator line equivalents for classifier inputs.
- `interventions.csv` contains scenario selection and estimated cost targets for N2.
- `allocations.csv` supplies typical conflict and worker-efficiency context. Its order-specific assignments are made after an order is accepted, so those fields are not used as live N1 inputs.
- `workers.csv`, `machines_shifts.csv`, and `operations.csv` do not currently have a reliable order/date join to the fulfilment target, so they are not direct line-model inputs.

## Replace sample records with factory data

The form fields mean order quantity in pieces, handover time in days, sewing lines that can be staffed on each shift, SMV in minutes per piece, order value in LKR, style, and priority. This factory's shift count is fixed at two. The capacity plan assumes each active line-shift meets the 17-operator grade standard; it does not know who is actually present.

For model training, populate `orders.csv` with these fields at minimum: `order_id`, `order_quantity`, `handover_days`, `required_smv_total`, `order_value`, `style_id`, `priority`, `n_available_workers`, `n_shifts` (1 or 2), `shift_hours`, and `order_status` (`Accept`/`Reject` recorded at the time the order decision was made). `on_time_flag` should remain the separate observed delivery outcome. Keep actual decisions and outcomes separate; do not infer real acceptance decisions from whether an order was delivered on time.

`interventions.csv` should contain actual trial or chosen scenarios, their added workers, overtime, machines, outsourced quantity, selection outcome, and actual itemized costs. The `Current` scenario deliberately has zero *additional* resources and zero incremental cost. It does not represent total wages or say current staffing is adequate. Do not train or report its historical choice score as a delivery-success probability.

The current line and shift output is an order-level average. To generate real line-specific or shift-specific predictions, maintain a separate actual output history keyed by order, line, date and shift, including good pieces, hours, staffing and downtime. The current generated sample data has no such observations, so the app intentionally avoids fabricating line rankings or separate shift rates.

Validate model performance with completed orders, actual handover outcomes, dated capacity snapshots, and actual intervention costs before using predictions for customer commitments.


