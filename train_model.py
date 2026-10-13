"""
 Train N1 (order fulfilment probability) and N2 (intervention decision)
 Generates all plots, saves models, scalers, and encoders.
"""
import os, json, joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                             precision_recall_curve, classification_report,
                             mean_absolute_error, r2_score,
                             accuracy_score, ConfusionMatrixDisplay)

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

# ==================================================================
# 1.  LOAD DATA
# ==================================================================
orders   = pd.read_csv("data/orders.csv")
workers  = pd.read_csv("data/workers.csv")
allocs   = pd.read_csv("data/allocations.csv")
scen     = pd.read_csv("data/interventions.csv")

# ==================================================================
# 2.  FEATURE ENGINEERING for N1
# ==================================================================
# Order level
orders["priority_enc"] = orders["priority"].map({"Low": 0, "Normal": 1, "High": 2})
orders["style_enc"] = LabelEncoder().fit_transform(orders["style_id"])

# Aggregate allocation stats per order
agg = allocs.groupby("order_id").agg(
    total_assigned_workload=("allocated_workload", "sum"),
    n_allocations=("allocation_id", "count"),
    n_conflicts=("conflict_flag", "sum"),
    avg_worker_eff=("actual_efficiency", "mean"),
    avg_worker_qual=("actual_quality", "mean"),
    distinct_workers=("worker_id", "nunique"),
    distinct_machines=("machine_id", "nunique"),
    active_allocs=("allocation_status", lambda x: (x == "Active").sum())
).reset_index()

df = orders.merge(agg, on="order_id", how="left").fillna(0)

# Worker availability features
avail_rate = (workers["availability_status"].value_counts(normalize=True).get("Available", 0))
workers["grade_enc"] = workers["grade"].map({"C": 0, "B": 1, "A": 2, "A+": 3, "Jumper": 4})

df["availability_rate"] = avail_rate
df["required_workload_min"] = df["required_workload_min"].astype(float)
df["workload_per_day"] = df["required_workload_min"] / df["handover_days"]

n1_features = [
    "order_quantity", "handover_days", "order_value",
    "required_smv_total", "required_workload_min", "workload_per_day",
    "priority_enc", "style_enc",
    "total_assigned_workload", "n_allocations", "n_conflicts",
    "avg_worker_eff", "avg_worker_qual", "distinct_workers",
    "distinct_machines", "active_allocs", "availability_rate"
]
n1_target = "on_time_flag"

X1 = df[n1_features].values
y1 = df[n1_target].values

scaler_n1 = StandardScaler()
X1s = scaler_n1.fit_transform(X1)

X1_train, X1_test, y1_train, y1_test = train_test_split(
    X1s, y1, test_size=0.2, random_state=42, stratify=y1
)

joblib.dump(scaler_n1, "models/scaler_n1.pkl")
joblib.dump(n1_features, "models/n1_features.pkl")

# ==================================================================
# 3.  TRAIN N1 — Neural Network (Keras) + RandomForest baseline
# ==================================================================
tf.random.set_seed(42)

model_n1 = models.Sequential([
    layers.Input(shape=(X1_train.shape[1],)),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])
model_n1.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model_n1.summary()

history_n1 = model_n1.fit(
    X1_train, y1_train,
    validation_split=0.2,
    epochs=60,
    batch_size=32,
    verbose=1,
    callbacks=[callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
)
model_n1.save("models/model_n1.h5")

# RandomForest baseline
rf_n1 = RandomForestClassifier(n_estimators=200, random_state=42)
rf_n1.fit(X1_train, y1_train)
joblib.dump(rf_n1, "models/model_n1_rf.pkl")

# ==================================================================
# 4.  PLOTS for N1
# ==================================================================
# 4.1 Training history (Loss vs Accuracy curves)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].plot(history_n1.history["loss"], label="Train Loss", lw=2)
ax[0].plot(history_n1.history["val_loss"], label="Val Loss", lw=2)
ax[0].set_title("N1 - Loss vs Epoch")
ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Loss"); ax[0].legend()
ax[1].plot(history_n1.history["accuracy"], label="Train Accuracy", lw=2)
ax[1].plot(history_n1.history["val_accuracy"], label="Val Accuracy", lw=2)
ax[1].set_title("N1 - Accuracy vs Epoch")
ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Accuracy"); ax[1].legend()
plt.tight_layout(); plt.savefig("plots/n1_training_history.png"); plt.close()

# 4.2 Predictions
y1_prob = model_n1.predict(X1_test, verbose=0).ravel()
y1_pred = (y1_prob >= 0.5).astype(int)

# Confusion Matrix
cm = confusion_matrix(y1_test, y1_pred)
fig, ax = plt.subplots(figsize=(5.5, 4.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Late", "On-Time"],
            yticklabels=["Late", "On-Time"], ax=ax)
ax.set_title("N1 - Confusion Matrix (Neural Net)")
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout(); plt.savefig("plots/n1_confusion_matrix.png"); plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(y1_test, y1_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("N1 - ROC Curve"); plt.legend(loc="lower right")
plt.tight_layout(); plt.savefig("plots/n1_roc_curve.png"); plt.close()

# Precision-Recall Curve
prec, rec, _ = precision_recall_curve(y1_test, y1_prob)
plt.figure(figsize=(6, 5))
plt.plot(rec, prec, color="green", lw=2)
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("N1 - Precision-Recall Curve")
plt.tight_layout(); plt.savefig("plots/n1_pr_curve.png"); plt.close()

# Feature Importance (RandomForest)
importances = rf_n1.feature_importances_
idx = np.argsort(importances)[::-1]
plt.figure(figsize=(9, 6))
sns.barplot(x=importances[idx], y=np.array(n1_features)[idx], palette="viridis")
plt.title("N1 - Feature Importance (RandomForest)")
plt.tight_layout(); plt.savefig("plots/n1_feature_importance.png"); plt.close()

# Actual vs Predicted (scatter of probabilities)
plt.figure(figsize=(6, 5))
sns.histplot(y1_prob[y1_test == 0], color="red", label="Actual Late", kde=True, bins=20)
sns.histplot(y1_prob[y1_test == 1], color="green", label="Actual On-Time", kde=True, bins=20)
plt.xlabel("Predicted P(On-Time)"); plt.title("N1 - Predicted Probability Distribution")
plt.legend()
plt.tight_layout(); plt.savefig("plots/n1_actual_vs_pred.png"); plt.close()

# Training vs Validation AUC curve (add after training)
plt.figure(figsize=(7, 5))
plt.plot(history_n1.history["accuracy"], label="Train Acc", lw=2)
plt.plot(history_n1.history["val_accuracy"], label="Val Acc", lw=2)
plt.fill_between(range(len(history_n1.history["accuracy"])),
                 history_n1.history["accuracy"],
                 history_n1.history["val_accuracy"], alpha=0.15, color="orange")
plt.title("N1 - Train vs Validation Accuracy (Gap)")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
plt.tight_layout(); plt.savefig("plots/n1_train_val_gap.png"); plt.close()

print("✅ N1 model + plots done. AUC =", round(roc_auc, 3))

# ==================================================================
# 5. FEATURE ENGINEERING for N2
# ==================================================================
scen["scenario_enc"] = LabelEncoder().fit_transform(scen["scenario_type"])

# Merge order-level features
scen_full = scen.merge(
    df[["order_id", "order_quantity", "handover_days", "order_value",
        "required_workload_min", "n_conflicts", "avg_worker_eff"]],
    on="order_id", how="left"
).fillna(0)

n2_features = [
    "scenario_enc", "added_workers", "overtime_hours",
    "added_machines", "outsourced_qty",
    "order_quantity", "handover_days", "order_value",
    "required_workload_min", "n_conflicts", "avg_worker_eff"
]

X2 = scen_full[n2_features].values
y2_decision = scen_full["selected_flag"].values          # classification (0/1)
y2_cost     = scen_full["total_cost"].values             # regression

scaler_n2 = StandardScaler()
X2s = scaler_n2.fit_transform(X2)
joblib.dump(scaler_n2, "models/scaler_n2.pkl")
joblib.dump(n2_features, "models/n2_features.pkl")

X2_train, X2_test, yd_train, yd_test, yc_train, yc_test = train_test_split(
    X2s, y2_decision, y2_cost, test_size=0.2, random_state=42, stratify=y2_decision
)

# ==================================================================
# 6. TRAIN N2 — NN for decision + RandomForest for cost
# ==================================================================
model_n2 = models.Sequential([
    layers.Input(shape=(X2_train.shape[1],)),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])
model_n2.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history_n2 = model_n2.fit(
    X2_train, yd_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    verbose=1,
    callbacks=[callbacks.EarlyStopping(patience=8, restore_best_weights=True)]
)
model_n2.save("models/model_n2.h5")

# Cost regressor
rf_cost = RandomForestRegressor(n_estimators=200, random_state=42)
rf_cost.fit(X2_train, yc_train)
joblib.dump(rf_cost, "models/model_n2_cost.pkl")

# ==================================================================
# 7. PLOTS for N2
# ==================================================================
fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].plot(history_n2.history["loss"], label="Train Loss", lw=2)
ax[0].plot(history_n2.history["val_loss"], label="Val Loss", lw=2)
ax[0].set_title("N2 - Loss vs Epoch")
ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Loss"); ax[0].legend()
ax[1].plot(history_n2.history["accuracy"], label="Train Accuracy", lw=2)
ax[1].plot(history_n2.history["val_accuracy"], label="Val Accuracy", lw=2)
ax[1].set_title("N2 - Accuracy vs Epoch")
ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Accuracy"); ax[1].legend()
plt.tight_layout(); plt.savefig("plots/n2_training_history.png"); plt.close()

y2_prob = model_n2.predict(X2_test, verbose=0).ravel()
y2_pred = (y2_prob >= 0.5).astype(int)

cm2 = confusion_matrix(yd_test, y2_pred)
fig, ax = plt.subplots(figsize=(5.5, 4.5))
sns.heatmap(cm2, annot=True, fmt="d", cmap="Greens",
            xticklabels=["Not Selected", "Selected"],
            yticklabels=["Not Selected", "Selected"], ax=ax)
ax.set_title("N2 - Confusion Matrix (Intervention Selection)")
plt.tight_layout(); plt.savefig("plots/n2_confusion_matrix.png"); plt.close()

# Cost regression actual vs predicted
yc_pred = rf_cost.predict(X2_test)
plt.figure(figsize=(6, 5.5))
plt.scatter(yc_test, yc_pred, alpha=0.5, color="teal")
lims = [0, max(yc_test.max(), yc_pred.max()) * 1.05]
plt.plot(lims, lims, "r--", lw=2)
plt.xlabel("Actual Cost"); plt.ylabel("Predicted Cost")
plt.title(f"N2 - Cost Regression (R²={r2_score(yc_test, yc_pred):.3f})")
plt.tight_layout(); plt.savefig("plots/n2_cost_regression.png"); plt.close()

# Correlation heatmap
plt.figure(figsize=(10, 8))
corr = df[n1_features + [n1_target]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Feature Correlation Heatmap")
plt.tight_layout(); plt.savefig("plots/correlation_heatmap.png"); plt.close()

# Target distribution
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.countplot(x=df[n1_target], ax=ax[0])
ax[0].set_title("On-Time vs Late (N1 Target)")
ax[0].set_xticklabels(["Late", "On-Time"])
sns.histplot(y2_cost, bins=30, kde=True, ax=ax[1], color="purple")
ax[1].set_title("Intervention Cost Distribution (N2)")
plt.tight_layout(); plt.savefig("plots/target_distribution.png"); plt.close()

# ==================================================================
# 8. METRICS SUMMARY
# ==================================================================
metrics = {
    "N1_neural_net": {
        "accuracy": float(accuracy_score(y1_test, y1_pred)),
        "roc_auc": float(roc_auc)
    },
    "N1_random_forest": {
        "accuracy": float(accuracy_score(y1_test, rf_n1.predict(X1_test)))
    },
    "N2_neural_net": {
        "accuracy": float(accuracy_score(yd_test, y2_pred))
    },
    "N2_cost_regressor": {
        "mae": float(mean_absolute_error(yc_test, yc_pred)),
        "r2": float(r2_score(yc_test, yc_pred))
    }
}
with open("models/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\n===== METRICS =====")
print(json.dumps(metrics, indent=2))
print("\n✅ All models saved to models/ ; all plots to plots/")