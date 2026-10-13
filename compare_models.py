"""
 Multi-model comparison for Component 4:
  - DNN (Keras)
  - Decision Tree
  - Random Forest
  - Gradient Boosting (sklearn)
  - XGBoost
  - LightGBM
  - GANN (Genetic Algorithm Neural Network via DEAP)

 Selects the BEST model by ROC-AUC (N1) and by F1 + Cost / R² (N2).
 Generates comparison plots and saves the winner for the Flask app.
"""
import os, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve, auc, mean_absolute_error, r2_score)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

from deap import base, creator, tools, algorithms

#  Project setup and reproducibility
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ==================================================================
# 1.  LOAD DATA & FEATURE ENGINEERING
# ==================================================================
# Load the raw CSV files and create a merged feature set for the
# first prediction task (N1: on-time completion probability).
orders  = pd.read_csv("data/orders.csv")
workers = pd.read_csv("data/workers.csv")
allocs  = pd.read_csv("data/allocations.csv")
scen    = pd.read_csv("data/interventions.csv")

orders["priority_enc"] = orders["priority"].map({"Low": 0, "Normal": 1, "High": 2})
orders["style_enc"]    = LabelEncoder().fit_transform(orders["style_id"])

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
avail_rate = workers["availability_status"].value_counts(normalize=True).get("Available", 0)
df["availability_rate"] = avail_rate
df["workload_per_day"]  = df["required_workload_min"] / df["handover_days"]

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
    X1s, y1, test_size=0.2, random_state=SEED, stratify=y1
)
joblib.dump(scaler_n1, "models/scaler_n1.pkl")
joblib.dump(n1_features, "models/n1_features.pkl")

# ==================================================================
# 2.  GANN — Genetic Algorithm Neural Network
# ==================================================================
# GANN is used to evolve a strong neural-network configuration while
# keeping the architecture within safe training bounds.
def build_keras_model(input_dim, hidden=(64, 32, 16), dropout=0.3, lr=1e-3):
    m = models.Sequential()
    m.add(layers.Input(shape=(input_dim,)))
    for h in hidden:
        m.add(layers.Dense(h, activation="relu"))
        m.add(layers.Dropout(dropout))
    m.add(layers.Dense(1, activation="sigmoid"))
    m.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="binary_crossentropy",
              metrics=["accuracy"])
    return m

def gann_fitness(individual, input_dim, X_tr, y_tr, X_va, y_va):
    """
    individual = [h1, h2, dropout, lr_exp, epochs]
    NOTE: all values are clamped to safe ranges before building the model,
          because GA mutation (mutGaussian) can push them out of bounds.
    """
    h1, h2, drop, lr_exp, ep = individual

    # ---- CLAMP to safe ranges ----
    h1     = int(np.clip(h1,     8,   256))
    h2     = int(np.clip(h2,     4,   128))
    drop   = float(np.clip(drop, 0.0, 0.5))      # must be in [0, 1)
    lr_exp = float(np.clip(lr_exp, 2.0, 4.5))
    ep     = int(np.clip(ep,     5,    50))

    lr = 10 ** (-lr_exp)

    m = build_keras_model(input_dim, hidden=(h1, h2, 8), dropout=drop, lr=lr)
    hist = m.fit(X_tr, y_tr, validation_data=(X_va, y_va),
                 epochs=ep, batch_size=32, verbose=0)
    return (hist.history["val_accuracy"][-1],)

def run_gann(X_train, y_train, input_dim, pop_size=8, generations=5):
    """Small GA with DEAP — values clamped everywhere to avoid Keras errors."""
    if "FitnessMax" not in dir(creator):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if "Individual" not in dir(creator):
        creator.create("Individual", list, fitness=creator.FitnessMax)

    Xg_tr, Xg_va, yg_tr, yg_va = train_test_split(
        X_train, y_train, test_size=0.2, random_state=SEED, stratify=y_train
    )

    toolbox = base.Toolbox()
    toolbox.register("attr_h1",    np.random.randint, 16, 128)
    toolbox.register("attr_h2",    np.random.randint, 8,  64)
    toolbox.register("attr_drop",  np.random.uniform, 0.05, 0.45)   # never negative
    toolbox.register("attr_lrexp", np.random.uniform, 2.5, 3.8)
    toolbox.register("attr_ep",    np.random.randint, 10, 30)

    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_h1, toolbox.attr_h2, toolbox.attr_drop,
                      toolbox.attr_lrexp, toolbox.attr_ep), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", gann_fitness,
                     input_dim=input_dim, X_tr=Xg_tr, y_tr=yg_tr,
                     X_va=Xg_va, y_va=yg_va)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.08, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.3,
                                    ngen=generations, stats=stats,
                                    halloffame=hof, verbose=False)

    # ==========================================================
    # Clamp the best individual before final training
    # ==========================================================
    # This is the critical stability guard that keeps the GA output
    # inside valid ranges for the Keras model.
    h1, h2, drop, lr_exp, ep = hof[0]
    h1     = int(np.clip(h1,     16,  128))
    h2     = int(np.clip(h2,      8,   64))
    drop   = float(np.clip(drop, 0.05, 0.45))    # never negative, never > 0.45
    lr_exp = float(np.clip(lr_exp, 2.5, 3.8))
    ep     = int(np.clip(ep,     10,   30))
    lr     = 10 ** (-lr_exp)

    print(f"     GANN best config: hidden=({h1},{h2},8) "
          f"drop={drop:.3f} lr={lr:.5f} epochs={ep}")

    # ---- Refit best individual on FULL training set ----
    final = build_keras_model(input_dim, hidden=(h1, h2, 8),
                              dropout=drop, lr=lr)
    final.fit(X_train, y_train, epochs=ep, batch_size=32, verbose=0)

    # ---- Fitness evolution plot ----
    plt.figure(figsize=(7, 4.5))
    plt.plot(log.select("max"), marker="o", lw=2, color="darkgreen")
    plt.title("GANN - Fitness Evolution (Val Accuracy)")
    plt.xlabel("Generation"); plt.ylabel("Best Val Accuracy")
    plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("plots/gann_fitness_evolution.png"); plt.close()

    return final, {"hidden": [h1, h2, 8], "dropout": drop,
                   "lr": lr, "epochs": ep}

# ==================================================================
# 3.  TRAIN ALL MODELS — N1
# ==================================================================
# N1 predicts whether an order will be completed on time.
print("\n🚀 Training N1 models...")
n1_models, n1_preds, n1_probs, n1_metrics = {}, {}, {}, {}

# --- 3.1 DNN ---
print("  → DNN")
dnn = build_keras_model(X1_train.shape[1], hidden=(64, 32, 16))
dnn.fit(X1_train, y1_train, validation_split=0.2, epochs=40, batch_size=32,
        verbose=0, callbacks=[callbacks.EarlyStopping(patience=8,
        restore_best_weights=True)])
n1_models["DNN"] = dnn
n1_probs["DNN"]  = dnn.predict(X1_test, verbose=0).ravel()
n1_preds["DNN"]  = (n1_probs["DNN"] >= 0.5).astype(int)

# --- 3.2 Decision Tree ---
print("  → Decision Tree")
dt = DecisionTreeClassifier(max_depth=8, random_state=SEED)
dt.fit(X1_train, y1_train)
n1_models["Decision Tree"] = dt
n1_probs["Decision Tree"]  = dt.predict_proba(X1_test)[:, 1]
n1_preds["Decision Tree"]  = dt.predict(X1_test)

# --- 3.3 Random Forest ---
print("  → Random Forest")
rf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
rf.fit(X1_train, y1_train)
n1_models["Random Forest"] = rf
n1_probs["Random Forest"]  = rf.predict_proba(X1_test)[:, 1]
n1_preds["Random Forest"]  = rf.predict(X1_test)

# --- 3.4 Gradient Boosting (sklearn) ---
print("  → Gradient Boosting (sklearn)")
gb = GradientBoostingClassifier(n_estimators=150, learning_rate=0.08,
                                 max_depth=4, random_state=SEED)
gb.fit(X1_train, y1_train)
n1_models["Gradient Boosting"] = gb
n1_probs["Gradient Boosting"]  = gb.predict_proba(X1_test)[:, 1]
n1_preds["Gradient Boosting"]  = gb.predict(X1_test)

# --- 3.5 XGBoost ---
print("  → XGBoost")
xgb = XGBClassifier(n_estimators=250, learning_rate=0.07, max_depth=5,
                    random_state=SEED, eval_metric="logloss", verbosity=0)
xgb.fit(X1_train, y1_train)
n1_models["XGBoost"] = xgb
n1_probs["XGBoost"]  = xgb.predict_proba(X1_test)[:, 1]
n1_preds["XGBoost"]  = xgb.predict(X1_test)

# --- 3.6 LightGBM ---
print("  → LightGBM")
lgb = LGBMClassifier(n_estimators=250, learning_rate=0.07, max_depth=5,
                     random_state=SEED, verbose=-1)
lgb.fit(X1_train, y1_train)
n1_models["LightGBM"] = lgb
n1_probs["LightGBM"]  = lgb.predict_proba(X1_test)[:, 1]
n1_preds["LightGBM"]  = lgb.predict(X1_test)

# --- 3.7 GANN ---
print("  → GANN (this takes ~1-2 min)")
gann, gann_cfg = run_gann(X1_train, y1_train, X1_train.shape[1],
                          pop_size=8, generations=5)
n1_models["GANN"] = gann
n1_probs["GANN"]  = gann.predict(X1_test, verbose=0).ravel()
n1_preds["GANN"]  = (n1_probs["GANN"] >= 0.5).astype(int)
print("     GANN best config:", gann_cfg)

# --- Metrics ---
for name in n1_models:
    n1_metrics[name] = {
        "accuracy":  round(accuracy_score(y1_test, n1_preds[name]), 4),
        "precision": round(precision_score(y1_test, n1_preds[name], zero_division=0), 4),
        "recall":    round(recall_score(y1_test, n1_preds[name], zero_division=0), 4),
        "f1":        round(f1_score(y1_test, n1_preds[name], zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y1_test, n1_probs[name]), 4),
    }

# --- Best model = highest ROC-AUC ---
best_n1_name = max(n1_metrics, key=lambda k: n1_metrics[k]["roc_auc"])
print(f"\n🏆 Best N1 model: {best_n1_name}  "
      f"(ROC-AUC = {n1_metrics[best_n1_name]['roc_auc']})")

joblib.dump({"name": best_n1_name,
             "model": n1_models[best_n1_name],
             "scaler": scaler_n1,
             "features": n1_features,
             "type": "keras" if best_n1_name in ("DNN", "GANN") else "sklearn"},
            "models/best_n1_model.pkl")

# ==================================================================
# 4.  N1 COMPARISON PLOTS
# ==================================================================
# Visual comparison shows how the models rank across key classification metrics.
# 4.1 Bar chart — metrics
mdf = pd.DataFrame(n1_metrics).T
fig, ax = plt.subplots(figsize=(12, 5.5))
mdf.plot(kind="bar", ax=ax, colormap="viridis", edgecolor="black")
ax.set_title("N1 — Model Comparison (Higher is Better)")
ax.set_ylabel("Score"); ax.set_ylim(0, 1.05)
ax.legend(loc="lower right", ncol=5, fontsize=9)
plt.xticks(rotation=20, ha="right")
plt.tight_layout(); plt.savefig("plots/model_comparison_n1.png"); plt.close()

# 4.2 ROC curves for all
plt.figure(figsize=(8, 6.5))
colors = plt.cm.tab10(np.linspace(0, 1, len(n1_models)))
for (name, prob), c in zip(n1_probs.items(), colors):
    fpr, tpr, _ = roc_curve(y1_test, prob)
    plt.plot(fpr, tpr, lw=2, color=c,
             label=f"{name} (AUC={auc(fpr, tpr):.3f})")
plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlabel("FPR"); plt.ylabel("TPR")
plt.title("N1 — ROC Comparison of All Models")
plt.legend(loc="lower right", fontsize=9)
plt.tight_layout(); plt.savefig("plots/roc_comparison_n1.png"); plt.close()

# 4.3 Confusion matrices — small multiples
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.ravel()
for i, name in enumerate(n1_models):
    cm = confusion_matrix(y1_test, n1_preds[name])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i],
                xticklabels=["Late", "OnT"], yticklabels=["Late", "OnT"])
    axes[i].set_title(name)
for j in range(len(n1_models), len(axes)):
    axes[j].axis("off")
plt.suptitle("N1 — Confusion Matrices", fontsize=14)
plt.tight_layout(); plt.savefig("plots/cm_grid_n1.png"); plt.close()

# ==================================================================
# 5.  N2 — TRAIN & COMPARE INTERVENTION STRATEGIES
# ==================================================================
# N2 chooses the best intervention strategy and estimates the associated cost.
print("\n🚀 Training N2 models...")
scen["scenario_enc"] = LabelEncoder().fit_transform(scen["scenario_type"])
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
y2_dec = scen_full["selected_flag"].values
y2_cost = scen_full["total_cost"].values

scaler_n2 = StandardScaler()
X2s = scaler_n2.fit_transform(X2)
joblib.dump(scaler_n2, "models/scaler_n2.pkl")
joblib.dump(n2_features, "models/n2_features.pkl")

X2_tr, X2_te, yd_tr, yd_te, yc_tr, yc_te = train_test_split(
    X2s, y2_dec, y2_cost, test_size=0.2, random_state=SEED, stratify=y2_dec
)

n2_models, n2_preds, n2_probs, n2_metrics = {}, {}, {}, {}

# DNN
dnn2 = build_keras_model(X2_tr.shape[1], hidden=(64, 32, 16))
dnn2.fit(X2_tr, yd_tr, validation_split=0.2, epochs=35, batch_size=32,
         verbose=0, callbacks=[callbacks.EarlyStopping(patience=8,
         restore_best_weights=True)])
n2_models["DNN"] = dnn2
n2_probs["DNN"]  = dnn2.predict(X2_te, verbose=0).ravel()
n2_preds["DNN"]  = (n2_probs["DNN"] >= 0.5).astype(int)

for name, mdl in [
    ("Decision Tree", DecisionTreeClassifier(max_depth=8, random_state=SEED)),
    ("Random Forest", RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)),
    ("Gradient Boosting", GradientBoostingClassifier(n_estimators=150, learning_rate=0.08,
                                                     max_depth=4, random_state=SEED)),
    ("XGBoost", XGBClassifier(n_estimators=250, learning_rate=0.07, max_depth=5,
                              random_state=SEED, eval_metric="logloss", verbosity=0)),
    ("LightGBM", LGBMClassifier(n_estimators=250, learning_rate=0.07, max_depth=5,
                                random_state=SEED, verbose=-1)),
]:
    print(f"  → {name}")
    mdl.fit(X2_tr, yd_tr)
    n2_models[name] = mdl
    n2_probs[name]  = mdl.predict_proba(X2_te)[:, 1]
    n2_preds[name]  = mdl.predict(X2_te)

for name in n2_models:
    n2_metrics[name] = {
        "accuracy":  round(accuracy_score(yd_te, n2_preds[name]), 4),
        "precision": round(precision_score(yd_te, n2_preds[name], zero_division=0), 4),
        "recall":    round(recall_score(yd_te, n2_preds[name], zero_division=0), 4),
        "f1":        round(f1_score(yd_te, n2_preds[name], zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(yd_te, n2_probs[name]), 4),
    }

best_n2_name = max(n2_metrics, key=lambda k: n2_metrics[k]["roc_auc"])
print(f"\n🏆 Best N2 model: {best_n2_name}  "
      f"(ROC-AUC = {n2_metrics[best_n2_name]['roc_auc']})")

joblib.dump({"name": best_n2_name,
             "model": n2_models[best_n2_name],
             "scaler": scaler_n2,
             "features": n2_features,
             "type": "keras" if best_n2_name == "DNN" else "sklearn"},
            "models/best_n2_model.pkl")

#  Cost regressor — keep the best of Random Forest and XGBoost
from xgboost import XGBRegressor
cost_models = {
    "RandomForest": RandomForestClassifier(),  # placeholder, replaced below
}
rf_cost  = RandomForestClassifier()          # (unused)
from sklearn.ensemble import RandomForestRegressor
rf_cost  = RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1).fit(X2_tr, yc_tr)
xgb_cost = XGBRegressor(n_estimators=250, learning_rate=0.07, max_depth=5,
                        random_state=SEED, verbosity=0).fit(X2_tr, yc_tr)

rf_r2  = r2_score(yc_te, rf_cost.predict(X2_te))
xgb_r2 = r2_score(yc_te, xgb_cost.predict(X2_te))
best_cost = rf_cost if rf_r2 >= xgb_r2 else xgb_cost
print(f"💰 Best cost model: {'RandomForest' if rf_r2 >= xgb_r2 else 'XGBoost'} "
      f"(R² = {max(rf_r2, xgb_r2):.3f})")
joblib.dump(best_cost, "models/best_cost_model.pkl")

# N2 plots
mdf2 = pd.DataFrame(n2_metrics).T
fig, ax = plt.subplots(figsize=(12, 5.5))
mdf2.plot(kind="bar", ax=ax, colormap="plasma", edgecolor="black")
ax.set_title("N2 — Model Comparison (Intervention Selection)")
ax.set_ylabel("Score"); ax.set_ylim(0, 1.05)
ax.legend(loc="lower right", ncol=5, fontsize=9)
plt.xticks(rotation=20, ha="right")
plt.tight_layout(); plt.savefig("plots/model_comparison_n2.png"); plt.close()

plt.figure(figsize=(8, 6.5))
for (name, prob), c in zip(n2_probs.items(), plt.cm.tab10(np.linspace(0, 1, len(n2_models)))):
    fpr, tpr, _ = roc_curve(yd_te, prob)
    plt.plot(fpr, tpr, lw=2, color=c, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.title("N2 — ROC Comparison"); plt.xlabel("FPR"); plt.ylabel("TPR")
plt.legend(loc="lower right", fontsize=9)
plt.tight_layout(); plt.savefig("plots/roc_comparison_n2.png"); plt.close()

# ==================================================================
# 6.  SAVE COMPARISON REPORT
# ==================================================================
# Save the final results so the Flask app can load the selected models and
# present the winning architecture to the user interface.
report = {
    "N1_models": n1_metrics,
    "N1_best": best_n1_name,
    "N2_models": n2_metrics,
    "N2_best": best_n2_name,
    "cost_model_r2": {"RandomForest": round(rf_r2, 4),
                      "XGBoost": round(xgb_r2, 4)},
    "GANN_best_config": gann_cfg
}
with open("models/model_comparison.json", "w") as f:
    json.dump(report, f, indent=2)

print("\n" + "="*65)
print("📊 FINAL RESULTS")
print("="*65)
print(f"\nN1 — Best: {best_n1_name}")
print(pd.DataFrame(n1_metrics).T.sort_values("roc_auc", ascending=False))
print(f"\nN2 — Best: {best_n2_name}")
print(pd.DataFrame(n2_metrics).T.sort_values("roc_auc", ascending=False))
print("\n✅ Comparison complete. Best models saved.")