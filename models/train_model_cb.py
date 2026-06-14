import numpy as np
import pandas as pd
from pathlib import Path
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE   = PROJECT_ROOT / "data_prep" / "flight_pairs.csv"
RESULTS_DIR  = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(exist_ok=True)

FEATURES = [
    "turnaround_airport",
    "dest_j",
    "month",
    "day_of_week",
    "next_dep_hour",
    "upstream_arr_delay_min",
    "scheduled_turnaround_min",
    "actual_available_turnaround_min",
    "crs_elapsed_time_j",
]

CATEGORICAL = ["turnaround_airport", "dest_j"]

TARGET = "target_dep_delay_min"

print("Loading flight pair data...")

df = pd.read_csv(INPUT_FILE, low_memory=False)
print(f"Rows loaded: {len(df)}")

df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], errors="coerce")
df = df.dropna(subset=["FL_DATE"])

for col in CATEGORICAL:
    df[col] = df[col].astype(str)

numeric_features = [f for f in FEATURES if f not in CATEGORICAL]
for col in numeric_features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").fillna(0.0)


print("Splitting data (80/20)")

df = df.sort_values("FL_DATE")

all_dates = sorted(df["FL_DATE"].dt.date.unique())
split_idx = int(len(all_dates) * 0.8)
cutoff_date = all_dates[split_idx]

train = df[df["FL_DATE"].dt.date <  cutoff_date].copy()
test  = df[df["FL_DATE"].dt.date >= cutoff_date].copy()

for col in numeric_features:
    median = train[col].median()
    train[col] = train[col].fillna(median)
    test[col]  = test[col].fillna(median)

print(f"Train rows: {len(train)}")
print(f"Test rows: {len(test)}")

X_train, y_train = train[FEATURES], train[TARGET]
X_test,  y_test  = test[FEATURES],  test[TARGET]

cat_indices = [FEATURES.index(c) for c in CATEGORICAL]


print("Evaluating baseline (predicting training mean)...")
baseline = np.full(len(test), y_train.mean())
rmse = np.sqrt(mean_squared_error(y_test, baseline))
mae  = mean_absolute_error(y_test, baseline)
print(f"Baseline RMSE: {rmse:.2f}")
print(f"Baseline MAE: {mae:.2f}")


print("Training CatBoost model...")

model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=42,
    verbose=100,
)

model.fit(
    X_train, y_train,
    cat_features=cat_indices,
    eval_set=(X_test, y_test),
    use_best_model=True,
)


print("Evaluating model...")
predictions = np.maximum(model.predict(X_test), 0)

rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae  = mean_absolute_error(y_test, predictions)
r2   = r2_score(y_test, predictions)

print(f"Model RMSE: {rmse:.2f}")
print(f"Model MAE: {mae:.2f}")
print(f"Model R2: {r2:.4f}")

metrics_rows = [{
    "target": TARGET,
    "train_mean": y_train.mean(),
    "test_mean":  y_test.mean(),
    "rmse": round(rmse, 4),
    "mae":  round(mae, 4),
    "r2":   round(r2, 4),
}]


print("Saving outputs...")

model_path = RESULTS_DIR / "catboost_model.cbm"
model.save_model(str(model_path))
print(f"Saved model to: {model_path}")

metrics_df = pd.DataFrame(metrics_rows)
metrics_path = RESULTS_DIR / "metrics.csv"
metrics_df.to_csv(metrics_path, index=False)
print(f"Saved metrics to: {metrics_path}")

importance = pd.DataFrame({
    "feature":    FEATURES,
    "importance": model.get_feature_importance(),
}).sort_values("importance", ascending=False)

importance_path = RESULTS_DIR / "feature_importance.csv"
importance.to_csv(importance_path, index=False)
print(f"Saved importances to: {importance_path}")


pred_df = test[[
    "FL_DATE", "TAIL_NUM", "carrier",
    "turnaround_airport", "dest_j",
    "upstream_arr_delay_min",
    "scheduled_turnaround_min",
    "actual_available_turnaround_min",
    "target_dep_delay_min",
]].copy()

pred_df["pred_dep_delay_min"] = predictions

predictions_path = RESULTS_DIR / "predictions.csv"
pred_df.to_csv(predictions_path, index=False)
print(f"Saved predictions to: {predictions_path}")
