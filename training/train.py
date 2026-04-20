import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import mlflow.keras
import numpy as np
import optuna
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense, Dropout, Input
from keras.models import Sequential
from keras.optimizers import Adam
from prefect import flow, task
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import lightgbm as lgb
from xgboost import XGBRegressor

PROCESSED_DIR = Path("data/processed")
MODEL_DIR     = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME     = "electricity-price-forecasting"

# Train: 2020-2022 | Val: 2023 | Test: 2024
TRAIN_END = "2022-12-31 23:00:00+00:00"
VAL_END   = "2023-12-31 23:00:00+00:00"

FEATURE_COLS = [
    "lag_1h", "lag_24h", "lag_168h",
    "rolling_mean_24h", "rolling_std_24h",
    "rolling_mean_168h", "rolling_std_168h",
    "hour", "day_of_week", "month",
    "is_weekend", "is_holiday",
]
TARGET_COL = "price_eur_mwh"


# ── Helpers ────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "mape": round(mape, 4)}


def split_data(df: pd.DataFrame):
    train = df[df["timestamp"] <= TRAIN_END]
    val   = df[(df["timestamp"] > TRAIN_END) & (df["timestamp"] <= VAL_END)]
    test  = df[df["timestamp"] > VAL_END]
    return train, val, test


# ── Tasks ──────────────────────────────────────────────────────────────────────

@task(name="load-features")
def load_features() -> pd.DataFrame:
    path = PROCESSED_DIR / "features_2020_2024.parquet"
    df = pd.read_parquet(path)
    print(f"Loaded {len(df)} rows from {path}")
    return df


@task(name="train-xgboost")
def train_xgboost(df: pd.DataFrame) -> dict:
    """
    Train XGBoost baseline with Optuna hyperparameter tuning.
    Logs params, metrics, and model artifact to MLflow.
    """
    train, val, test = split_data(df)

    X_train, y_train = train[FEATURE_COLS].values, train[TARGET_COL].values
    X_val,   y_val   = val[FEATURE_COLS].values,   val[TARGET_COL].values
    X_test,  y_test  = test[FEATURE_COLS].values,  test[TARGET_COL].values

    # ── Optuna tuning ──────────────────────────────────────────────────────────
    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 200, 800),
            "max_depth":         trial.suggest_int("max_depth", 3, 8),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
            "random_state":      42,
            "n_jobs":            -1,
        }
        model = XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return mean_absolute_error(y_val, model.predict(X_val))

    print("Starting Optuna search for XGBoost (30 trials)...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30, show_progress_bar=False)
    best_params = study.best_params
    best_params.update({"random_state": 42, "n_jobs": -1})
    print(f"Best params: {best_params}")

    # ── Train final model with best params ─────────────────────────────────────
    final_model = XGBRegressor(**best_params)
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    val_metrics  = compute_metrics(y_val,  final_model.predict(X_val))
    test_metrics = compute_metrics(y_test, final_model.predict(X_test))
    print(f"XGBoost — val:  {val_metrics}")
    print(f"XGBoost — test: {test_metrics}")

    # ── Log to MLflow ──────────────────────────────────────────────────────────
    with mlflow.start_run(run_name="xgboost-baseline"):
        mlflow.log_params(best_params)
        mlflow.log_metrics({f"val_{k}":  v for k, v in val_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.sklearn.log_model(
            final_model,
            artifact_path="model",
            registered_model_name="xgboost-electricity-price",
        )
        run_id = mlflow.active_run().info.run_id

    return {"model": "xgboost", "run_id": run_id, **test_metrics}


@task(name="train-lstm")
def train_lstm(df: pd.DataFrame) -> dict:
    """
    Train LSTM with sequence input (lookback=24 hours).

    Each sample contains the past 24 hours of all features,
    giving the LSTM real temporal context to learn from.
    Target is the price at hour t (next step after the window).
    """
    train, val, test = split_data(df)

    LOOKBACK = 24  # use past 24 hours as input sequence

    # ── Scale ──────────────────────────────────────────────────────────────────
    feature_scaler = MinMaxScaler()
    target_scaler  = MinMaxScaler()

    X_train_s = feature_scaler.fit_transform(train[FEATURE_COLS].values)
    X_val_s   = feature_scaler.transform(val[FEATURE_COLS].values)
    X_test_s  = feature_scaler.transform(test[FEATURE_COLS].values)

    y_train_s = target_scaler.fit_transform(train[[TARGET_COL]].values).flatten()
    y_val_s   = target_scaler.transform(val[[TARGET_COL]].values).flatten()
    y_test    = test[TARGET_COL].values

    # ── Build sequences: (samples, lookback, features) ────────────────────────
    def make_sequences(X: np.ndarray, y: np.ndarray, lookback: int):
        Xs, ys = [], []
        for i in range(lookback, len(X)):
            Xs.append(X[i - lookback:i])   # shape: (lookback, n_features)
            ys.append(y[i])
        return np.array(Xs), np.array(ys)

    X_tr, y_tr = make_sequences(X_train_s, y_train_s, LOOKBACK)
    X_vl, y_vl = make_sequences(X_val_s,   y_val_s,   LOOKBACK)

    # For test sequences we need the last LOOKBACK rows of val as context
    X_test_full_s = np.concatenate([X_val_s[-LOOKBACK:], X_test_s], axis=0)
    y_test_full_s = np.concatenate([
        target_scaler.transform(val[[TARGET_COL]].values).flatten()[-LOOKBACK:],
        target_scaler.transform(test[[TARGET_COL]].values).flatten(),
    ])
    X_te, y_te = make_sequences(X_test_full_s, y_test_full_s, LOOKBACK)

    print(
        f"Sequence shapes — "
        f"train: {X_tr.shape}, val: {X_vl.shape}, test: {X_te.shape}"
    )

    # ── Optuna tuning ──────────────────────────────────────────────────────────
    def build_model(units_1, units_2, dropout_rate, learning_rate):
        model = Sequential([
            Input(shape=(LOOKBACK, len(FEATURE_COLS))),
            LSTM(units_1, return_sequences=True),
            Dropout(dropout_rate),
            LSTM(units_2, return_sequences=False),
            Dropout(dropout_rate),
            Dense(1),
        ])
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
        return model

    def objective(trial):
        units_1       = trial.suggest_int("units_1", 32, 256, step=32)
        units_2       = trial.suggest_int("units_2", 16, 128, step=16)
        dropout_rate  = trial.suggest_float("dropout_rate", 0.1, 0.4)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

        model = build_model(units_1, units_2, dropout_rate, learning_rate)
        model.fit(
            X_tr, y_tr,
            validation_data=(X_vl, y_vl),
            epochs=20,
            batch_size=64,
            callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
            verbose=0,
        )
        y_pred_s = model.predict(X_vl, verbose=0).flatten()
        y_pred   = target_scaler.inverse_transform(
            y_pred_s.reshape(-1, 1)
        ).flatten()
        y_true   = target_scaler.inverse_transform(
            y_vl.reshape(-1, 1)
        ).flatten()
        return mean_absolute_error(y_true, y_pred)

    print("Starting Optuna search for LSTM (15 trials)...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=15, show_progress_bar=False)
    best = study.best_params
    print(f"Best LSTM params: {best}")

    # ── Train final model ──────────────────────────────────────────────────────
    final_model = build_model(
        best["units_1"], best["units_2"],
        best["dropout_rate"], best["learning_rate"],
    )
    history = final_model.fit(
        X_tr, y_tr,
        validation_data=(X_vl, y_vl),
        epochs=50,
        batch_size=64,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=1,
    )

    y_val_pred  = target_scaler.inverse_transform(
        final_model.predict(X_vl, verbose=0).reshape(-1, 1)
    ).flatten()
    y_test_pred = target_scaler.inverse_transform(
        final_model.predict(X_te, verbose=0).reshape(-1, 1)
    ).flatten()
    y_val_true  = target_scaler.inverse_transform(y_vl.reshape(-1, 1)).flatten()

    val_metrics  = compute_metrics(y_val_true, y_val_pred)
    test_metrics = compute_metrics(y_test,     y_test_pred)
    print(f"LSTM — val:  {val_metrics}")
    print(f"LSTM — test: {test_metrics}")

    # ── Log to MLflow ──────────────────────────────────────────────────────────
    with mlflow.start_run(run_name="lstm-sequence24"):
        mlflow.log_params({
            **best,
            "lookback":      LOOKBACK,
            "epochs_trained": len(history.history["loss"]),
            "batch_size":    64,
        })
        mlflow.log_metrics({f"val_{k}":  v for k, v in val_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        scaler_info = {
            "feature_scaler_min":   feature_scaler.data_min_.tolist(),
            "feature_scaler_scale": feature_scaler.scale_.tolist(),
            "target_scaler_min":    target_scaler.data_min_.tolist(),
            "target_scaler_scale":  target_scaler.scale_.tolist(),
            "feature_cols":         FEATURE_COLS,
            "lookback":             LOOKBACK,
        }
        scaler_path = MODEL_DIR / "lstm_scaler_info.json"
        with open(scaler_path, "w") as f:
            json.dump(scaler_info, f, indent=2)
        mlflow.log_artifact(str(scaler_path))

        mlflow.keras.log_model(
            final_model,
            artifact_path="model",
            registered_model_name="lstm-electricity-price",
        )
        run_id = mlflow.active_run().info.run_id

    return {"model": "lstm", "run_id": run_id, **test_metrics}

@task(name="train-lightgbm")
def train_lightgbm(df: pd.DataFrame) -> dict:
    """
    Train LightGBM with Optuna hyperparameter tuning.
    Logs params, metrics, and model artifact to MLflow.
    """
    train, val, test = split_data(df)

    X_train, y_train = train[FEATURE_COLS].values, train[TARGET_COL].values
    X_val,   y_val   = val[FEATURE_COLS].values,   val[TARGET_COL].values
    X_test,  y_test  = test[FEATURE_COLS].values,  test[TARGET_COL].values

    # ── Optuna tuning ──────────────────────────────────────────────────────────
    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 200, 800),
            "max_depth":         trial.suggest_int("max_depth", 3, 8),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "num_leaves":        trial.suggest_int("num_leaves", 20, 150),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "random_state":      42,
            "n_jobs":            -1,
            "verbose":           -1,
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(10, verbose=False)])
        return mean_absolute_error(y_val, model.predict(X_val))

    print("Starting Optuna search for LightGBM (30 trials)...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30, show_progress_bar=False)
    best_params = study.best_params
    best_params.update({"random_state": 42, "n_jobs": -1, "verbose": -1})
    print(f"Best params: {best_params}")

    # ── Train final model with best params ─────────────────────────────────────
    final_model = lgb.LGBMRegressor(**best_params)
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(20, verbose=False)],
    )

    val_metrics  = compute_metrics(y_val,  final_model.predict(X_val))
    test_metrics = compute_metrics(y_test, final_model.predict(X_test))
    print(f"LightGBM — val:  {val_metrics}")
    print(f"LightGBM — test: {test_metrics}")

    # ── Log to MLflow ──────────────────────────────────────────────────────────
    with mlflow.start_run(run_name="lightgbm"):
        mlflow.log_params(best_params)
        mlflow.log_metrics({f"val_{k}":  v for k, v in val_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.sklearn.log_model(
            final_model,
            artifact_path="model",
            registered_model_name="lightgbm-electricity-price",
        )
        run_id = mlflow.active_run().info.run_id

    return {"model": "lightgbm", "run_id": run_id, **test_metrics}


# ── Flow ───────────────────────────────────────────────────────────────────────

@flow(name="train-models", log_prints=True)
def train_flow():
    """
    Model training flow.

    Steps:
        1. Load feature-engineered Parquet
        2. Train XGBoost baseline (with Optuna)
        3. Train LSTM main model  (with Optuna)
        4. Print comparison summary
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_features()

    xgb_result  = train_xgboost(df)
    lgbm_result = train_lightgbm(df)
    lstm_result = train_lstm(df)

    print("\n── Training Summary ──────────────────────────────")
    print(f"XGBoost  | MAE={xgb_result['mae']:.2f}  RMSE={xgb_result['rmse']:.2f}  MAPE={xgb_result['mape']:.2f}%")
    print(f"LightGBM | MAE={lgbm_result['mae']:.2f}  RMSE={lgbm_result['rmse']:.2f}  MAPE={lgbm_result['mape']:.2f}%")
    print(f"LSTM     | MAE={lstm_result['mae']:.2f}  RMSE={lstm_result['rmse']:.2f}  MAPE={lstm_result['mape']:.2f}%")
    print("\nMLflow run IDs:")
    print(f"  XGBoost  : {xgb_result['run_id']}")
    print(f"  LightGBM : {lgbm_result['run_id']}")
    print(f"  LSTM     : {lstm_result['run_id']}")
    print("─────────────────────────────────────────────────")


@flow(name="train-lstm-only", log_prints=True)
def train_lstm_flow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    df = load_features()
    lstm_result = train_lstm(df)
    print("\n── LSTM Result ───────────────────────────────────")
    print(f"LSTM | MAE={lstm_result['mae']:.2f}  RMSE={lstm_result['rmse']:.2f}  MAPE={lstm_result['mape']:.2f}%")
    print(f"MLflow run ID: {lstm_result['run_id']}")
    print("─────────────────────────────────────────────────")


@flow(name="train-lightgbm-only", log_prints=True)
def train_lightgbm_flow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    df = load_features()
    lgbm_result = train_lightgbm(df)
    print("\n── LightGBM Result ───────────────────────────────")
    print(f"LightGBM | MAE={lgbm_result['mae']:.2f}  RMSE={lgbm_result['rmse']:.2f}  MAPE={lgbm_result['mape']:.2f}%")
    print(f"MLflow run ID: {lgbm_result['run_id']}")
    print("─────────────────────────────────────────────────")


if __name__ == "__main__":
    import sys
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "--lstm-only":
        train_lstm_flow()
    elif arg == "--lgbm-only":
        train_lightgbm_flow()
    else:
        train_flow()