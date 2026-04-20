import logging

import mlflow
import pandas as pd
import mlflow.sklearn
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MODEL_NAME          = "lightgbm-electricity-price"
MODEL_ALIAS         = "production"

FEATURE_COLS = [
    "lag_1h", "lag_24h", "lag_168h",
    "rolling_mean_24h", "rolling_std_24h",
    "rolling_mean_168h", "rolling_std_168h",
    "hour", "day_of_week", "month",
    "is_weekend", "is_holiday",
]


class ModelStore:
    """
    Singleton that holds the loaded Production model and its metadata.
    Loaded once at API startup via lifespan.
    """

    def __init__(self):
        self.model         = None
        self.model_name    = MODEL_NAME
        self.model_version = "unknown"
        self.model_stage   = MODEL_ALIAS

    def load(self):
        """Load the Production model from MLflow registry."""
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()

        # Resolve version from alias
        mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        self.model_version = str(mv.version)
        model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

        logger.info(
            f"Loading model '{MODEL_NAME}' "
            f"v{self.model_version} (alias: {MODEL_ALIAS})"
        )
        self.model = mlflow.sklearn.load_model(model_uri)
        logger.info("Model loaded successfully")

    def predict(self, features: list[float]) -> float:
        """Run inference on a single feature vector."""
        if self.model is None:
            raise RuntimeError("Model is not loaded")
        X = pd.DataFrame([features], columns=FEATURE_COLS)
        return float(self.model.predict(X)[0])


# Module-level singleton — imported by main.py
model_store = ModelStore()