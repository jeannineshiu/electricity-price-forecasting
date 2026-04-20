import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from api.model import FEATURE_COLS, model_store
from api.schemas import HealthResponse, PredictRequest, PredictResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Lifespan: load model once at startup ──────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — loading model from MLflow...")
    try:
        model_store.load()
        logger.info("Model ready")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    yield
    logger.info("Shutting down")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Electricity Price Forecasting API",
    description="Predicts German day-ahead electricity prices (EUR/MWh) using LightGBM.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health():
    """Liveness check — confirms the model is loaded and ready."""
    if model_store.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return HealthResponse(
        status="ok",
        model_name=model_store.model_name,
        model_version=model_store.model_version,
        model_alias=model_store.model_stage,
    )


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(request: PredictRequest):
    """
    Predict the next-hour electricity price.

    Accepts 12 engineered features (lags, rolling stats, calendar).
    Returns predicted price in EUR/MWh.
    """
    features = [getattr(request, col) for col in FEATURE_COLS]

    try:
        price = model_store.predict(features)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    return PredictResponse(
        predicted_price_eur_mwh=round(price, 2),
        model_name=model_store.model_name,
        model_version=model_store.model_version,
    )