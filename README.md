# Electricity Price Forecasting

An end-to-end MLOps project that forecasts German day-ahead electricity prices (EUR/MWh) using LightGBM and LSTM — and translates those forecasts into a **personal electricity bill simulator** that shows real households how much they could save by switching to a dynamic tariff.

---

## What Makes This Project Unique

Most price-forecasting projects stop at the model. This one closes the loop: the forecast feeds directly into a **consumer-facing simulation** that answers the question people actually care about — *"How much will my electricity bill be, and should I switch to a dynamic tariff?"*

### Bill Simulator (Streamlit)

**Try it:** `python -m streamlit run app.py`

![Bill Simulator](docs/screenshots/simulator.png)

Choose your consumption profile, set your monthly usage, and pick a month — the app computes your expected bill under both fixed and dynamic pricing using real ENTSO-E wholesale data.

| Profile | Who it models | Typical dynamic saving |
|---|---|---|
| **Office worker (evenings)** | Away 9–5, peak usage 18–23h | High — avoids daytime price peaks |
| **WFH (steady daytime)** | Home all day, flat usage | Moderate — exposed to midday prices |
| **Industrial (daytime peak)** | Heavy Mon–Fri shift operation | Low or negative — peak overlaps with peak prices |

**What the simulator shows:**
- Monthly bill under **fixed tariff** (€0.30/kWh, German 2024 average)
- Monthly bill under **dynamic tariff** (wholesale spot + €0.18/kWh grid/tax markup)
- **Confidence interval** on the dynamic bill, propagated from the LightGBM MAE of 7.23 EUR/MWh
- Hourly cost breakdown chart — see exactly which hours drive your bill
- Cheapest and most expensive hours of the day to shift flexible loads (EV charging, dishwasher, laundry)

**Key insight from the simulation:**

> An office worker consuming 300 kWh/month in a low-price month can save ~€5–15 vs. the flat tariff. In high-price months (e.g., cold January with low wind), dynamic pricing can cost €10–20 *more*. The forecast quantifies this risk — the confidence interval tells you how uncertain the bill estimate is before committing.

---

## Business Value

Electricity prices in the European market are highly volatile — driven by renewable generation, cross-border flows, and demand spikes. This project addresses two distinct audiences:

**For energy market professionals:**
- Energy traders optimise buy/sell decisions in day-ahead markets
- Grid operators anticipate demand-supply imbalances before they occur
- Renewable asset owners schedule battery dispatch to maximise revenue

**For end consumers (the simulator layer):**
- Households on dynamic tariffs (e.g., Tibber, aWATTar) can forecast their bill before the month ends
- Industrial consumers identify which hours to shift flexible loads
- Anyone evaluating a tariff switch can compare fixed vs. dynamic risk across multiple historical months

This project targets the German bidding zone (DE_LU), one of the largest and most liquid electricity markets in Europe, using 5 years of hourly price data (2020–2024) plus live ENTSO-E data for 2026.

---

## Architecture

![Architecture](docs/architecture.svg)

---

## Observability Dashboard

![Grafana Dashboard](docs/screenshots/grafana-dashboard.png)

---

## Model Performance

| Model | Test MAE | Test RMSE | Test MAPE | Status |
|---|---|---|---|---|
| **LightGBM** | **7.23** | **13.75** | **37.79%** | ✅ Production |
| XGBoost | 7.36 | 14.03 | 39.78% | Archived |
| LSTM | 10.47 | 16.98 | 68.97% | — |

LightGBM selected for production. LSTM underperforms because lag and rolling features already encode temporal structure — a deliberate engineering decision documented in the experiment tracking.

The MAE of 7.23 EUR/MWh translates to a **bill uncertainty of ±€2–5/month** for a typical 300 kWh household, small enough to make the simulation useful in practice.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Data ingestion** | ENTSO-E API (`entsoe-py`), CSV fallback |
| **Orchestration** | Prefect 3 |
| **Feature engineering** | Pandas — lag features, rolling stats, calendar features |
| **ML models** | LightGBM, LSTM (Keras), XGBoost baseline |
| **Hyperparameter tuning** | Optuna |
| **Experiment tracking** | MLflow (SQLite backend) |
| **Model registry** | MLflow Model Registry (alias-based promotion) |
| **API serving** | FastAPI + Uvicorn |
| **Consumer simulator** | Streamlit — 3 consumption profiles, fixed vs. dynamic tariff |
| **Containerisation** | Docker (multi-stage build) |
| **Orchestration (infra)** | Kubernetes (minikube), HPA |
| **CI/CD** | GitHub Actions — lint, test, build, push to GHCR |
| **Observability** | Prometheus + Grafana |
| **Testing** | pytest + httpx |
| **Linting** | Ruff |

---

## Models

| Model | Description |
|---|---|
| **XGBoost** | Baseline — tabular features, Optuna-tuned |
| **LSTM** | Sequence model — 24-hour lookback window, dual-layer, Optuna-tuned |
| **LightGBM** | Production model — tabular features, Optuna-tuned, promoted to `@production` alias |

**Features (12 total):** `lag_1h`, `lag_24h`, `lag_168h`, `rolling_mean_24h`, `rolling_std_24h`, `rolling_mean_168h`, `rolling_std_168h`, `hour`, `day_of_week`, `month`, `is_weekend`, `is_holiday`

**Train/Val/Test split:** 2020–2022 / 2023 / 2024

---

## Project Structure

```
.
├── app.py                  # Streamlit bill simulator (consumer-facing)
├── simulation/
│   ├── consumption.py      # Hourly consumption profiles (office/WFH/industrial)
│   └── tariff.py           # Fixed vs dynamic tariff logic + bill uncertainty
├── api/                    # FastAPI application
│   ├── main.py             # App entrypoint, lifespan, endpoints
│   ├── model.py            # MLflow model loader (singleton)
│   └── schemas.py          # Pydantic request/response schemas
├── flows/                  # Prefect flows
│   ├── ingest.py           # Data ingestion (ENTSO-E API + CSV fallback)
│   └── features.py         # Feature engineering
├── training/
│   ├── train.py            # XGBoost / LightGBM / LSTM training + MLflow logging
│   └── evaluate.py         # Model evaluation
├── infra/
│   ├── k8s/                # Kubernetes manifests (Deployment, Service, HPA)
│   ├── prometheus/         # Prometheus scrape config
│   ├── grafana/            # Grafana dashboard JSON
│   └── docker-compose.monitoring.yml
├── scripts/
│   └── simulate_traffic.py # Traffic simulation for dashboard demo
├── tests/
│   └── test_api.py         # API unit tests (mocked MLflow)
├── Dockerfile              # Multi-stage build
├── requirements.txt        # Training dependencies
└── requirements-api.txt    # API-only dependencies
```

---

## Quick Start

### 1. Install dependencies

```bash
conda create -n elec-forecast python=3.11
conda activate elec-forecast
python -m pip install -r requirements.txt
```

### 2. Run the bill simulator (no model training required)

```bash
python -m streamlit run app.py
```

The simulator uses historical ENTSO-E price data directly — no model needed. Open http://localhost:8501, pick a month and consumption profile, and explore your bill.

### 3. Run ingestion flow

```bash
python flows/ingest.py
```

### 4. Run feature engineering

```bash
python flows/features.py
```

### 5. Train models

```bash
# Train all models
python training/train.py

# Train LightGBM only
python training/train.py --lgbm-only

# Train LSTM only
python training/train.py --lstm-only
```

### 6. Promote model to production

```python
from mlflow import MlflowClient
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()
client.set_registered_model_alias(
    name="lightgbm-electricity-price",
    alias="production",
    version="1",
)
```

### 7. Start MLflow UI

```bash
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### 8. Start API

```bash
python -m uvicorn api.main:app --reload
```

### 9. Run tests

```bash
python -m pytest tests/ -v
```

---

## API Reference

### `GET /health`

```json
{
  "status": "ok",
  "model_name": "lightgbm-electricity-price",
  "model_version": "1",
  "model_alias": "production"
}
```

### `POST /predict`

**Request:**
```json
{
  "lag_1h": 45.2,
  "lag_24h": 38.7,
  "lag_168h": 41.1,
  "rolling_mean_24h": 42.3,
  "rolling_std_24h": 8.1,
  "rolling_mean_168h": 40.5,
  "rolling_std_168h": 9.2,
  "hour": 14,
  "day_of_week": 1,
  "month": 6,
  "is_weekend": 0,
  "is_holiday": 0
}
```

**Response:**
```json
{
  "predicted_price_eur_mwh": 43.8,
  "model_name": "lightgbm-electricity-price",
  "model_version": "1"
}
```

Interactive docs: `http://127.0.0.1:8000/docs`

---

## Docker

```bash
docker build -t elec-price-api .
docker run -p 8000:8000 \
  -v $(pwd)/mlflow.db:/app/mlflow.db \
  -v $(pwd)/mlruns:/app/mlruns \
  elec-price-api
```

---

## Kubernetes (minikube)

```bash
minikube start
minikube image load elec-price-api:latest
minikube mount $(pwd):/mnt/elec-forecast &

kubectl apply -f infra/k8s/
minikube service elec-forecast-api --url
```

---

## Observability

```bash
# Start Prometheus + Grafana
docker compose -f infra/docker-compose.monitoring.yml up -d

# Simulate realistic traffic
python scripts/simulate_traffic.py
```

| Endpoint | URL |
|---|---|
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin / admin) |
| API metrics | http://localhost:8000/metrics |

**Dashboard panels:** Requests/sec · Latency P99 · Error rate · Total predict requests

---

## CI/CD

GitHub Actions pipeline (`.github/workflows/ci-cd.yml`):

| Job | Trigger | Steps |
|---|---|---|
| **Lint & Test** | push / PR to main | Ruff lint → pytest |
| **Build & Push** | push to main | Build Docker image → push to GHCR |
| **Deploy** | push to main + `DEPLOY_ENABLED=true` | `kubectl set image` → rollout |

---

## Data Source

- **Primary:** [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) — Day-ahead prices, DE_LU bidding zone
- **Fallback:** Ember Climate hourly price CSV (2020–2024)
