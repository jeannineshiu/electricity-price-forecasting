from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch


# Mock the model store so tests don't need MLflow
def get_mock_app():
    with patch("api.model.model_store") as mock_store:
        mock_store.model = MagicMock()
        mock_store.model_name = "lightgbm-electricity-price"
        mock_store.model_version = "1"
        mock_store.model_stage = "production"
        mock_store.predict.return_value = 43.8

        from api.main import app
        return TestClient(app)


def test_health():
    with patch("api.main.model_store") as mock_store:
        mock_store.model = MagicMock()
        mock_store.model_name = "lightgbm-electricity-price"
        mock_store.model_version = "1"
        mock_store.model_stage = "production"
        mock_store.model_alias = "production"

        from api.main import app
        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["status"] == "ok"


def test_predict():
    with patch("api.main.model_store") as mock_store:
        mock_store.model = MagicMock()
        mock_store.model_name = "lightgbm-electricity-price"
        mock_store.model_version = "1"
        mock_store.predict.return_value = 43.8

        from api.main import app
        client = TestClient(app)

        payload = {
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
            "is_holiday": 0,
        }
        response = client.post("/predict", json=payload)

        assert response.status_code == 200
        assert "predicted_price_eur_mwh" in response.json()


def test_predict_invalid_input():
    with patch("api.main.model_store") as mock_store:
        mock_store.model = MagicMock()

        from api.main import app
        client = TestClient(app)

        # hour out of range
        payload = {"hour": 99, "day_of_week": 1, "month": 6}
        response = client.post("/predict", json=payload)

        assert response.status_code == 422