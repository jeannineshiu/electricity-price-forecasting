"""
Simulate realistic API traffic for Grafana dashboard demo.

Traffic pattern:
  - Normal predict requests (majority)
  - Health checks (periodic)
  - Invalid requests (occasional 422s)
  - Burst traffic every ~30s
"""
import random
import time

import httpx

BASE_URL = "http://127.0.0.1:8000"

VALID_PAYLOAD = {
    "lag_1h":            45.2,
    "lag_24h":           38.7,
    "lag_168h":          41.1,
    "rolling_mean_24h":  42.3,
    "rolling_std_24h":    8.1,
    "rolling_mean_168h": 40.5,
    "rolling_std_168h":   9.2,
    "hour":              14,
    "day_of_week":        1,
    "month":              6,
    "is_weekend":         0,
    "is_holiday":         0,
}

INVALID_PAYLOAD = {"lag_1h": 45.2}  # missing required fields → 422


def send_predict(client: httpx.Client, payload: dict):
    try:
        client.post(f"{BASE_URL}/predict", json=payload, timeout=5)
    except Exception:
        pass


def send_health(client: httpx.Client):
    try:
        client.get(f"{BASE_URL}/health", timeout=5)
    except Exception:
        pass


def varied_payload() -> dict:
    payload = VALID_PAYLOAD.copy()
    payload["lag_1h"]   = round(random.uniform(20, 120), 2)
    payload["lag_24h"]  = round(random.uniform(20, 120), 2)
    payload["hour"]     = random.randint(0, 23)
    payload["month"]    = random.randint(1, 12)
    payload["is_weekend"] = random.randint(0, 1)
    return payload


def main():
    print("Simulating traffic — Ctrl+C to stop\n")
    tick = 0

    with httpx.Client() as client:
        while True:
            tick += 1

            # Normal predict requests (2–4 per cycle)
            for _ in range(random.randint(2, 4)):
                send_predict(client, varied_payload())

            # Health check every 5 ticks
            if tick % 5 == 0:
                send_health(client)

            # Burst traffic every 30 ticks
            if tick % 30 == 0:
                print(f"[tick {tick}] Burst traffic...")
                for _ in range(15):
                    send_predict(client, varied_payload())
                    time.sleep(0.1)

            # Occasional invalid request (~10% chance)
            if random.random() < 0.1:
                send_predict(client, INVALID_PAYLOAD)

            print(f"[tick {tick}] sent requests", end="\r")
            time.sleep(2)


if __name__ == "__main__":
    main()
