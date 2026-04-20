from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    lag_1h:             float = Field(..., description="Price 1 hour ago (EUR/MWh)")
    lag_24h:            float = Field(..., description="Price 24 hours ago (EUR/MWh)")
    lag_168h:           float = Field(..., description="Price 168 hours ago (EUR/MWh)")
    rolling_mean_24h:   float = Field(..., description="Rolling mean over past 24h")
    rolling_std_24h:    float = Field(..., description="Rolling std over past 24h")
    rolling_mean_168h:  float = Field(..., description="Rolling mean over past 168h")
    rolling_std_168h:   float = Field(..., description="Rolling std over past 168h")
    hour:               int   = Field(..., ge=0, le=23)
    day_of_week:        int   = Field(..., ge=0, le=6,  description="0=Monday, 6=Sunday")
    month:              int   = Field(..., ge=1, le=12)
    is_weekend:         int   = Field(..., ge=0, le=1)
    is_holiday:         int   = Field(..., ge=0, le=1)

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    predicted_price_eur_mwh: float
    model_name:              str
    model_version:           str


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status:        str
    model_name:    str
    model_version: str
    model_alias:   str