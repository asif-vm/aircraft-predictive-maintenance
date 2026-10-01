from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ARTIFACT = Path(__file__).parent / "artifacts" / "model.joblib"
app = FastAPI(title="Aircraft Predictive Maintenance API", version="1.0")
bundle = joblib.load(ARTIFACT) if ARTIFACT.exists() else None


class EngineReading(BaseModel):
    cycle: int = Field(ge=1)
    setting_1: float = 0
    setting_2: float = 0
    setting_3: float = 0
    sensors: list[float] = Field(min_length=21, max_length=21)


@app.get("/health")
def health():
    return {"model_ready": bundle is not None}


@app.post("/predict")
def predict(reading: EngineReading):
    if bundle is None:
        raise HTTPException(503, "Run train.py first")
    values = [reading.cycle, reading.setting_1, reading.setting_2, reading.setting_3, *reading.sensors]
    row = pd.DataFrame([values], columns=bundle["features"])
    rul = max(0.0, min(125.0, float(bundle["model"].predict(row)[0])))
    return {"predicted_rul_cycles": round(rul, 2), "maintenance_priority": "high" if rul <= 30 else "normal"}

