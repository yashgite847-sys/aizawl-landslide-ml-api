from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import time

app = FastAPI(
    title="GeoX Landslide Risk Prediction API",
    description="AI-based landslide risk prediction using XGBoost",
    version="2.1"
)

MODEL_FILE = "Aizawl_Landslide_XGBoost_Model.pkl"

FEATURES = [
    "rainfall_mm",
    "soil_moisture_0_10cm",
    "temperature_c",
    "specific_humidity_kg_kg",
    "month",
    "day_of_year",
    "is_monsoon"
]

MODEL_THRESHOLD = 0.42

model = None
model_loaded_at = None


def load_model():
    global model, model_loaded_at

    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(
            f"Model file '{MODEL_FILE}' was not found."
        )

    model = joblib.load(MODEL_FILE)
    model_loaded_at = time.time()

    print("============================================")
    print("GeoX XGBoost model loaded successfully")
    print(f"Model file: {MODEL_FILE}")
    print(f"Features: {len(FEATURES)}")
    print(f"Threshold: {MODEL_THRESHOLD}")
    print("============================================")


@app.on_event("startup")
def startup_event():
    load_model()


@app.get("/")
def home():
    return {
        "status": "online",
        "message": "GeoX Landslide Risk Prediction API",
        "model": "XGBoost",
        "model_version": "2.1",
        "features": len(FEATURES),
        "threshold": MODEL_THRESHOLD
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_version": "2.1"
    }


class SensorData(BaseModel):
    rainfall_mm: float
    soil_moisture_0_10cm: float
    temperature_c: float
    specific_humidity_kg_kg: float
    month: int
    day_of_year: int
    is_monsoon: int


@app.post("/predict")
def predict(data: SensorData):

    if model is None:
        load_model()

    input_data = data.model_dump()

    input_df = pd.DataFrame(
        [input_data],
        columns=FEATURES
    )

    print("============================================")
    print("NEW PREDICTION REQUEST")
    print(input_data)
    print("============================================")

    probability = model.predict_proba(input_df)[0][1]

    probability = float(probability)
    probability_percent = probability * 100.0

    prediction = int(
        probability >= MODEL_THRESHOLD
    )

    if probability < 0.30:
        risk = "LOW"
    elif probability < MODEL_THRESHOLD:
        risk = "MODERATE"
    else:
        risk = "HIGH"

    result = {
        "landslide_probability": round(
            probability_percent,
            2
        ),
        "prediction": prediction,
        "risk": risk,
        "model": "XGBoost",
        "model_version": "2.1",
        "threshold": MODEL_THRESHOLD
    }

    print("PREDICTION RESULT")
    print(result)
    print("============================================")

    return result
