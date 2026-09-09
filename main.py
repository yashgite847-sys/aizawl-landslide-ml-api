from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os


# ============================================================
# CREATE API
# ============================================================

app = FastAPI(
    title="Aizawl Landslide Risk Prediction API",
    description="AI-based landslide risk prediction using XGBoost",
    version="1.0"
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

MODEL_FILE = "Aizawl_Landslide_XGBoost_Model.pkl"

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(
        f"Model file '{MODEL_FILE}' was not found."
    )

model = joblib.load(MODEL_FILE)


# ============================================================
# MODEL FEATURES
# ============================================================

FEATURES = [
    "rainfall_mm",
    "rain_sum_3d",
    "rain_sum_7d",
    "rain_sum_15d",
    "rain_sum_30d",
    "specific_humidity_kg_kg",
    "soil_moisture_0_10cm",
    "soil_moisture_10_40cm",
    "soil_moisture_0_10cm_chg_3d",
    "soil_moisture_10_40cm_chg_3d",
    "soil_saturation_ratio",
    "temperature_c",
    "month",
    "day_of_year",
    "is_monsoon"
]


# ============================================================
# INPUT DATA
# ============================================================

class SensorData(BaseModel):

    rainfall_mm: float
    rain_sum_3d: float
    rain_sum_7d: float
    rain_sum_15d: float
    rain_sum_30d: float

    specific_humidity_kg_kg: float

    soil_moisture_0_10cm: float
    soil_moisture_10_40cm: float

    soil_moisture_0_10cm_chg_3d: float
    soil_moisture_10_40cm_chg_3d: float

    soil_saturation_ratio: float

    temperature_c: float

    month: int
    day_of_year: int
    is_monsoon: int


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {
        "status": "online",
        "message": "Aizawl Landslide Risk Prediction API",
        "model": "XGBoost",
        "features": len(FEATURES)
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True
    }


# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict")
def predict(data: SensorData):

    # Convert Pydantic input to dictionary
    input_data = data.model_dump()

    # Create dataframe in EXACT feature order
    input_df = pd.DataFrame(
        [input_data],
        columns=FEATURES
    )

    # Get model probability
    probability = model.predict_proba(input_df)[0][1]

    # IMPORTANT:
    # Convert NumPy value to normal Python float
    probability = float(probability)

    # Convert to percentage
    probability_percent = probability * 100.0

    # Binary prediction
    prediction = int(probability >= 0.50)

    # Risk level
    if probability < 0.30:
        risk = "LOW"

    elif probability < 0.60:
        risk = "MODERATE"

    else:
        risk = "HIGH"

    # Return JSON-safe values
    return {
        "landslide_probability": round(
            probability_percent,
            2
        ),
        "prediction": prediction,
        "risk": risk,
        "model": "XGBoost"
    }