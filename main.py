from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os


# ============================================================
# CREATE API
# ============================================================

app = FastAPI(
    title="GeoX Landslide Risk Prediction API",
    description="AI-based landslide risk prediction using the trained ML model",
    version="2.0"
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
    "soil_moisture_0_10cm",
    "temperature_c",
    "specific_humidity_kg_kg",
    "month",
    "day_of_year",
    "is_monsoon"
]


# ============================================================
# MODEL THRESHOLD
# ============================================================

MODEL_THRESHOLD = 0.42


# ============================================================
# INPUT DATA
# ============================================================

class SensorData(BaseModel):

    rainfall_mm: float

    soil_moisture_0_10cm: float

    temperature_c: float

    specific_humidity_kg_kg: float

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
        "message": "GeoX Landslide Risk Prediction API",
        "model": "XGBoost",
        "model_version": "2.0",
        "features": len(FEATURES),
        "threshold": MODEL_THRESHOLD
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True,
        "model_version": "2.0"
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

    # Convert NumPy value to normal Python float
    probability = float(probability)

    # Convert to percentage
    probability_percent = probability * 100.0

    # ========================================================
    # BINARY PREDICTION
    # ========================================================
    #
    # The threshold was selected during model training:
    # threshold = 0.42
    #

    prediction = int(probability >= MODEL_THRESHOLD)


    # ========================================================
    # RISK LEVEL
    # ========================================================

    if probability < 0.30:

        risk = "LOW"

    elif probability < MODEL_THRESHOLD:

        risk = "MODERATE"

    else:

        risk = "HIGH"


    # ========================================================
    # RETURN RESULT
    # ========================================================

    return {

        "landslide_probability": round(
            probability_percent,
            2
        ),

        "prediction": prediction,

        "risk": risk,

        "model": "XGBoost",

        "model_version": "2.0",

        "threshold": MODEL_THRESHOLD

    }
