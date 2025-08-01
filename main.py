from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Predictive Maintenance API")

# Load trained model
model = joblib.load("model.pkl")

# Define input schema
class SensorData(BaseModel):
    temperature: float
    vibration: float
    pressure: float
    runtime_hours: float

@app.post("/predict")
def predict_failure(data: SensorData):
    features = np.array([[data.temperature, data.vibration, data.pressure, data.runtime_hours]])
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    return {
        "failure_prediction": bool(prediction),
        "probability_of_failure": round(probability, 4)
    }
