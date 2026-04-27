from fastapi import FastAPI
import numpy as np
import joblib
from pydantic import BaseModel


app = FastAPI()

model = joblib.load("models/EcoSmart_Model.pkl")

@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI 🚀"}

class Features(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(data: Features):
    X = np.array(data.features).reshape(1, -1)
    prediction = model.predict(X)
    return {"prediction": int(prediction[0])}
