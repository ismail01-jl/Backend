from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="EcoSmartX API")

# Load models once at startup
classifier = joblib.load("models/best_classifier.pkl")
regressor  = joblib.load("models/best_regressor.pkl")
scaler     = joblib.load("models/scaler.pkl")
enc_source = joblib.load("models/encoder_source.pkl")
le         = joblib.load("models/label_encoder.pkl")

# Input schema
class RecyclingLot(BaseModel):
    Poids: float
    Volume: float
    Conductivite: float
    Opacite: float
    Rigidite: float
    Source: str  # "Usine_A", "Usine_B", "Centre_Tri", "Collecte_Citoyenne"

@app.get("/")
def root():
    return {"message": "EcoSmartX API is running"}

@app.post("/predict/categorie")
def predict_categorie(lot: RecyclingLot):
    features = preprocess(lot)
    pred_enc = classifier.predict(features)[0]
    categorie = le.inverse_transform([pred_enc])[0]
    return {"categorie_predite": categorie}

def preprocess(lot: RecyclingLot):
    # 1. Encode Source
    source_enc = enc_source.transform([[lot.Source]])[0][0]
    # 2. Scale numerical features
    nums = scaler.transform([[lot.Poids, lot.Volume,
                              lot.Conductivite, lot.Opacite,
                              lot.Rigidite]])
    # 3. Build final feature array [scaled nums + source_enc]
    return np.append(nums[0], source_enc).reshape(1, -1)