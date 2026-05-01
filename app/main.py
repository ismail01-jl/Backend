from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import joblib
import numpy as np
import pandas as pd

app = FastAPI(title="EcoSmartX API")

# ── Load models ──
classifier = joblib.load("models/best_classifier.pkl")
regressor  = joblib.load("models/best_regressor.pkl")
scaler     = joblib.load("models/scaler.pkl")
enc_source = joblib.load("models/encoder_source.pkl")
le         = joblib.load("models/label_encoder.pkl")

# ── Load data ──
df_train = pd.read_csv("train.csv")
df_test  = pd.read_csv("test.csv")
df_full  = pd.concat([df_train, df_test], ignore_index=True)

# ── In-memory storage ──
lots_db     = {}
lot_counter = 1

#Input schema
class RecyclingLot(BaseModel):
    Poids:        float
    Volume:       float
    Conductivite: float
    Opacite:      float
    Rigidite:     float
    Source:       str

# ── Preprocessing ──
def preprocess(lot: RecyclingLot):
    source_enc = enc_source.transform([[lot.Source]])[0][0]
    nums = scaler.transform([[lot.Poids, lot.Volume,
                              lot.Conductivite, lot.Opacite,
                              lot.Rigidite]])
    return np.append(nums[0], source_enc).reshape(1, -1)

# GET endpoints:

@app.get("/")
def root():
    return {"message": "EcoSmartX API is running"}

@app.get("/categories")
def get_categories():
    return {"categories": le.classes_.tolist()}

@app.get("/sources")
def get_sources():
    return {"sources": enc_source.categories_[0].tolist()}

@app.get("/model/info")
def get_model_info():
    return {
        "classifier": {
            "type":         type(classifier).__name__,
            "n_estimators": classifier.n_estimators,
            "max_depth":    classifier.max_depth,
            "accuracy":     0.9568
        },
        "regressor": {
            "type": type(regressor).__name__,
            "r2":   0.6926
        },
        "features": list(df_train.columns)
    }

@app.get("/stats")
def get_stats():
    return {
        "total_lots":      len(df_full),
        "category_counts": df_full["Categorie"].value_counts().to_dict(),
        "source_counts":   df_full["Source"].value_counts().to_dict(),
        "prix_moyen":      round(df_full["Prix_Revente"].mean(), 2),
        "prix_max":        round(df_full["Prix_Revente"].max(), 2),
        "prix_min":        round(df_full["Prix_Revente"].min(), 2),
        "poids_moyen":     round(df_full["Poids"].mean(), 2),
    }

@app.get("/stats/{categorie}")
def get_stats_by_category(categorie: str):
    valid = le.classes_.tolist()
    if categorie not in valid:
        raise HTTPException(
            status_code=404,
            detail=f"Categorie '{categorie}' not found. Valid: {valid}"
        )
    subset = df_full[df_full["Categorie"] == categorie]
    return {
        "categorie":   categorie,
        "total_lots":  len(subset),
        "prix_moyen":  round(subset["Prix_Revente"].mean(), 2),
        "prix_max":    round(subset["Prix_Revente"].max(), 2),
        "prix_min":    round(subset["Prix_Revente"].min(), 2),
        "poids_moyen": round(subset["Poids"].mean(), 2),
        "sources":     subset["Source"].value_counts().to_dict()
    }

@app.get("/model/feature-importance")
def get_feature_importance():
    features    = ["Poids", "Volume", "Conductivite", "Opacite", "Rigidite", "Source_enc"]
    importances = classifier.feature_importances_.tolist()
    return {"feature_importance": dict(zip(features, importances))}

@app.get("/lots")
def get_all_lots():
    if not lots_db:
        raise HTTPException(status_code=404, detail="No lots found")
    return {"total": len(lots_db), "lots": lots_db}

@app.get("/lot/{lot_id}")
def get_lot(lot_id: int):
    if lot_id not in lots_db:
        raise HTTPException(status_code=404, detail=f"Lot {lot_id} not found")
    return {"lot_id": lot_id, "data": lots_db[lot_id]}

# POST endpoints:

@app.post("/predict/categorie")
def predict_categorie(lot: RecyclingLot):
    features  = preprocess(lot)
    pred_enc  = classifier.predict(features)[0]
    categorie = le.inverse_transform([pred_enc])[0]
    return {"categorie_predite": categorie}

@app.post("/predict/prix")
def predict_prix(lot: RecyclingLot):
    features = preprocess(lot)
    prix     = regressor.predict(features)[0]
    return {"prix_revente_predit": round(float(prix), 2)}

@app.post("/predict/full")
def predict_full(lot: RecyclingLot):
    features  = preprocess(lot)
    pred_enc  = classifier.predict(features)[0]
    categorie = le.inverse_transform([pred_enc])[0]
    prix      = regressor.predict(features)[0]
    return {
        "categorie_predite":   categorie,
        "prix_revente_predit": round(float(prix), 2)
    }
