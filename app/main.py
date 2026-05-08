from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
from schemas.recycling import RecyclingLot
from middleware.cors import setup_cors
from services.prediction_service import enc_source , le ,scaler, classifier , regressor


app = FastAPI(title="EcoSmartX API")
setup_cors(app)

lots_db     = {}
lot_counter = 1

# ── Load data ──
df_train = pd.read_csv("./data/train.csv")
df_test  = pd.read_csv("./data/test.csv")
df_full  = pd.concat([df_train, df_test], ignore_index=True)

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

@app.get("/stats/categories/count")
def categories_count():

    counts = df_full["Categorie"].value_counts()

    return {
        "labels": counts.index.tolist(),
        "values": counts.values.tolist()
    }

@app.get("/stats/categories/prixmoyen")
def avg_price_by_category():

    stats = (
        df_full
        .groupby("Categorie")["Prix_Revente"]
        .mean()
        .round(2)
    )

    return {
        "labels": stats.index.tolist(),
        "values": stats.values.tolist()
    }

@app.get("/stats/sources/count")
def sources_count():

    counts = df_full["Source"].value_counts()

    return {
        "labels": counts.index.tolist(),
        "values": counts.values.tolist()
    }




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

@app.post("/lot")
def create_lot(lot: RecyclingLot):
    global lot_counter
    features  = preprocess(lot)
    pred_enc  = classifier.predict(features)[0]
    categorie = le.inverse_transform([pred_enc])[0]
    prix      = regressor.predict(features)[0]

    lot_id = lot_counter
    lots_db[lot_id] = {
        **lot.dict(),
        "categorie_predite":   categorie,
        "prix_revente_predit": round(float(prix), 2),
        "timestamp":           datetime.now().isoformat()
    }
    lot_counter += 1
    return {
        "message":             "Lot created successfully",
        "lot_id":              lot_id,
        "categorie_predite":   categorie,
        "prix_revente_predit": round(float(prix), 2)
    }