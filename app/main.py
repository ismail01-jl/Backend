from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import re
import os
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import FrenchStemmer
from nltk.stem import WordNetLemmatizer
from scipy.sparse import hstack

from schemas.recycling import RecyclingLot
from schemas.nlp import TextInput
from middleware.cors import setup_cors
from services.prediction_service import enc_source , le ,scaler, classifier , regressor, kmeans, pca, tfidf, nlp_clf,nlp_model , multimodal ,scaler_mm, tfidf_mm


app = FastAPI(title="EcoSmartX API")
setup_cors(app)

NLTK_DATA_PATH = "/tmp/nltk_data"
os.makedirs(NLTK_DATA_PATH, exist_ok=True)
nltk.data.path.append(NLTK_DATA_PATH)

for resource in ['punkt', 'stopwords', 'wordnet', 'punkt_tab']:
    nltk.download(resource, download_dir=NLTK_DATA_PATH, quiet=True)

# ── Load data ──
df_train = pd.read_csv("./data/train.csv")
df_test  = pd.read_csv("./data/test.csv")
df_full  = pd.concat([df_train, df_test], ignore_index=True)
df_pca = pd.read_csv("./data/pca_clusters.csv")

def preprocess(lot: RecyclingLot):
    source_enc = enc_source.transform(pd.DataFrame([[lot.Source]], columns=["Source"]))[0][0]
    nums = scaler.transform([[lot.Poids, lot.Volume,
                              lot.Conductivite, lot.Opacite,
                              lot.Rigidite]])
    return np.append(nums[0], source_enc).reshape(1, -1)

FEATURES     = ["Poids", "Volume", "Conductivite", "Opacite", "Rigidite", "Source_enc"]
LABEL_NAMES  = ['Métal', 'Papier', 'Plastique', 'Verre']
CLUSTER_NAMES = {
    0: "Groupe Métal",
    1: "Groupe Mixte",
    2: "Groupe Verre",
    3: "Groupe Minoritaire"
}

# ── Module 4 NLP preprocessing ──
STOPWORDS_FR = set(stopwords.words('french'))
STOPWORDS_DOMAINE = {
    'collecte', 'rapport', 'materiau', 'matériau',
    'dechet', 'déchet', 'lot', 'site', 'usine',
    'provenance', 'source', 'type', 'objet',
    'collecté', 'collectés', 'collectée',
    'issu', 'issus', 'présente', 'présent',
    'environ', 'estimation', 'estimé', 'estimée'
}
STOPWORDS_NLP  = STOPWORDS_FR | STOPWORDS_DOMAINE
lemmatizer     = WordNetLemmatizer()

def preprocess_text_nlp(texte: str) -> str:
    if not isinstance(texte, str) or texte.strip() == '':
        return ''
    texte = texte.lower()
    texte = re.sub(r'(\d+[\.,]?\d*)\s*(kg|g|cm|mm|m²|l|ml|%)', r'\1\2', texte)
    texte = re.sub(r'[^\w\s]', ' ', texte)
    tokens = word_tokenize(texte, language='french')
    tokens = [t for t in tokens if t not in STOPWORDS_NLP and (len(t) > 2 or t.isdigit())]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

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
        "nlp": {
            "type":       type(nlp_clf).__name__,
            "vectorizer": "TF-IDF (500 features)",
            "accuracy":   1.0
        },
        "multimodal": {
            "type":     "hstack + LinearSVC",
            "accuracy": 1.0
        },
        "clustering": {
            "type":              "K-Means",
            "k":                 4,
            "silhouette_score":  0.5049,
            "ari":               0.5468
        },
        "features": FEATURES
    }

@app.get("/model/feature-importance")
def get_feature_importance():
    importances = classifier.feature_importances_.tolist()
    return {"feature_importance": dict(zip(FEATURES, importances))}

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

# ── Module 3 : Clustering GET ──
@app.get("/clustering/stats")
def get_clustering_stats():
    counts = df_pca["Cluster"].value_counts().sort_index()
    return {
        "k":                4,
        "silhouette_score": 0.5049,
        "ari":              0.5468,
        "nmi":              0.6402,
        "variance_pca":     52.8,
        "cluster_counts": {
            CLUSTER_NAMES[i]: int(counts[i])
            for i in range(4)
        }
    }

@app.get("/clustering/pca-data")
def get_pca_data():
    # Return sample of 500 points for frontend visualization
    sample = df_pca.sample(n=min(500, len(df_pca)), random_state=42)
    return {
        "points": sample[["PC1", "PC2", "Cluster", "Categorie"]].to_dict(orient="records")
    }

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
# ── Module 3 : Clustering prediction ──
@app.post("/predict/cluster")
def predict_cluster(lot: RecyclingLot):
    source_enc = enc_source.transform([[lot.Source]])[0][0]
    # Use same features as clustering training
    features = np.array([[
        lot.Poids,
        lot.Volume,
        lot.Conductivite,
        lot.Opacite,
        source_enc
    ]])

    cluster_id  = int(kmeans.predict(features)[0])
    pca_coords  = pca.transform(features)[0]

    return {
        "cluster_id":    cluster_id,
        "cluster_name":  CLUSTER_NAMES[cluster_id],
        "pca_coords":    {"PC1": round(float(pca_coords[0]), 4),
                          "PC2": round(float(pca_coords[1]), 4)},
    }
@app.post("/predict/nlp")
def predict_nlp(data: TextInput):
    try:
        original_text = data.rapport_collecte
        cleaned = preprocess_text_nlp(original_text)

        if cleaned.strip() == "":
            raise HTTPException(status_code=400, detail="Empty or invalid text provided")

        categorie = nlp_model.predict([cleaned])[0]
        return {
            "texte_original": original_text,
            "texte_nettoye": cleaned,
            "categorie_predite": categorie
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"NLP prediction error: {str(e)}")