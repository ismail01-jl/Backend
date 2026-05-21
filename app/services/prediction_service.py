import joblib

#Load models :
"""Module 1 & 2"""
classifier = joblib.load("./models/best_classifier2.pkl")
regressor  = joblib.load("./models/best_regressor2.pkl")
scaler     = joblib.load("./models/scaler2.pkl")
enc_source = joblib.load("./models/encoder_source2.pkl")
le         = joblib.load("./models/label_encoder2.pkl")
"""Module 3"""
kmeans      = joblib.load("./models/kmeans_model.pkl")
pca         = joblib.load("./models/pca_model.pkl")
"""Module 4"""
tfidf       = joblib.load("./models/tfidf_vectorizer.pkl")
nlp_clf     = joblib.load("./models/nlp_best_classifier.pkl")
nlp_model   = joblib.load("./models/best_nlp_model_2.pkl")
"""Module 5"""
multimodal  = joblib.load("./models/best_multimodal_pipeline.pkl")
tfidf_mm    = joblib.load("./models/tfidf_vectorizer.pkl")
multimodal_mm  = joblib.load("./models/nlp_best_classifier.pkl")
scaler_mm   = joblib.load("./models/scaler.pkl")