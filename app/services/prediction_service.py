import joblib

#Load models :
"""Module 1 & 2"""
classifier = joblib.load("models/best_classifier.pkl")
regressor  = joblib.load("models/best_regressor.pkl")
scaler     = joblib.load("models/scaler.pkl")
enc_source = joblib.load("models/encoder_source.pkl")
le         = joblib.load("models/label_encoder.pkl")
"""Module 3"""
kmeans      = joblib.load("./models/kmeans_model.pkl")
pca         = joblib.load("./models/pca_model.pkl")
"""Module 4"""
tfidf       = joblib.load("./models/tfidf_vectorizer.pkl")
nlp_clf     = joblib.load("./models/nlp_best_classifier.pkl")
"""Module 5"""
multimodal  = joblib.load("./models/best_multimodal_pipeline.pkl")