import joblib

#Load models :
classifier = joblib.load("models/best_classifier.pkl")
regressor  = joblib.load("models/best_regressor.pkl")
scaler     = joblib.load("models/scaler.pkl")
enc_source = joblib.load("models/encoder_source.pkl")
le         = joblib.load("models/label_encoder.pkl")