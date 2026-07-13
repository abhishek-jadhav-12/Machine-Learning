import os
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "loan_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")


def load_model():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def predict_loan(input_df):
    model, scaler = load_model()

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)

    return prediction[0], probability[0]