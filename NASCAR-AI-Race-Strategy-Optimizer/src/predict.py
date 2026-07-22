import joblib
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models"

def load_model():

    model = joblib.load(
        MODEL_PATH / "best_model.pkl"
    )

    return model

def load_preprocessor():

    preprocessor = joblib.load(
        MODEL_PATH / "preprocessor.pkl"
    )

    return preprocessor

def predict_finish(input_df):

    model = load_model()

    preprocessor = load_preprocessor()

    processed = preprocessor.transform(input_df)

    prediction = model.predict(processed)

    return prediction[0]

if __name__ == "__main__":

    sample = pd.DataFrame({

        "Season":[2025],
        "Race":[5],
        "Track":["Daytona"],
        "Length":[2.5],
        "Surface":["Asphalt"],
        "Start":[8],
        "Car":[24],
        "Driver":["William Byron"],
        "Make":["Chevrolet"],
        "Team":["Hendrick Motorsports"],
        "Series":["Cup"],
        "Laps":[200],

        "Driver_Experience":[350],
        "Driver_Historical_Avg_Finish":[11.2],
        "Driver_Last5_Avg_Finish":[8.4],
        "Driver_Historical_Avg_Rating":[92.3],
        "Team_Historical_Avg_Finish":[10.1],
        "Make_Historical_Avg_Finish":[11.4],
        "Track_Historical_Avg_Finish":[12.7],

        "Track_Type":["Superspeedway"],
        "Start_Category":["Mid"]

    })

    prediction = predict_finish(sample)

    print(f"\nPredicted Finish Position: {prediction:.2f}")

    