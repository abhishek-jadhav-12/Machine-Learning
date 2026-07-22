import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "processed"

MODEL_PATH = PROJECT_ROOT / "models"

REPORT_PATH = PROJECT_ROOT / "reports"

def load_data():

    print("Loading dataset...")

    df = pd.read_parquet(
        DATA_PATH / "feature_engineered.parquet"
    )

    return df

def load_preprocessor():

    print("Loading preprocessor...")

    return joblib.load(
        MODEL_PATH / "preprocessor.pkl"
    )

def load_model():

    print("Loading best model...")

    return joblib.load(
        MODEL_PATH / "best_model.pkl"
    )


def prepare_data(df, preprocessor):

    target = "Finish"

    drop_columns = [
        "Finish",
        "Pts",
        "Led",
        "Win",
        "Rating",
        "S1",
        "S2",
        "S3",
        "Status",
        "Name"
    ]

    X = df.drop(columns=drop_columns)

    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    X_test = preprocessor.transform(X_test)

    return X_test, y_test

def make_predictions(model, X_test):

    print("Making predictions...")

    predictions = model.predict(X_test)

    return predictions

def evaluate_predictions(y_test, predictions):

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    mse = mean_squared_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        predictions
    )

    print("\nEvaluation Metrics")

    print(f"MAE  : {mae:.3f}")
    print(f"MSE  : {mse:.3f}")
    print(f"RMSE : {rmse:.3f}")
    print(f"R²   : {r2:.3f}")

    return mae, mse, rmse, r2

def save_metrics(mae, mse, rmse, r2):

    REPORT_PATH.mkdir(exist_ok=True)

    with open(REPORT_PATH / "metrics.txt", "w") as f:

        f.write("Model Evaluation Metrics\n")
        f.write("========================\n\n")

        f.write(f"MAE  : {mae:.4f}\n")
        f.write(f"MSE  : {mse:.4f}\n")
        f.write(f"RMSE : {rmse:.4f}\n")
        f.write(f"R2   : {r2:.4f}\n")

    print("Metrics saved.")

def plot_actual_vs_predicted(y_test, predictions):

    plt.figure(figsize=(8,6))

    plt.scatter(
        y_test,
        predictions,
        alpha=0.5
    )

    plt.xlabel("Actual Finish")

    plt.ylabel("Predicted Finish")

    plt.title("Actual vs Predicted Finish")

    min_val = min(y_test.min(), predictions.min())
    max_val = max(y_test.max(), predictions.max())

    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--"
    )

    plt.tight_layout()

    plt.savefig(
        REPORT_PATH / "actual_vs_predicted.png"
    )

    plt.close()

    print("Actual vs Predicted plot saved.")

def plot_residuals(y_test, predictions):

    residuals = y_test - predictions

    plt.figure(figsize=(8,6))

    plt.scatter(
        predictions,
        residuals,
        alpha=0.5
    )

    plt.axhline(
        y=0,
        linestyle="--"
    )

    plt.xlabel("Predicted Finish")

    plt.ylabel("Residual")

    plt.title("Residual Plot")

    plt.tight_layout()

    plt.savefig(
        REPORT_PATH / "residual_plot.png"
    )

    plt.close()

    print("Residual plot saved.")

def plot_error_distribution(y_test, predictions):

    errors = y_test - predictions

    plt.figure(figsize=(8,6))

    plt.hist(
        errors,
        bins=40
    )

    plt.xlabel("Prediction Error")

    plt.ylabel("Frequency")

    plt.title("Error Distribution")

    plt.tight_layout()

    plt.savefig(
        REPORT_PATH / "error_distribution.png"
    )

    plt.close()

    print("Error distribution saved.")

def plot_feature_importance(model, preprocessor):

    # Only tree-based models expose feature_importances_
    if not hasattr(model, "feature_importances_"):
        print("Feature importance not available for this model.")
        return

    feature_names = preprocessor.get_feature_names_out()

    importance = model.feature_importances_

    feature_importance = pd.DataFrame({

        "Feature": feature_names,

        "Importance": importance

    })

    feature_importance = feature_importance.sort_values(
        by="Importance",
        ascending=False
    )

    top_features = feature_importance.head(20)

    plt.figure(figsize=(10,8))

    plt.barh(
        top_features["Feature"],
        top_features["Importance"]
    )

    plt.xlabel("Importance")

    plt.title("Top 20 Feature Importance")

    plt.gca().invert_yaxis()

    plt.tight_layout()

    plt.savefig(
        REPORT_PATH / "feature_importance.png"
    )

    plt.close()

    feature_importance.to_csv(
        REPORT_PATH / "feature_importance.csv",
        index=False
    )

    print("Feature importance saved.")

def main():

    df = load_data()

    preprocessor = load_preprocessor()

    model = load_model()

    X_test, y_test = prepare_data(
        df,
        preprocessor
    )

    predictions = make_predictions(
        model,
        X_test
    )

    evaluate_predictions(
        y_test,
        predictions
    )
 
    mae, mse, rmse, r2 = evaluate_predictions(
    y_test,
    predictions
    )

    save_metrics(
    mae,
    mse,
    rmse,
    r2
    )

    plot_actual_vs_predicted(
    y_test,
    predictions
    )

    plot_residuals(
    y_test,
    predictions
    )

    plot_error_distribution(
    y_test,
    predictions
    )

    plot_feature_importance(
    model,
    preprocessor
    )

if __name__ == "__main__":
    main()  