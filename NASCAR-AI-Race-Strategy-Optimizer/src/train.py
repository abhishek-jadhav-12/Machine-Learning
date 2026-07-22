import pandas as pd
from pathlib import Path
import joblib
import numpy as np
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "processed"

MODEL_PATH = PROJECT_ROOT / "models"

REPORT_PATH = PROJECT_ROOT / "reports"

INPUT_FILE = DATA_PATH / "feature_engineered.parquet"

def load_data():

    print("Loading feature engineered dataset...")

    df = pd.read_parquet(INPUT_FILE)

    print(df.shape)

    return df

def prepare_data(df):

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

    categorical_columns = X.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    numerical_columns = X.select_dtypes(
        exclude=["object", "string"]
    ).columns.tolist()

    print("\nCategorical Columns:")
    print(categorical_columns)

    print("\nNumerical Columns:")
    print(numerical_columns)

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore")
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_pipeline,
                numerical_columns
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_columns
            )
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    )


def train_linear_regression(X_train, y_train):

    print("\nTraining Linear Regression model...")

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    print("\nEvaluation Metrics:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R^2: {r2:.4f}")


def save_model(model):

    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_PATH / "linear_regression_model.joblib"
    joblib.dump(model, model_path)

    print(f"\nSaved model to {model_path}")

def train_models(X_train, y_train):

    models = {

     "Linear Regression": LinearRegression(),

     "Decision Tree": DecisionTreeRegressor(
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=50,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        random_state=42
    ),

    "XGBoost": XGBRegressor(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    ),

    "CatBoost": CatBoostRegressor(
        iterations=50,
        learning_rate=0.1,
        depth=6,
        random_state=42,
        verbose=0
    )

}

    trained_models = {}

    for name, model in models.items():

        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)

        trained_models[name] = model

    return trained_models

def evaluate_models(models, X_test, y_test):

    results = []

    best_model = None
    best_r2 = -999

    for name, model in models.items():

        predictions = model.predict(X_test)

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )

        r2 = r2_score(
            y_test,
            predictions
        )

        results.append({

            "Model": name,

            "MAE": round(mae,3),

            "RMSE": round(rmse,3),

            "R2": round(r2,3)

        })

        print()

        print(name)

        print("MAE :",mae)

        print("RMSE:",rmse)

        print("R²  :",r2)

        if r2 > best_r2:

            best_r2 = r2

            best_model = model

    results = pd.DataFrame(results)

    return results, best_model

def save_all_models(models):

    MODEL_PATH.mkdir(exist_ok=True)

    for name, model in models.items():

        filename = (
            name.lower()
            .replace(" ","_")
            + ".pkl"
        )

        joblib.dump(
            model,
            MODEL_PATH / filename
        )

    print("\nAll models saved.")

def save_best_model(model):

    joblib.dump(

        model,

        MODEL_PATH / "best_model.pkl"

    )

    print("Best model saved.")

def save_report(results):

    REPORT_PATH.mkdir(exist_ok=True)

    results.to_csv(

        REPORT_PATH /

        "model_comparison.csv",

        index=False

    )

    print("Comparison report saved.")

def save_preprocessor(preprocessor):

    joblib.dump(
        preprocessor,
        MODEL_PATH / "preprocessor.pkl"
    )

    print("Preprocessor saved.")


def main():

    # Load feature engineered dataset
    df = load_data()

    # Prepare data
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(df)

    print("\nTraining Shape :", X_train.shape)
    print("Testing Shape  :", X_test.shape)

    # Train all models
    trained_models = train_models(
        X_train,
        y_train
    )

    # Evaluate all models
    results, best_model = evaluate_models(
        trained_models,
        X_test,
        y_test
    )

    # Display comparison table
    print("\nModel Comparison")
    print(results)

    # Save all trained models
    save_all_models(trained_models)

    # Save best model
    save_best_model(best_model)

    # Save preprocessor
    save_preprocessor(preprocessor)

    # Save comparison report
    save_report(results)

    print("\nTraining Pipeline Completed Successfully!")


if __name__ == "__main__":
    main()