"""
preprocessing.py

Loads the merged NASCAR dataset, performs basic data cleaning,
validates data quality, and saves a cleaned dataset.

Author: Abhishek Jadhav
"""

from pathlib import Path
import pandas as pd

# --------------------------------------------------
# Project Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

INPUT_FILE = PROCESSED_DIR / "merged_dataset.parquet"
OUTPUT_FILE = PROCESSED_DIR / "cleaned_dataset.parquet"


# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

def load_data():
    """Load merged dataset."""
    print("Loading merged dataset...")

    df = pd.read_parquet(INPUT_FILE)

    print(f"Dataset Loaded Successfully")
    print(f"Shape : {df.shape}")

    return df


# --------------------------------------------------
# Data Validation
# --------------------------------------------------

def data_summary(df):
    """Print dataset summary."""

    print("\n========== DATA SUMMARY ==========")

    print("\nShape")
    print(df.shape)

    print("\nColumns")
    print(df.columns.tolist())

    print("\nData Types")
    print(df.dtypes)

    print("\nMissing Values")
    print(df.isnull().sum())

    print("\nDuplicate Rows")
    print(df.duplicated().sum())


# --------------------------------------------------
# Data Cleaning
# --------------------------------------------------

def clean_data(df):
    """Perform basic data cleaning."""

    print("\nCleaning Dataset...")

    # Remove duplicates
    before = len(df)

    df = df.drop_duplicates()

    after = len(df)

    print(f"Removed {before-after} duplicate rows")

    # Remove leading/trailing spaces from object columns
    object_cols = df.select_dtypes(include="object").columns

    for col in object_cols:
        df[col] = df[col].astype(str).str.strip()

    # Convert numeric columns if required
    numeric_columns = [
        "Season",
        "Race",
        "Length",
        "Finish",
        "Start",
        "Car",
        "Pts",
        "Laps",
        "Led",
        "S1",
        "S2",
        "S3",
        "Rating",
        "Win"
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# --------------------------------------------------
# Missing Values
# --------------------------------------------------

def handle_missing_values(df):

    print("\nHandling Missing Values...")

    numeric_cols = df.select_dtypes(include="number").columns

    categorical_cols = df.select_dtypes(include="object").columns

    # Median for numeric columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Mode for categorical columns
    for col in categorical_cols:
        if not df[col].mode().empty:
            df[col] = df[col].fillna(df[col].mode()[0])

    print("Missing values handled.")

    return df


# --------------------------------------------------
# Save Dataset
# --------------------------------------------------

def save_dataset(df):

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print("\nCleaned dataset saved successfully.")

    print(f"Location : {OUTPUT_FILE}")


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    df = load_data()

    data_summary(df)

    df = clean_data(df)

    df = handle_missing_values(df)

    print("\nFinal Shape")
    print(df.shape)

    print("\nRemaining Missing Values")
    print(df.isnull().sum().sum())

    save_dataset(df)


if __name__ == "__main__":
    main()