import pandas as pd
from pathlib import Path

# -----------------------------------------------------
# Project Paths
# -----------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "processed"

INPUT_FILE = DATA_PATH / "cleaned_dataset.parquet"
OUTPUT_FILE = DATA_PATH / "feature_engineered.parquet"


# -----------------------------------------------------
# Helper Functions
# -----------------------------------------------------

def classify_track(length):
    if length < 1:
        return "Short Track"
    elif length < 2:
        return "Intermediate"
    else:
        return "Superspeedway"


def classify_start(position):
    if position <= 5:
        return "Front"
    elif position <= 15:
        return "Mid"
    else:
        return "Back"


# -----------------------------------------------------
# Load Dataset
# -----------------------------------------------------

def load_data():

    print("=" * 60)
    print("Loading cleaned dataset...")
    print("=" * 60)

    df = pd.read_parquet(INPUT_FILE)

    print(f"Dataset Shape : {df.shape}")

    return df


# -----------------------------------------------------
# Feature Engineering
# -----------------------------------------------------

def create_features(df):

    print("\nCreating Features...\n")

    # Sort chronologically
    df = df.sort_values(
        by=["Series", "Season", "Race"]
    ).reset_index(drop=True)

    # -------------------------------------------------
    # Driver Experience
    # -------------------------------------------------

    df["Driver_Experience"] = (
        df.groupby("Driver").cumcount()
    )

    # -------------------------------------------------
    # Historical Driver Average Finish
    # -------------------------------------------------

    df["Driver_Historical_Avg_Finish"] = (
        df.groupby("Driver")["Finish"]
        .transform(lambda x: x.shift().expanding().mean())
    )

    # -------------------------------------------------
    # Driver Last 5 Average Finish
    # -------------------------------------------------

    df["Driver_Last5_Avg_Finish"] = (
        df.groupby("Driver")["Finish"]
        .transform(
            lambda x: x.shift().rolling(
                window=5,
                min_periods=1
            ).mean()
        )
    )

    # -------------------------------------------------
    # Historical Driver Rating
    # -------------------------------------------------

    df["Driver_Historical_Avg_Rating"] = (
        df.groupby("Driver")["Rating"]
        .transform(lambda x: x.shift().expanding().mean())
    )

    # -------------------------------------------------
    # Team Historical Finish
    # -------------------------------------------------

    df["Team_Historical_Avg_Finish"] = (
        df.groupby("Team")["Finish"]
        .transform(lambda x: x.shift().expanding().mean())
    )

    # -------------------------------------------------
    # Manufacturer Historical Finish
    # -------------------------------------------------

    df["Make_Historical_Avg_Finish"] = (
        df.groupby("Make")["Finish"]
        .transform(lambda x: x.shift().expanding().mean())
    )

    # -------------------------------------------------
    # Track Historical Finish
    # -------------------------------------------------

    df["Track_Historical_Avg_Finish"] = (
        df.groupby("Track")["Finish"]
        .transform(lambda x: x.shift().expanding().mean())
    )

    # -------------------------------------------------
    # Track Type
    # -------------------------------------------------

    df["Track_Type"] = df["Length"].apply(classify_track)

    # -------------------------------------------------
    # Starting Position Category
    # -------------------------------------------------

    df["Start_Category"] = df["Start"].apply(classify_start)

    # -------------------------------------------------
    # Fill Missing Values Created by shift()
    # -------------------------------------------------

    historical_columns = [
        "Driver_Historical_Avg_Finish",
        "Driver_Last5_Avg_Finish",
        "Driver_Historical_Avg_Rating",
        "Team_Historical_Avg_Finish",
        "Make_Historical_Avg_Finish",
        "Track_Historical_Avg_Finish",
    ]

    for col in historical_columns:
        df[col] = df[col].fillna(df[col].median())

    # -------------------------------------------------
    # Display Features
    # -------------------------------------------------

    feature_columns = [
        "Driver_Experience",
        "Driver_Historical_Avg_Finish",
        "Driver_Last5_Avg_Finish",
        "Driver_Historical_Avg_Rating",
        "Team_Historical_Avg_Finish",
        "Make_Historical_Avg_Finish",
        "Track_Historical_Avg_Finish",
        "Track_Type",
        "Start_Category",
    ]

    print("Created Features:")
    print(feature_columns)

    print("\nSample:\n")
    print(df[feature_columns].head())

    return df


# -----------------------------------------------------
# Save Dataset
# -----------------------------------------------------

def save_data(df):

    DATA_PATH.mkdir(parents=True, exist_ok=True)

    df.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print("\nFeature engineered dataset saved successfully!")
    print(f"Location : {OUTPUT_FILE}")


# -----------------------------------------------------
# Main
# -----------------------------------------------------

def main():

    df = load_data()

    df = create_features(df)

    save_data(df)

    print("\nPipeline Completed Successfully.")


if __name__ == "__main__":
    main()