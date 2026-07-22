import pandas as pd
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data folders
RAW_DATA = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed"

#path to raw data files
cup_file = RAW_DATA / "cup_series.parquet"
nxs_file = RAW_DATA / "nxs_series.parquet"
truck_file = RAW_DATA / "truck_series.parquet"

# Load data from raw files
cup = pd.read_parquet(cup_file)
nxs = pd.read_parquet(nxs_file)
truck = pd.read_parquet(truck_file)

# Add a new column to each DataFrame to indicate the series
cup["Series"] = "Cup"
nxs["Series"] = "NXS"
truck["Series"] = "Truck"

#merge the three DataFrames into a single DataFrame
merged_df = pd.concat(
    [cup, nxs, truck],
    ignore_index=True
)

#size of the merged dataset
print("Merged Dataset Shape:")
print(merged_df.shape)

merged_df.drop_duplicates(inplace=True)
print("Shape After Removing Duplicates:")
print(merged_df.shape)

PROCESSED_DATA.mkdir(
    parents=True,
    exist_ok=True
)

merged_df.to_parquet(
    PROCESSED_DATA / "merged_dataset.parquet",
    index=False
)

print("Merged dataset saved successfully!")

def load_and_merge_data():
    """
    Load and merge the Cup, NXS, and Truck series datasets.

    Returns:
        pd.DataFrame: Merged DataFrame containing all series data.
    """
    cup = pd.read_parquet(cup_file)
    nxs = pd.read_parquet(nxs_file)
    truck = pd.read_parquet(truck_file)

    # Add a new column to each DataFrame to indicate the series
    cup["Series"] = "Cup"
    nxs["Series"] = "NXS"
    truck["Series"] = "Truck"

    # Merge the three DataFrames into a single DataFrame
    merged_df = pd.concat(
        [cup, nxs, truck],
        ignore_index=True
    )

    # Remove duplicates
    merged_df.drop_duplicates(inplace=True)

    return merged_df

if __name__ == "__main__":
    load_and_merge_data()

