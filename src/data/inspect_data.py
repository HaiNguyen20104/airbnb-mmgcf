from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "bangkok"

LISTINGS_PATH = DATA_DIR / "listings.csv.gz"
REVIEWS_PATH = DATA_DIR / "reviews.csv.gz"


def main():
    print("=== LOADING INSIDE AIRBNB DATA ===")

    listings = pd.read_csv(
        LISTINGS_PATH,
        low_memory=False
    )

    reviews = pd.read_csv(
        REVIEWS_PATH,
        low_memory=False
    )

    print("\n=== LISTINGS ===")
    print("Shape:", listings.shape)
    print("Columns:")
    print(listings.columns.tolist())

    print("\nFirst 5 rows:")
    print(listings.head())

    print("\n" + "=" * 80)

    print("\n=== REVIEWS ===")
    print("Shape:", reviews.shape)
    print("Columns:")
    print(reviews.columns.tolist())

    print("\nFirst 5 rows:")
    print(reviews.head())


if __name__ == "__main__":
    main()