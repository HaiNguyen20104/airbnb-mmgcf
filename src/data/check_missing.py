from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "bangkok"

LISTINGS_PATH = DATA_DIR / "listings.csv.gz"
REVIEWS_PATH = DATA_DIR / "reviews.csv.gz"


LISTING_COLUMNS_TO_CHECK = [
    "id",
    "name",
    "description",
    "picture_url",
    "price",
    "latitude",
    "longitude",
]

REVIEW_COLUMNS_TO_CHECK = [
    "listing_id",
    "reviewer_id",
]


def print_missing_statistics(df, columns, dataset_name):
    print(f"\n=== {dataset_name} MISSING VALUES ===")

    total_rows = len(df)

    for column in columns:
        missing_count = df[column].isna().sum()
        missing_percentage = (missing_count / total_rows) * 100

        print(
            f"{column:20} "
            f"missing = {missing_count:8} "
            f"({missing_percentage:.2f}%)"
        )


def main():
    print("=== CHECKING MISSING VALUES ===")

    listings = pd.read_csv(
        LISTINGS_PATH,
        low_memory=False
    )

    reviews = pd.read_csv(
        REVIEWS_PATH,
        low_memory=False
    )

    print_missing_statistics(
        listings,
        LISTING_COLUMNS_TO_CHECK,
        "LISTINGS"
    )

    print_missing_statistics(
        reviews,
        REVIEW_COLUMNS_TO_CHECK,
        "REVIEWS"
    )


if __name__ == "__main__":
    main()