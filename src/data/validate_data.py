from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "bangkok"

LISTINGS_PATH = DATA_DIR / "listings.csv.gz"
REVIEWS_PATH = DATA_DIR / "reviews.csv.gz"


LISTING_REQUIRED_COLUMNS = {
    "id",
    "name",
    "description",
    "picture_url",
    "price",
    "latitude",
    "longitude",
}

REVIEW_REQUIRED_COLUMNS = {
    "listing_id",
    "reviewer_id",
}


def check_required_columns(df, required_columns, name):
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        print(f"[ERROR] {name} thiếu các cột:")
        print(missing_columns)
    else:
        print(f"[OK] {name} có đầy đủ các cột bắt buộc.")


def main():
    print("=== VALIDATING INSIDE AIRBNB DATA ===")

    listings = pd.read_csv(
        LISTINGS_PATH,
        low_memory=False
    )

    reviews = pd.read_csv(
        REVIEWS_PATH,
        low_memory=False
    )

    print("\n=== REQUIRED COLUMNS ===")

    check_required_columns(
        listings,
        LISTING_REQUIRED_COLUMNS,
        "Listings"
    )

    check_required_columns(
        reviews,
        REVIEW_REQUIRED_COLUMNS,
        "Reviews"
    )

    print("\n=== BASIC STATISTICS ===")

    print(
        "Number of unique listings:",
        listings["id"].nunique()
    )

    print(
        "Number of unique reviewers:",
        reviews["reviewer_id"].nunique()
    )

    print(
        "Number of reviewed listings:",
        reviews["listing_id"].nunique()
    )

    print(
        "Number of interactions/reviews:",
        len(reviews)
    )

    print("\n=== LISTING-REVIEW RELATION ===")

    listing_ids = set(listings["id"])

    invalid_reviews = reviews[
        ~reviews["listing_id"].isin(listing_ids)
    ]

    print(
        "Reviews whose listing_id is not in listings:",
        len(invalid_reviews)
    )


if __name__ == "__main__":
    main()