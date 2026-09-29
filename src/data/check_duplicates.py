import pandas as pd


LISTINGS_PATH = "data/raw/bangkok/listings.csv.gz"
REVIEWS_PATH = "data/raw/bangkok/reviews.csv.gz"


def main():
    listings = pd.read_csv(LISTINGS_PATH)
    reviews = pd.read_csv(REVIEWS_PATH)

    print("=== LISTINGS ===")

    duplicate_listing_id = listings["id"].duplicated().sum()
    duplicate_listing_rows = listings.duplicated().sum()

    print(f"Duplicated listing IDs: {duplicate_listing_id}")
    print(f"Duplicated listing rows: {duplicate_listing_rows}")

    print()
    print("=== REVIEWS ===")

    duplicate_review_id = reviews["id"].duplicated().sum()
    duplicate_review_rows = reviews.duplicated().sum()

    print(f"Duplicated review IDs: {duplicate_review_id}")
    print(f"Duplicated review rows: {duplicate_review_rows}")

    print()
    print("=== USER-ITEM INTERACTIONS ===")

    duplicate_interactions = reviews.duplicated(
        subset=["reviewer_id", "listing_id"]
    ).sum()

    print(
        f"Duplicated (reviewer_id, listing_id) pairs: "
        f"{duplicate_interactions}"
    )


if __name__ == "__main__":
    main()