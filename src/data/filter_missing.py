import pandas as pd


INPUT_PATH = "data/processed/bangkok/listings_image_filtered.parquet"
OUTPUT_PATH = "data/processed/bangkok/listings_cleaned.parquet"


def main():
    # ============================================================
    # 1. LOAD DATASET
    # ============================================================

    listings = pd.read_parquet(INPUT_PATH)

    print("=== BEFORE MISSING VALUE FILTERING ===")
    print(f"Listings: {len(listings)}")

    # ============================================================
    # 2. CHECK MISSING VALUES
    # ============================================================

    print()
    print("=== MISSING VALUES ===")

    missing_price = listings["price"].isna().sum()
    missing_latitude = listings["latitude"].isna().sum()
    missing_longitude = listings["longitude"].isna().sum()

    print(f"Missing price: {missing_price}")
    print(f"Missing latitude: {missing_latitude}")
    print(f"Missing longitude: {missing_longitude}")

    # ============================================================
    # 3. REMOVE LISTINGS WITH MISSING REQUIRED VALUES
    # ============================================================

    required_columns = [
        "price",
        "latitude",
        "longitude",
    ]

    listings = listings.dropna(
        subset=required_columns
    ).copy()

    # ============================================================
    # 4. FINAL CHECK
    # ============================================================

    print()
    print("=== AFTER MISSING VALUE FILTERING ===")
    print(f"Listings: {len(listings)}")
    print(
        f"Removed: "
        f"{30618 - len(listings)}"
    )

    print()
    print("=== REMAINING MISSING VALUES ===")

    print(
        f"Missing price: "
        f"{listings['price'].isna().sum()}"
    )

    print(
        f"Missing latitude: "
        f"{listings['latitude'].isna().sum()}"
    )

    print(
        f"Missing longitude: "
        f"{listings['longitude'].isna().sum()}"
    )

    # ============================================================
    # 5. SAVE CLEANED DATASET
    # ============================================================

    listings.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()