import pandas as pd


LISTINGS_PATH = "data/processed/bangkok/listings_image_filtered.parquet"


def main():
    # ============================================================
    # 1. LOAD CLEANED DATASET
    # ============================================================

    listings = pd.read_parquet(
        LISTINGS_PATH
    )

    print("=== CURRENT DATASET ===")
    print(f"Total listings: {len(listings)}")

    # ============================================================
    # 2. CHECK LISTING ID
    # ============================================================

    print()
    print("=== LISTING ID ===")

    unique_ids = listings["id"].nunique()
    duplicated_ids = listings["id"].duplicated().sum()

    print(f"Unique IDs: {unique_ids}")
    print(f"Duplicated IDs: {duplicated_ids}")

    invalid_ids = (
        ~listings["id"]
        .astype("string")
        .str.fullmatch(r"\d+")
    ).sum()

    print(f"Invalid IDs: {invalid_ids}")

    # ============================================================
    # 3. CHECK MISSING PRICE
    # ============================================================

    print()
    print("=== MISSING PRICE ===")

    missing_price = listings["price"].isna().sum()

    print(f"Missing price: {missing_price}")

    # ============================================================
    # 4. CHECK MISSING LOCATION
    # ============================================================

    print()
    print("=== MISSING LOCATION ===")

    missing_latitude = listings["latitude"].isna().sum()
    missing_longitude = listings["longitude"].isna().sum()

    print(f"Missing latitude: {missing_latitude}")
    print(f"Missing longitude: {missing_longitude}")

    # ============================================================
    # 5. CHECK PICTURE URL
    # ============================================================

    print()
    print("=== PICTURE URL ===")

    missing_picture = listings["picture_url"].isna().sum()

    empty_picture = (
        listings["picture_url"]
        .astype("string")
        .str.strip()
        .eq("")
        .sum()
    )

    print(f"Missing picture_url: {missing_picture}")
    print(f"Empty picture_url: {empty_picture}")

    # ============================================================
    # 6. CHECK DATASET INTEGRITY
    # ============================================================

    print()
    print("=== DATASET INTEGRITY ===")

    print(
        f"Rows == unique IDs: "
        f"{len(listings) == unique_ids}"
    )

    print(
        f"All IDs numeric: "
        f"{invalid_ids == 0}"
    )


if __name__ == "__main__":
    main()