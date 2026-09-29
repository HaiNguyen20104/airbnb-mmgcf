import pandas as pd


LISTINGS_PATH = "data/raw/bangkok/listings.csv.gz"
IMAGE_CHECK_PATH = "data/processed/bangkok/image_check.csv"
OUTPUT_PATH = "data/processed/bangkok/listings_image_filtered.parquet"


def main():
    # ============================================================
    # 1. LOAD DATA
    # ============================================================

    listings = pd.read_csv(
        LISTINGS_PATH,
        low_memory=False,
    )

    image_check = pd.read_csv(
        IMAGE_CHECK_PATH,
        low_memory=False,
    )

    print("=== BEFORE FILTERING ===")
    print(f"Listings: {len(listings)}")

    # ============================================================
    # 2. FILTER LISTINGS WITH VALID IMAGES
    # ============================================================

    valid_image_ids = set(
        image_check.loc[
            image_check["image_status"] == "valid",
            "listing_id"
        ].astype("string")
    )

    # Convert ID to string để tránh vấn đề khác kiểu dữ liệu
    listings["id"] = listings["id"].astype("string")

    listings = listings[
        listings["id"].isin(valid_image_ids)
    ].copy()

    print()
    print("=== AFTER IMAGE FILTERING ===")
    print(f"Listings: {len(listings)}")
    print(f"Removed: {31069 - len(listings)}")

    # ============================================================
    # 3. REMOVE INVALID LISTING IDs
    # ============================================================

    valid_id_mask = (
        listings["id"]
        .astype("string")
        .str.fullmatch(r"\d+")
    )

    invalid_id_count = (~valid_id_mask).sum()

    print()
    print("=== INVALID ID FILTERING ===")
    print(f"Invalid IDs removed: {invalid_id_count}")

    listings = listings[
        valid_id_mask
    ].copy()

    # ============================================================
    # 4. CHECK ID INTEGRITY
    # ============================================================

    unique_ids = listings["id"].nunique()
    duplicated_ids = listings["id"].duplicated().sum()

    print()
    print("=== ID CHECK ===")
    print(f"Unique IDs: {unique_ids}")
    print(f"Duplicated IDs: {duplicated_ids}")

    # ============================================================
    # 5. FINAL CHECK BEFORE SAVING
    # ============================================================

    print()
    print("=== BEFORE SAVING ===")
    print(f"Rows: {len(listings)}")

    invalid_id_rows = listings[
        ~listings["id"].astype("string").str.fullmatch(r"\d+")
    ]

    print(f"Rows with invalid ID: {len(invalid_id_rows)}")

    if len(invalid_id_rows) > 0:
        print(
            invalid_id_rows[
                ["id", "name", "picture_url"]
            ].to_string(index=False)
        )

    # ============================================================
    # 6. SAVE AS PARQUET
    # ============================================================

    listings.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()