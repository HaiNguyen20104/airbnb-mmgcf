import pandas as pd

INPUT_PATH = "data/processed/bangkok/listings_kcore.parquet"


def is_empty(series):
    """Check values that are NaN or empty/whitespace strings."""
    return series.isna() | series.astype("string").str.strip().eq("")


def main():
    listings = pd.read_parquet(INPUT_PATH)

    print("=== BASIC INFO ===")
    print(f"Listings: {len(listings)}")
    print(f"Columns: {len(listings.columns)}")

    # ============================================================
    # 1. REQUIRED COLUMNS
    # ============================================================
    required_columns = [
        "id",
        "name",
        "description",
        "picture_url",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in listings.columns
    ]

    print("\n=== REQUIRED COLUMNS ===")

    if missing_columns:
        print("Missing columns:")
        for column in missing_columns:
            print(f"- {column}")
        raise ValueError("Required columns are missing.")
    else:
        print("All required columns exist.")

    # ============================================================
    # 2. LISTING ID
    # ============================================================
    print("\n=== ID CHECK ===")

    missing_id = listings["id"].isna().sum()
    duplicated_id = listings["id"].duplicated().sum()

    print(f"Missing id: {missing_id}")
    print(f"Duplicated id: {duplicated_id}")
    print(f"Unique id: {listings['id'].nunique()}")

    # ============================================================
    # 3. NAME
    # ============================================================
    print("\n=== NAME CHECK ===")

    name_empty = is_empty(listings["name"])

    print(f"Missing name: {listings['name'].isna().sum()}")
    print(f"Empty name: {name_empty.sum()}")

    # ============================================================
    # 4. DESCRIPTION
    # ============================================================
    print("\n=== DESCRIPTION CHECK ===")

    description_empty = is_empty(listings["description"])

    print(f"Missing description: {listings['description'].isna().sum()}")
    print(f"Empty description: {description_empty.sum()}")

    # ============================================================
    # 5. PICTURE URL
    # ============================================================
    print("\n=== PICTURE URL CHECK ===")

    picture_empty = is_empty(listings["picture_url"])

    print(f"Missing picture_url: {listings['picture_url'].isna().sum()}")
    print(f"Empty picture_url: {picture_empty.sum()}")

    # ============================================================
    # 6. COMBINED TEXT
    #
    # name + description
    # ============================================================
    print("\n=== COMBINED TEXT CHECK ===")

    name = listings["name"].fillna("").astype("string")
    description = listings["description"].fillna("").astype("string")

    text = (
        name.str.strip()
        + " "
        + description.str.strip()
    ).str.strip()

    empty_text = text.eq("")

    print(f"Empty combined text: {empty_text.sum()}")
    print(
        f"Non-empty combined text: "
        f"{(~empty_text).sum()}"
    )

    # ============================================================
    # 7. TEXT LENGTH
    # ============================================================
    text_length = text.str.len()

    print("\n=== TEXT LENGTH ===")
    print(f"Minimum characters: {text_length.min()}")
    print(f"Maximum characters: {text_length.max()}")
    print(f"Mean characters: {text_length.mean():.2f}")
    print(f"Median characters: {text_length.median():.2f}")

    # ============================================================
    # 8. FINAL SUMMARY
    # ============================================================
    print("\n=== MODALITY SUMMARY ===")

    print(f"Total listings: {len(listings)}")
    print(f"Listings with valid id: {len(listings) - missing_id}")
    print(f"Listings with name: {(~name_empty).sum()}")
    print(f"Listings with description: {(~description_empty).sum()}")
    print(f"Listings with picture_url: {(~picture_empty).sum()}")
    print(f"Listings with usable text: {(~empty_text).sum()}")

    print("\n=== CHECK COMPLETE ===")
    print("No data was modified.")


if __name__ == "__main__":
    main()