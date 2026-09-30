import pandas as pd
from pathlib import Path

INPUT_PATH = "data/processed/bangkok/listings_kcore.parquet"
OUTPUT_PATH = "data/processed/bangkok/features/item_mapping.parquet"


def main():
    listings = pd.read_parquet(INPUT_PATH)

    print("=== INPUT ===")
    print(f"Listings: {len(listings)}")

    # Validate required column.
    if "id" not in listings.columns:
        raise ValueError("Required column 'id' is missing.")

    # Normalize listing IDs to string so the mapping is consistent
    # with the interaction data used later in the pipeline.
    listing_ids = listings["id"].astype("string").str.strip()

    # Basic validation before creating the mapping.
    missing_id = listing_ids.isna().sum()
    empty_id = listing_ids.eq("").sum()
    duplicated_id = listing_ids.duplicated().sum()

    print("=== ID CHECK ===")
    print(f"Missing id: {missing_id}")
    print(f"Empty id: {empty_id}")
    print(f"Duplicated id: {duplicated_id}")

    if missing_id > 0 or empty_id > 0:
        raise ValueError("Listing IDs contain missing or empty values.")

    if duplicated_id > 0:
        raise ValueError("Duplicate listing IDs found.")

    # Sort by listing_id to make item_idx deterministic.
    mapping = (
        pd.DataFrame({"listing_id": listing_ids})
        .sort_values("listing_id", kind="stable")
        .reset_index(drop=True)
    )

    mapping.insert(0, "item_idx", range(len(mapping)))

    print("=== MAPPING CHECK ===")
    print(f"Rows: {len(mapping)}")
    print(f"Unique item_idx: {mapping['item_idx'].nunique()}")
    print(f"Unique listing_id: {mapping['listing_id'].nunique()}")
    print(f"Minimum item_idx: {mapping['item_idx'].min()}")
    print(f"Maximum item_idx: {mapping['item_idx'].max()}")

    # Final consistency checks.
    expected_item_idx = pd.RangeIndex(len(mapping))

    if not mapping["item_idx"].equals(pd.Series(expected_item_idx)):
        raise ValueError("item_idx is not a continuous range from 0 to n-1.")

    if mapping["listing_id"].duplicated().any():
        raise ValueError("Duplicate listing_id found in mapping.")

    if len(mapping) != len(listings):
        raise ValueError("Mapping row count does not match listings row count.")

    # Create output directory and save as Parquet.
    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_parquet(output_path, index=False)

    print("=== SAMPLE ===")
    print(mapping.head(10).to_string(index=False))

    print("=== SAVED ===")
    print(f"Path: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()