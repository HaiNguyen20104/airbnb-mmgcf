import numpy as np
import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "processed" / "bangkok"
FEATURES_DIR = DATA_DIR / "features"

INTERACTIONS_PATH = DATA_DIR / "interactions_final.parquet"
LISTINGS_PATH = DATA_DIR / "listings_final.parquet"
MAPPING_PATH = FEATURES_DIR / "item_mapping.parquet"

TEXT_FEATURES_PATH = FEATURES_DIR / "text_features.npy"
IMAGE_FEATURES_PATH = FEATURES_DIR / "image_features.npy"


def inspect_file(path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_parquet(path)

    print(f"\n{'=' * 60}")
    print(f"FILE: {path.name}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("\nDtypes:")
    print(df.dtypes)
    print("\nFirst 5 rows:")
    print(df.head())

    return df


def main():
    # 1. Read data
    interactions = inspect_file(INTERACTIONS_PATH)
    listings = inspect_file(LISTINGS_PATH)
    mapping = inspect_file(MAPPING_PATH)

    # 2. Inspect interaction columns
    print(f"\n{'=' * 60}")
    print("INTERACTION CHECKS")

    user_candidates = ["reviewer_id", "user_id", "user_idx"]
    item_candidates = ["listing_id", "item_id", "item_idx"]

    user_col = next(
        (col for col in user_candidates if col in interactions.columns),
        None,
    )
    item_col = next(
        (col for col in item_candidates if col in interactions.columns),
        None,
    )

    if user_col is None or item_col is None:
        print("Cannot identify user/item columns automatically.")
        print(f"Available columns: {interactions.columns.tolist()}")
        print("Please inspect the columns above before proceeding.")
        return

    print(f"User column: {user_col}")
    print(f"Item column: {item_col}")
    print(f"Interaction rows: {len(interactions)}")
    print(f"Unique users: {interactions[user_col].nunique()}")
    print(f"Unique items: {interactions[item_col].nunique()}")

    missing_interaction_ids = interactions[
        [user_col, item_col]
    ].isna().any(axis=1).sum()

    duplicate_pairs = interactions.duplicated(
        subset=[user_col, item_col]
    ).sum()

    print(f"Rows with missing user/item ID: {missing_interaction_ids}")
    print(f"Duplicate user-item pairs: {duplicate_pairs}")

    # 3. Check mapping
    if "listing_id" not in mapping.columns:
        raise ValueError("item_mapping is missing listing_id.")

    if "item_idx" not in mapping.columns:
        raise ValueError("item_mapping is missing item_idx.")

    if item_col == "listing_id":
        interaction_item_ids = set(interactions[item_col].dropna())
        mapping_listing_ids = set(mapping["listing_id"].dropna())

        unmatched_items = interaction_item_ids - mapping_listing_ids

        print(f"Interaction listing IDs absent from mapping: {len(unmatched_items)}")

        if unmatched_items:
            print("Example unmatched IDs:", list(unmatched_items)[:10])

    # 4. Check listings against mapping
    if "id" not in listings.columns:
        raise ValueError("listings_final is missing the id column.")

    listing_ids = set(listings["id"].dropna())
    mapping_ids = set(mapping["listing_id"].dropna())

    print(f"\n{'=' * 60}")
    print("LISTING-MAPPING CHECKS")
    print(f"Listings rows: {len(listings)}")
    print(f"Unique listing IDs: {listings['id'].nunique()}")
    print(f"Mapping rows: {len(mapping)}")
    print(f"Listings absent from mapping: {len(listing_ids - mapping_ids)}")
    print(f"Mapping IDs absent from listings: {len(mapping_ids - listing_ids)}")

    # 5. Check feature matrices
    print(f"\n{'=' * 60}")
    print("FEATURE CHECKS")

    for name, path in [
        ("Text", TEXT_FEATURES_PATH),
        ("Image", IMAGE_FEATURES_PATH),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{name} features not found: {path}")

        features = np.load(path, mmap_mode="r")

        print(f"{name} features shape: {features.shape}")
        print(f"{name} rows match mapping: {features.shape[0] == len(mapping)}")

        if features.shape[0] != len(mapping):
            raise ValueError(
                f"{name} feature rows do not match item mapping."
            )

    print(f"\n{'=' * 60}")
    print("INSPECTION COMPLETED")


if __name__ == "__main__":
    main()