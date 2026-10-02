from pathlib import Path

import pandas as pd


BASE_DIR = Path("data/processed/bangkok")
FEATURES_DIR = BASE_DIR / "features"

LISTINGS_PATH = BASE_DIR / "listings_kcore.parquet"
INTERACTIONS_PATH = BASE_DIR / "interactions_kcore.parquet"
MAPPING_PATH = FEATURES_DIR / "item_mapping.parquet"
STATUS_PATH = FEATURES_DIR / "image_download_status.parquet"

OUTPUT_LISTINGS = BASE_DIR / "listings_final.parquet"
OUTPUT_INTERACTIONS = BASE_DIR / "interactions_final.parquet"
OUTPUT_REMOVED = FEATURES_DIR / "listings_removed_invalid_images.parquet"


def main():
    listings = pd.read_parquet(LISTINGS_PATH)
    interactions = pd.read_parquet(INTERACTIONS_PATH)
    mapping = pd.read_parquet(MAPPING_PATH)
    status = pd.read_parquet(STATUS_PATH)

    # Chỉ giữ những ảnh tải về thành công hoặc đã có trong cache.
    valid_statuses = {"downloaded", "cached"}

    invalid_status = status.loc[
        ~status["status"].isin(valid_statuses)
    ].copy()

    invalid_item_indices = set(invalid_status["item_idx"].astype(int))

    # Ánh xạ item_idx sang listing_id.
    invalid_listing_ids = set(
        mapping.loc[
            mapping["item_idx"].astype(int).isin(invalid_item_indices),
            "listing_id",
        ]
    )

    print(f"Listings ban đầu: {len(listings)}")
    print(f"Interactions ban đầu: {len(interactions)}")
    print(f"Item có ảnh không hợp lệ: {len(invalid_item_indices)}")
    print(f"Listing cần loại bỏ: {len(invalid_listing_ids)}")

    # Loại các listing bị lỗi ảnh.
    listings_final = listings.loc[
        ~listings["id"].isin(invalid_listing_ids)
    ].copy()

    # Loại tương tác liên quan đến các listing đó.
    interactions_final = interactions.loc[
        ~interactions["listing_id"].isin(invalid_listing_ids)
    ].copy()

    removed = listings.loc[
        listings["id"].isin(invalid_listing_ids)
    ].copy()

    listings_final.to_parquet(OUTPUT_LISTINGS, index=False)
    interactions_final.to_parquet(OUTPUT_INTERACTIONS, index=False)
    removed.to_parquet(OUTPUT_REMOVED, index=False)

    print(f"\nListings còn lại: {len(listings_final)}")
    print(f"Interactions còn lại: {len(interactions_final)}")
    print(f"Đã lưu listings: {OUTPUT_LISTINGS}")
    print(f"Đã lưu interactions: {OUTPUT_INTERACTIONS}")
    print(f"Đã lưu listing bị loại: {OUTPUT_REMOVED}")


if __name__ == "__main__":
    main()