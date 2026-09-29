import pandas as pd

K = 5

LISTINGS_PATH = "data/processed/bangkok/listings_cleaned.parquet"
REVIEWS_PATH = "data/raw/bangkok/reviews.csv.gz"

INTERACTIONS_OUTPUT = "data/processed/bangkok/interactions_kcore.parquet"
LISTINGS_OUTPUT = "data/processed/bangkok/listings_kcore.parquet"


def main():
    # ============================================================
    # 1. LOAD CLEANED LISTINGS
    # ============================================================
    listings = pd.read_parquet(LISTINGS_PATH)

    print("=== CLEANED LISTINGS ===")
    print(f"Listings: {len(listings)}")

    # ============================================================
    # 2. LOAD RAW REVIEWS
    #    Only keep columns needed for User-Item interactions
    # ============================================================
    reviews = pd.read_csv(
        REVIEWS_PATH,
        compression="gzip",
        usecols=["reviewer_id", "listing_id"],
    )

    print("\n=== RAW REVIEWS ===")
    print(f"Reviews: {len(reviews)}")

    # Make ID types consistent
    reviews["reviewer_id"] = reviews["reviewer_id"].astype("string")
    reviews["listing_id"] = reviews["listing_id"].astype("string")

    listings["id"] = listings["id"].astype("string")

    # ============================================================
    # 3. KEEP ONLY INTERACTIONS FOR CLEANED LISTINGS
    # ============================================================
    valid_listing_ids = set(listings["id"])

    interactions = reviews[
        reviews["listing_id"].isin(valid_listing_ids)
    ].copy()

    print("\n=== INTERACTIONS FOR CLEANED LISTINGS ===")
    print(f"Interactions: {len(interactions)}")
    print(f"Users: {interactions['reviewer_id'].nunique()}")
    print(f"Items: {interactions['listing_id'].nunique()}")

    # ============================================================
    # 4. COLLAPSE REPEATED USER-ITEM INTERACTIONS
    #
    # A user may review the same listing multiple times.
    # For implicit feedback, keep only one User-Item interaction.
    # ============================================================
    interactions = interactions.drop_duplicates(
        subset=["reviewer_id", "listing_id"]
    ).reset_index(drop=True)

    print("\n=== UNIQUE USER-ITEM INTERACTIONS ===")
    print(f"Interactions: {len(interactions)}")
    print(f"Users: {interactions['reviewer_id'].nunique()}")
    print(f"Items: {interactions['listing_id'].nunique()}")

    # ============================================================
    # 5. CALCULATE USER / ITEM DEGREE ONCE
    # ============================================================
    user_degree = interactions.groupby("reviewer_id").size()
    item_degree = interactions.groupby("listing_id").size()

    print("\n=== INITIAL DEGREE ===")
    print(f"Minimum user degree: {user_degree.min()}")
    print(f"Minimum item degree: {item_degree.min()}")

    # ============================================================
    # 6. ONE-PASS FILTERING
    #
    # Keep users with >= K interactions
    # Keep items with >= K interactions
    #
    # Degree is NOT recalculated after filtering.
    # ============================================================
    valid_users = user_degree[user_degree >= K].index
    valid_items = item_degree[item_degree >= K].index

    print("\n=== ONE-PASS FILTERING ===")
    print(f"Users with degree >= {K}: {len(valid_users)}")
    print(f"Items with degree >= {K}: {len(valid_items)}")

    # ============================================================
    # 7. KEEP INTERACTIONS BETWEEN VALID USERS AND ITEMS
    # ============================================================
    interactions_kcore = interactions[
        interactions["reviewer_id"].isin(valid_users)
        & interactions["listing_id"].isin(valid_items)
    ].copy()

    interactions_kcore = interactions_kcore.reset_index(drop=True)

    # ============================================================
    # 8. FILTER LISTINGS
    # ============================================================
    final_listing_ids = set(interactions_kcore["listing_id"])

    listings_kcore = listings[
        listings["id"].isin(final_listing_ids)
    ].copy()

    listings_kcore = listings_kcore.reset_index(drop=True)

    # ============================================================
    # 9. REPORT RESULTS
    # ============================================================
    print("\n=== FINAL RESULT ===")
    print(f"Interactions: {len(interactions_kcore)}")
    print(f"Users: {interactions_kcore['reviewer_id'].nunique()}")
    print(f"Items: {interactions_kcore['listing_id'].nunique()}")
    print(f"Listings: {len(listings_kcore)}")

    # Final degree is reported for observation only.
    # One-pass filtering does NOT guarantee final degree >= K.
    final_user_degree = interactions_kcore.groupby("reviewer_id").size()
    final_item_degree = interactions_kcore.groupby("listing_id").size()

    if len(interactions_kcore) > 0:
        print("\n=== FINAL DEGREE (AFTER INTERSECTION) ===")
        print(f"Minimum user degree: {final_user_degree.min()}")
        print(f"Minimum item degree: {final_item_degree.min()}")

    # ============================================================
    # 10. SAVE
    # ============================================================
    interactions_kcore.to_parquet(
        INTERACTIONS_OUTPUT,
        index=False,
    )

    listings_kcore.to_parquet(
        LISTINGS_OUTPUT,
        index=False,
    )

    print("\n=== SAVED ===")
    print(f"Interactions: {INTERACTIONS_OUTPUT}")
    print(f"Listings: {LISTINGS_OUTPUT}")


if __name__ == "__main__":
    main()