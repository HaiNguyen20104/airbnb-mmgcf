import pandas as pd

INTERACTIONS_PATH = "data/processed/bangkok/interactions_kcore.parquet"
LISTINGS_PATH = "data/processed/bangkok/listings_kcore.parquet"


def main():
    interactions = pd.read_parquet(INTERACTIONS_PATH)
    listings = pd.read_parquet(LISTINGS_PATH)

    print("=== BASIC INFO ===")
    print(f"Interactions: {len(interactions)}")
    print(f"Users: {interactions['reviewer_id'].nunique()}")
    print(f"Listings: {len(listings)}")
    print(f"Unique listing IDs in interactions: {interactions['listing_id'].nunique()}")

    # ------------------------------------------------------------
    # 1. Check missing IDs
    # ------------------------------------------------------------
    print("\n=== MISSING ID CHECK ===")

    print(
        f"Missing reviewer_id: "
        f"{interactions['reviewer_id'].isna().sum()}"
    )

    print(
        f"Missing listing_id: "
        f"{interactions['listing_id'].isna().sum()}"
    )

    print(
        f"Missing listing id in listings: "
        f"{listings['id'].isna().sum()}"
    )

    # ------------------------------------------------------------
    # 2. Check interaction -> listing
    # ------------------------------------------------------------
    interaction_listing_ids = set(interactions["listing_id"])
    listing_ids = set(listings["id"])

    orphan_interactions = interaction_listing_ids - listing_ids

    print("\n=== INTERACTION -> LISTING ===")
    print(
        f"Listing IDs in interactions but not in listings: "
        f"{len(orphan_interactions)}"
    )

    if orphan_interactions:
        print("Example:", list(orphan_interactions)[:10])

    # ------------------------------------------------------------
    # 3. Check listing -> interaction
    # ------------------------------------------------------------
    listings_without_interactions = listing_ids - interaction_listing_ids

    print("\n=== LISTING -> INTERACTION ===")
    print(
        f"Listings without any interaction: "
        f"{len(listings_without_interactions)}"
    )

    if listings_without_interactions:
        print("Example:", list(listings_without_interactions)[:10])

    # ------------------------------------------------------------
    # 4. Check duplicate listing IDs
    # ------------------------------------------------------------
    duplicated_listing_ids = listings["id"].duplicated().sum()

    print("\n=== LISTING ID DUPLICATES ===")
    print(f"Duplicated listing IDs: {duplicated_listing_ids}")

    # ------------------------------------------------------------
    # 5. Check duplicate User-Item interactions
    # ------------------------------------------------------------
    duplicated_interactions = interactions.duplicated(
        subset=["reviewer_id", "listing_id"]
    ).sum()

    print("\n=== USER-ITEM DUPLICATES ===")
    print(
        f"Duplicated User-Item interactions: "
        f"{duplicated_interactions}"
    )

    # ------------------------------------------------------------
    # 6. Final result
    # ------------------------------------------------------------
    is_consistent = (
        interactions["reviewer_id"].notna().all()
        and interactions["listing_id"].notna().all()
        and listings["id"].notna().all()
        and len(orphan_interactions) == 0
        and len(listings_without_interactions) == 0
        and duplicated_listing_ids == 0
        and duplicated_interactions == 0
    )

    print("\n=== CONSISTENCY RESULT ===")

    if is_consistent:
        print("PASS: User-Item data is consistent.")
    else:
        print("FAIL: User-Item data has consistency issues.")


if __name__ == "__main__":
    main()