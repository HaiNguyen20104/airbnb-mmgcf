import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
INPUT_PATH = BASE_DIR / "data" / "processed" / "bangkok" / "interactions_final.parquet"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "bangkok" / "features" / "user_mapping.parquet"


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

    interactions = pd.read_parquet(INPUT_PATH)

    print("=== INPUT ===")
    print(f"Input path: {INPUT_PATH}")
    print(f"Interactions count: {len(interactions)}")

    # 1. Validate required column
    user_candidates = ["reviewer_id", "user_id"]
    user_col = next((col for col in user_candidates if col in interactions.columns), None)

    if user_col is None:
        raise ValueError(
            f"Required user column not found. Available columns: {interactions.columns.tolist()}"
        )

    # 2. Normalize reviewer IDs
    reviewer_ids = interactions[user_col].astype("string").str.strip()

    # 3. Check for missing or empty IDs
    missing_count = reviewer_ids.isna().sum()
    empty_count = reviewer_ids.eq("").sum()

    print("\n=== ID CHECK ===")
    print(f"Using user column: {user_col}")
    print(f"Missing IDs: {missing_count}")
    print(f"Empty IDs: {empty_count}")

    if missing_count > 0 or empty_count > 0:
        raise ValueError("Interactions contain missing or empty user IDs.")

    # 4. Extract unique users and sort deterministically
    unique_reviewers = reviewer_ids.drop_duplicates()
    mapping = (
        pd.DataFrame({"reviewer_id": unique_reviewers})
        .sort_values("reviewer_id", kind="stable")
        .reset_index(drop=True)
    )

    # 5. Assign user_idx (0 to num_users - 1)
    mapping.insert(0, "user_idx", range(len(mapping)))

    print("\n=== MAPPING CHECK ===")
    print(f"Total unique users: {len(mapping)}")
    print(f"Unique user_idx: {mapping['user_idx'].nunique()}")
    print(f"Unique reviewer_id: {mapping['reviewer_id'].nunique()}")
    print(f"Minimum user_idx: {mapping['user_idx'].min()}")
    print(f"Maximum user_idx: {mapping['user_idx'].max()}")

    # 6. Consistency validations
    expected_user_idx = pd.RangeIndex(len(mapping))
    if not mapping["user_idx"].equals(pd.Series(expected_user_idx)):
        raise ValueError("user_idx is not a continuous range from 0 to n-1.")

    if mapping["reviewer_id"].duplicated().any():
        raise ValueError("Duplicate reviewer_id found in mapping.")

    # 7. Save mapping to Parquet
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_parquet(OUTPUT_PATH, index=False)

    print("\n=== SAMPLE ===")
    print(mapping.head(10).to_string(index=False))

    print("\n=== SAVED ===")
    print(f"Saved user mapping to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
