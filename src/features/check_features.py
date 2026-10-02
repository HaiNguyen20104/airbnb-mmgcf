import numpy as np
import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent.parent

FEATURES_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "bangkok"
    / "features"
)

MAPPING_PATH = FEATURES_DIR / "item_mapping.parquet"

TEXT_FEATURES_PATH = FEATURES_DIR / "text_features.npy"
IMAGE_FEATURES_PATH = FEATURES_DIR / "image_features.npy"

EXPECTED_DIM = 512
L2_TOLERANCE = 1e-5


def check_feature_matrix(features, name, expected_rows):
    print(f"=== {name} ===")

    if features.ndim != 2:
        raise ValueError(
            f"{name}: expected a 2D matrix, got ndim={features.ndim}."
        )

    num_rows, num_dims = features.shape

    print(f"Shape: {features.shape}")
    print(f"dtype: {features.dtype}")
    print(f"Expected rows: {expected_rows}")
    print(f"Row count matches mapping: {num_rows == expected_rows}")

    if num_rows != expected_rows:
        raise ValueError(
            f"{name}: row count {num_rows} "
            f"does not match mapping {expected_rows}."
        )

    if num_dims != EXPECTED_DIM:
        raise ValueError(
            f"{name}: expected {EXPECTED_DIM} dimensions, "
            f"got {num_dims}."
        )

    if not np.issubdtype(features.dtype, np.floating):
        raise ValueError(
            f"{name}: expected floating-point features, "
            f"got {features.dtype}."
        )

    num_nan = int(np.isnan(features).sum())
    num_inf = int(np.isinf(features).sum())

    print(f"NaN values: {num_nan}")
    print(f"Inf values: {num_inf}")

    if num_nan > 0:
        raise ValueError(f"{name}: contains NaN values.")

    if num_inf > 0:
        raise ValueError(f"{name}: contains Inf values.")

    norms = np.linalg.norm(features, axis=1)

    min_norm = float(norms.min())
    max_norm = float(norms.max())
    mean_norm = float(norms.mean())

    l2_valid = np.allclose(
        norms,
        1.0,
        atol=L2_TOLERANCE,
    )

    print(f"Minimum L2 norm: {min_norm:.8f}")
    print(f"Maximum L2 norm: {max_norm:.8f}")
    print(f"Mean L2 norm: {mean_norm:.8f}")
    print(f"L2 normalization valid: {l2_valid}")

    if not l2_valid:
        raise ValueError(
            f"{name}: L2 normalization check failed."
        )

    print("Result: PASS")
    print()


def main():
    print("=== FEATURE VALIDATION ===\n")

    # 1. Check required files
    required_files = [
        MAPPING_PATH,
        TEXT_FEATURES_PATH,
        IMAGE_FEATURES_PATH,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    # 2. Validate item mapping
    mapping = pd.read_parquet(MAPPING_PATH)

    required_columns = {"item_idx", "listing_id"}
    missing_columns = required_columns - set(mapping.columns)

    if missing_columns:
        raise ValueError(
            f"Mapping is missing columns: {sorted(missing_columns)}"
        )

    print("=== ITEM MAPPING ===")
    print(f"Rows: {len(mapping)}")
    print(f"Unique item_idx: {mapping['item_idx'].nunique()}")
    print(f"Unique listing_id: {mapping['listing_id'].nunique()}")

    if mapping.empty:
        raise ValueError("Item mapping is empty.")

    if mapping[["item_idx", "listing_id"]].isna().any().any():
        raise ValueError("Item mapping contains missing IDs.")

    if mapping["item_idx"].duplicated().any():
        raise ValueError("Duplicate item_idx found in mapping.")

    if mapping["listing_id"].duplicated().any():
        raise ValueError("Duplicate listing_id found in mapping.")

    # Sort by item_idx before checking its continuity
    mapping = mapping.sort_values("item_idx").reset_index(drop=True)

    expected_item_idx = np.arange(len(mapping))

    mapping_valid = np.array_equal(
        mapping["item_idx"].to_numpy(),
        expected_item_idx,
    )

    print(
        f"item_idx continuous 0..N-1: {mapping_valid}"
    )

    if not mapping_valid:
        raise ValueError(
            "item_idx is not a continuous range from 0 to N-1."
        )

    print("Result: PASS\n")

    # 3. Validate text features
    text_features = np.load(TEXT_FEATURES_PATH)

    check_feature_matrix(
        text_features,
        "TEXT FEATURES",
        len(mapping),
    )

    # 4. Validate image features
    image_features = np.load(IMAGE_FEATURES_PATH)

    check_feature_matrix(
        image_features,
        "IMAGE FEATURES",
        len(mapping),
    )

    # 5. Compare feature matrices
    print("=== CROSS-MODAL CONSISTENCY ===")
    print(f"Text feature shape: {text_features.shape}")
    print(f"Image feature shape: {image_features.shape}")

    if text_features.shape[0] != image_features.shape[0]:
        raise ValueError(
            "Text and image feature row counts do not match."
        )

    if text_features.shape[1] != image_features.shape[1]:
        raise ValueError(
            "Text and image feature dimensions do not match."
        )

    print("Row counts match: True")
    print("Feature dimensions match: True")
    print("Result: PASS\n")

    # 6. Final result
    print("=== VALIDATION RESULT ===")
    print(
        "PASS: Item mapping, text features, and image features "
        "passed all validation checks."
    )


if __name__ == "__main__":
    main()