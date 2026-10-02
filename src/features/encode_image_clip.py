
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


# ==================================================
# CONFIG
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

FEATURES_DIR = (
    BASE_DIR / "data" / "processed" / "bangkok" / "features"
)

STATUS_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "bangkok"
    / "features"
    / "image_download_status.parquet"
)

MAPPING_PATH = FEATURES_DIR / "item_mapping.parquet"
IMAGES_DIR = FEATURES_DIR / "images"
OUTPUT_PATH = FEATURES_DIR / "image_features.npy"

MODEL_NAME = "openai/clip-vit-base-patch32"
BATCH_SIZE = 32


# ==================================================
# DEVICE
# ==================================================

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


# ==================================================
# IMAGE FILE DISCOVERY
# ==================================================

def build_image_index():
    """
    Index local image files by filename stem.

    Supports files named with either item_idx or listing_id,
    for example:
        0.jpg
        0.webp
        1000427460611414626.jpg
    """

    if not IMAGES_DIR.exists():
        raise FileNotFoundError(
            f"Image directory does not exist: {IMAGES_DIR}"
        )

    image_extensions = {
        ".jpg", ".jpeg", ".png", ".webp",
        ".bmp", ".gif", ".tif", ".tiff", ".img"
    }

    image_index = {}

    for path in IMAGES_DIR.rglob("*"):
        if (
            path.is_file()
            and path.suffix.lower() in image_extensions
        ):
            image_index.setdefault(path.stem, path)

    print(f"Image files discovered: {len(image_index)}")

    return image_index


def find_image_path(row, image_index):
    """
    Find image by item_idx first, then listing_id.
    """

    item_idx = str(int(row["item_idx"]))
    listing_id = str(row["listing_id"])

    if item_idx in image_index:
        return image_index[item_idx]

    if listing_id in image_index:
        return image_index[listing_id]

    return None


# ==================================================
# MAIN
# ==================================================

def main():
    print("=== STARTING CLIP IMAGE FEATURE EXTRACTION ===")

    # 1. Load mapping
    mapping = pd.read_parquet(MAPPING_PATH)

    status = pd.read_parquet(STATUS_PATH)

    required_columns = {"item_idx", "listing_id"}

    if not required_columns.issubset(mapping.columns):
        raise ValueError(
            f"Mapping must contain columns: {required_columns}"
        )

    mapping = mapping.sort_values("item_idx").reset_index(drop=True)

    if mapping["item_idx"].duplicated().any():
        raise ValueError("Duplicate item_idx found in mapping.")

    if mapping["listing_id"].duplicated().any():
        raise ValueError("Duplicate listing_id found in mapping.")

    expected_indices = np.arange(len(mapping))

    if not np.array_equal(
        mapping["item_idx"].to_numpy(),
        expected_indices,
    ):
        raise ValueError(
            "item_idx must be continuous from 0 to N-1."
        )

    print(f"Items in mapping: {len(mapping)}")

    # 2. Find local images
    status = status.copy()
    status["listing_id"] = status["listing_id"].astype(str)

    mapping = mapping.copy()
    mapping["listing_id"] = mapping["listing_id"].astype(str)

    mapping = mapping.sort_values("item_idx").reset_index(drop=True)

    status_by_listing = status.set_index("listing_id")

    image_paths = []

    for _, row in mapping.iterrows():
        listing_id = row["listing_id"]

        if listing_id not in status_by_listing.index:
            raise FileNotFoundError(
                f"No image status for listing_id={listing_id}"
            )

        image_info = status_by_listing.loc[listing_id]

        if isinstance(image_info, pd.DataFrame):
            raise ValueError(
                f"Duplicate image status for listing_id={listing_id}"
            )

        if image_info["status"] not in {"cached", "downloaded"}:
            raise ValueError(
                f"Image is not valid for listing_id={listing_id}: "
                f"{image_info['status']}"
            )

        local_path = image_info["local_path"]

        if pd.isna(local_path) or not local_path:
            raise FileNotFoundError(
                f"Missing local_path for listing_id={listing_id}"
            )

        image_path = Path(local_path)

        if not image_path.is_absolute():
            image_path = BASE_DIR / image_path

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image file does not exist: {image_path}"
            )

        image_paths.append(image_path)

    # 3. Load CLIP
    device = get_device()
    print(f"Using device: {device}")
    print(f"Loading model: {MODEL_NAME}")

    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME)
    model = model.to(device)
    model.eval()

    # 4. Extract image features in mapping order
    all_embeddings = []

    print(
        f"Extracting {len(image_paths)} images "
        f"with batch size {BATCH_SIZE}..."
    )

    with torch.inference_mode():
        for start in tqdm(range(0, len(image_paths), BATCH_SIZE)):
            batch_paths = image_paths[start:start + BATCH_SIZE]
            batch_images = []

            # Fail explicitly if any matched file is unreadable.
            for path in batch_paths:
                try:
                    with Image.open(path) as image:
                        image = ImageOps.exif_transpose(image)
                        image = image.convert("RGB")
                        batch_images.append(image.copy())
                except Exception as exc:
                    raise RuntimeError(
                        f"Cannot read image: {path}"
                    ) from exc

            inputs = processor(
                images=batch_images,
                return_tensors="pt",
            )

            pixel_values = inputs["pixel_values"].to(device)

            outputs = model.get_image_features(
                pixel_values=pixel_values
            )

            if isinstance(outputs, torch.Tensor):
                embeddings = outputs
            else:
                embeddings = outputs.pooler_output

            all_embeddings.append(
                embeddings.detach().float().cpu().numpy()
            )

    # 5. Combine batches
    image_features = np.concatenate(all_embeddings, axis=0)

    # 6. Validate shape before normalization
    expected_rows = len(mapping)

    if image_features.shape != (expected_rows, 512):
        raise ValueError(
            "Unexpected feature shape: "
            f"{image_features.shape}; "
            f"expected ({expected_rows}, 512)"
        )

    if not np.isfinite(image_features).all():
        raise ValueError("Image features contain NaN or Inf.")

    # 7. L2 normalization
    norms = np.linalg.norm(
        image_features,
        ord=2,
        axis=1,
        keepdims=True,
    )

    if np.any(norms == 0):
        raise ValueError("Found zero-norm image embeddings.")

    image_features = (
        image_features / norms
    ).astype(np.float32)

    # 8. Final checks
    post_norms = np.linalg.norm(image_features, axis=1)

    print("\n=== IMAGE FEATURE QUALITY CHECK ===")
    print(f"Total encoded images: {image_features.shape[0]}")
    print(f"Vector dimensions: {image_features.shape[1]}")
    print(f"dtype: {image_features.dtype}")
    print(f"NaN values: {np.isnan(image_features).sum()}")
    print(f"Inf values: {np.isinf(image_features).sum()}")
    print(
        "L2 normalization valid:",
        np.allclose(post_norms, 1.0, atol=1e-5),
    )

    # 9. Save only after all checks pass
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_PATH, image_features)

    print(f"\nSUCCESS: Saved image features to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()