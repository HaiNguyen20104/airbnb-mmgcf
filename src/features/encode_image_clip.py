import torch
import numpy as np
import pandas as pd
import requests
from io import BytesIO
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from transformers import CLIPProcessor, CLIPModel

BASE_DIR = Path(__file__).resolve().parent.parent.parent
LISTINGS_PATH = BASE_DIR / "data" / "processed" / "bangkok" / "listings_kcore.parquet"
MAPPING_PATH = BASE_DIR / "data" / "processed" / "bangkok" / "features" / "item_mapping.parquet"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "bangkok" / "features" / "image.npy"

MODEL_NAME = "openai/clip-vit-base-patch32"
BATCH_SIZE = 64
REQUEST_TIMEOUT = 5
NUM_DOWNLOAD_WORKERS = 20


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def download_single_image(url):
    if not url:
        return Image.new("RGB", (224, 224), (255, 255, 255))
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": "Mozilla/5.0"})
        if response.status_code == 200:
            return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception:
        pass
    return Image.new("RGB", (224, 224), (255, 255, 255))


def main():
    print("=== STARTING BATCH-BASED FAST CLIP IMAGE EMBEDDING ===")

    # 1. Load data
    print("Loading datasets...")
    df_listings = pd.read_parquet(LISTINGS_PATH)
    df_mapping = pd.read_parquet(MAPPING_PATH)

    # 2. Normalize ID & Merge
    df_listings["id"] = df_listings["id"].astype("string").str.strip()
    df_mapping["listing_id"] = df_mapping["listing_id"].astype("string").str.strip()

    df_merged = pd.merge(
        df_mapping,
        df_listings,
        left_on="listing_id",
        right_on="id",
        how="left",
        validate="one_to_one",
    )

    if len(df_merged) != len(df_mapping):
        raise ValueError("Mapping and listings size mismatch.")
    if df_merged["id"].isna().any():
        raise ValueError(f"{df_merged['id'].isna().sum()} listing IDs could not be matched.")

    df_merged = df_merged.sort_values("item_idx").reset_index(drop=True)
    urls = df_merged["picture_url"].tolist()

    # 3. Model setup
    device = get_device()
    print(f"Using device: {device}")
    print(f"Loading CLIP model: {MODEL_NAME}...")

    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    # 4. Extract image features (512-dim)
    print(f"Processing {len(urls)} images in chunks of {BATCH_SIZE}...")
    all_embeddings = []

    with ThreadPoolExecutor(max_workers=NUM_DOWNLOAD_WORKERS) as executor:
        with torch.inference_mode():
            for i in tqdm(range(0, len(urls), BATCH_SIZE)):
                batch_urls = urls[i : i + BATCH_SIZE]
                batch_images = list(executor.map(download_single_image, batch_urls))

                inputs = processor(images=batch_images, return_tensors="pt")
                inputs = {key: value.to(device) for key, value in inputs.items()}

                # Lấy visual feature chuẩn 512 chiều đã qua projection
                outputs = model.get_image_features(**inputs)
                embeddings = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs
                all_embeddings.append(embeddings.cpu().numpy())

                del batch_images

    image_features = np.concatenate(all_embeddings, axis=0)

    # 5. L2 Normalization
    print("\nNormalizing embeddings (L2)...")
    norms = np.linalg.norm(image_features, ord=2, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    image_features = (image_features / norms).astype(np.float32)

    # 6. Quality Checks
    print("\n=== EMBEDDING QUALITY CHECK ===")
    num_items, num_dims = image_features.shape
    num_nan = np.isnan(image_features).sum()
    num_inf = np.isinf(image_features).sum()

    post_norms = np.linalg.norm(image_features, axis=1)
    is_l2_valid = np.allclose(post_norms, 1.0, atol=1e-5)

    print(f"Total encoded items: {num_items}")
    print(f"Vector dimensions: {num_dims}")
    print(f"dtype: {image_features.dtype}")
    print(f"NaN values: {num_nan}")
    print(f"Inf values: {num_inf}")
    print(f"L2 Normalization valid: {is_l2_valid}")

    if num_items != len(df_mapping):
        raise ValueError("Feature count does not match item mapping.")
    if num_dims != 512:
        raise ValueError(f"Unexpected feature dimension: {num_dims}")
    if num_nan > 0 or num_inf > 0:
        raise ValueError("Features contain NaN or Inf.")
    if not is_l2_valid:
        raise ValueError("L2 normalization check failed.")

    # 7. Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_PATH, image_features)
    print(f"\nSUCCESS: Saved normalized embeddings to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
