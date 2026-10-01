import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import CLIPTokenizer, CLIPModel

BASE_DIR = Path(__file__).resolve().parent.parent.parent

LISTINGS_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "bangkok"
    / "listings_kcore.parquet"
)

MAPPING_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "bangkok"
    / "features"
    / "item_mapping.parquet"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "bangkok"
    / "features"
    / "text_features.npy"
)

MODEL_NAME = "openai/clip-vit-base-patch32"
BATCH_SIZE = 64
MAX_LENGTH = 77


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def main():
    print("=== STARTING CLIP TEXT EMBEDDING GENERATION ===")

    # --------------------------------------------------
    # 1. Load data
    # --------------------------------------------------
    print("Loading datasets...")

    df_listings = pd.read_parquet(LISTINGS_PATH)
    df_mapping = pd.read_parquet(MAPPING_PATH)

    # --------------------------------------------------
    # 2. Merge mapping -> listings
    # --------------------------------------------------
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

    if df_merged["name"].isna().any():
        raise ValueError("Some listing IDs could not be matched.")

    df_merged = (
        df_merged
        .sort_values("item_idx")
        .reset_index(drop=True)
    )

    # --------------------------------------------------
    # 3. Prepare text
    # --------------------------------------------------
    print("Preparing combined texts...")

    names = df_merged["name"].fillna("").astype("string")
    descriptions = df_merged["description"].fillna("").astype("string")

    texts = (
        names.str.strip()
        + " . "
        + descriptions.str.strip()
    ).tolist()

    # --------------------------------------------------
    # 4. Load CLIP
    # --------------------------------------------------
    print(f"Loading CLIP model: {MODEL_NAME}")

    device = get_device()
    print(f"Using device: {device}")

    tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME)

    model = CLIPModel.from_pretrained(MODEL_NAME)
    model = model.to(device)
    model.eval()

    # --------------------------------------------------
    # 5. Extract text features
    # --------------------------------------------------
    print(
        f"Encoding {len(texts)} texts "
        f"with batch size {BATCH_SIZE}..."
    )

    all_embeddings = []

    with torch.inference_mode():
        for i in tqdm(
            range(0, len(texts), BATCH_SIZE)
        ):
            batch_texts = texts[
                i:i + BATCH_SIZE
            ]

            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )

            inputs = {
                key: value.to(device)
                for key, value in inputs.items()
            }

            text_outputs = model.get_text_features(**inputs)

            embeddings = text_outputs.pooler_output

            all_embeddings.append(
                embeddings.cpu().numpy()
            )

    # --------------------------------------------------
    # 6. Combine batches
    # --------------------------------------------------
    text_features = np.concatenate(
        all_embeddings,
        axis=0,
    )

    # --------------------------------------------------
    # 7. L2 normalization
    # --------------------------------------------------
    print("Normalizing embeddings (L2)...")

    norms = np.linalg.norm(
        text_features,
        ord=2,
        axis=1,
        keepdims=True,
    )

    norms = np.where(
        norms == 0,
        1.0,
        norms,
    )

    text_features = (
        text_features / norms
    ).astype(np.float32)

    # --------------------------------------------------
    # 8. Quality check
    # --------------------------------------------------
    print("\n=== EMBEDDING QUALITY CHECK ===")

    num_items, num_dims = text_features.shape

    num_nan = np.isnan(text_features).sum()
    num_inf = np.isinf(text_features).sum()

    post_norms = np.linalg.norm(
        text_features,
        axis=1,
    )

    is_l2_valid = np.allclose(
        post_norms,
        1.0,
        atol=1e-5,
    )

    print(f"Total encoded items: {num_items}")
    print(f"Vector dimensions: {num_dims}")
    print(f"dtype: {text_features.dtype}")
    print(f"NaN values: {num_nan}")
    print(f"Inf values: {num_inf}")
    print(f"L2 normalization valid: {is_l2_valid}")

    if num_items != len(df_mapping):
        raise ValueError(
            "Feature count does not match item mapping."
        )

    if num_dims != 512:
        raise ValueError(
            f"Unexpected feature dimension: {num_dims}"
        )

    if num_nan > 0 or num_inf > 0:
        raise ValueError(
            "Features contain NaN or Inf."
        )

    if not is_l2_valid:
        raise ValueError(
            "L2 normalization check failed."
        )

    # --------------------------------------------------
    # 9. Save
    # --------------------------------------------------
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        OUTPUT_PATH,
        text_features,
    )

    print(
        f"\nSUCCESS: Saved text features to "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()