import torch
import numpy as np
import pandas as pd
import requests
from io import BytesIO
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from transformers import CLIPProcessor, CLIPVisionModel

BASE_DIR = Path(__file__).resolve().parent.parent.parent
LISTINGS_PATH = BASE_DIR / "data" / "processed" / "bangkok" / "listings_kcore.parquet"
MAPPING_PATH = BASE_DIR / "data" / "processed" / "bangkok" / "features" / "item_mapping.parquet"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "bangkok" / "features" / "image.npy"

MODEL_NAME = "openai/clip-vit-base-patch32"
BATCH_SIZE = 64
REQUEST_TIMEOUT = 5
NUM_DOWNLOAD_WORKERS = 20  # Giữ 20 workers an toàn cho mạng

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
    
    print("Loading datasets...")
    df_listings = pd.read_parquet(LISTINGS_PATH)
    df_mapping = pd.read_parquet(MAPPING_PATH)
    
    df_merged = pd.merge(
        df_mapping, 
        df_listings, 
        left_on="listing_id", 
        right_on="id", 
        how="left"
    )
    df_merged = df_merged.sort_values("item_idx").reset_index(drop=True)
    urls = df_merged["picture_url"].tolist()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print(f"Loading CLIP Vision model: {MODEL_NAME}...")
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPVisionModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    
    print(f"Processing {len(urls)} images in chunks of {BATCH_SIZE}...")
    all_embeddings = []
    
    # Sử dụng ThreadPool cố định cho toàn bộ chương trình
    with ThreadPoolExecutor(max_workers=NUM_DOWNLOAD_WORKERS) as executor:
        # Chia nhỏ và duyệt theo từng batch để không bị tràn RAM
        for i in tqdm(range(0, len(urls), BATCH_SIZE)):
            batch_urls = urls[i : i + BATCH_SIZE]
            
            # Tải song song 64 ảnh của batch hiện tại bằng 20 workers
            batch_images = list(executor.map(download_single_image, batch_urls))
            
            # Đẩy qua mô hình CLIP
            inputs = processor(images=batch_images, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                embeddings = outputs.pooler_output.cpu().numpy()
                
            all_embeddings.append(embeddings)
            
            # Xóa bỏ danh sách ảnh của batch vừa xong để giải phóng bộ nhớ ngay lập tức
            del batch_images
            
    image_features = np.concatenate(all_embeddings, axis=0)
    
    print("\nNormalizing embeddings (L2)...")
    norms = np.linalg.norm(image_features, ord=2, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    image_features = image_features / norms
    
    print("\n=== EMBEDDING QUALITY CHECK ===")
    num_items, num_dims = image_features.shape
    num_nan = np.isnan(image_features).sum()
    post_norms = np.linalg.norm(image_features, ord=2, axis=1)
    is_l2_valid = np.allclose(post_norms, 1.0, atol=1e-5)
    
    print(f"Total encoded items: {num_items}")
    print(f"Vector dimensions: {num_dims}")
    print(f"Missing/NaN values: {num_nan}")
    print(f"L2 Normalization valid (Norm = 1.0): {is_l2_valid}")
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_PATH, image_features)
    print(f"\nSUCCESS: Saved normalized embeddings to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
