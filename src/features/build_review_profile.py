# -*- coding: utf-8 -*-
"""
build_review_profile.py
=======================
G1 - Mục tiêu #3: Xây "modality bên cầu" từ review thực tế của khách.

3 giai đoạn trong 1 script:
  1) ENCODE: comments của 693.647 reviews -> CLIP text embedding (512-d).
     Encode theo chunk + RESUME (memmap fp16) để chạy dở vẫn tiếp tục được.
  2) AGGREGATE: gộp embedding review theo listing bằng time-decay mean
     (review gần đây nặng hơn) + cap số review gần nhất mỗi listing.
  3) GAP REPORT: đo độ tương đồng profile-review vs text/image bên cung ->
     bằng chứng định lượng cho hiện tượng "thổi phồng/quảng cáo" (cos thấp).

Đầu ra (data/processed/bangkok/features/):
  - review_embeddings.f16.npy   : (n_reviews, 512) fp16, có resume
  - review_embeddings_meta.json : trạng thái resume + tham số
  - review_profile.npy          : (n_items, 512) float32, L2-norm
  - review_count.npy            : (n_items,) int32
Kết quả (results/):
  - review_gap_report.csv       : cosine từng item + thống kê tổng
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import CLIPTokenizer, CLIPModel

from configs.config import (
    DEFAULT_CONFIG,
    REVIEWS_RAW_PATH, ITEM_MAPPING_PATH,
    TEXT_FEATURES_PATH, IMAGE_FEATURES_PATH,
    REVIEW_EMB_PATH, REVIEW_META_PATH, REVIEW_PROFILE_PATH, REVIEW_COUNT_PATH,
    REVIEW_GAP_REPORT_PATH,
)
from features.encoder_utils import extract_feature_tensor

FEATURE_DIM = 512
MODEL_NAME = "openai/clip-vit-base-patch32"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ------------------------------------------------------------------
# 1) ENCODE
# ------------------------------------------------------------------
def encode_reviews(df, device, batch_size, chunk_size, limit, resume=True):
    """Encode toàn bộ comments thành memmap fp16 (n_reviews, 512)."""
    n = len(df)
    if limit > 0:
        n = min(n, limit)

    meta = {}
    done = 0
    if REVIEW_META_PATH.exists() and resume:
        try:
            meta = json.loads(REVIEW_META_PATH.read_text(encoding="utf-8"))
            if meta.get("limit", -1) == limit:
                done = int(meta.get("num_done", 0))
        except (json.JSONDecodeError, OSError):
            done = 0

    emb = np.lib.format.open_memmap(
        REVIEW_EMB_PATH, mode="w+" if done == 0 else "r+",
        dtype=np.float16, shape=(n, FEATURE_DIM),
    )

    if done >= n:
        print(f"[ENCODE] Đã encode sẵn {done:,}/{n:,} reviews — bỏ qua.")
        return emb, n

    tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
    use_fp16 = device.type == "cuda"
    if use_fp16:
        model = model.half()

    comments = df["comments"].iloc[:n].astype(str).tolist()
    t0 = time.perf_counter()

    for start in tqdm(range(done, n, chunk_size), desc="Encode reviews"):
        end = min(start + chunk_size, n)
        with torch.inference_mode():
            for b0 in range(start, end, batch_size):
                b1 = min(b0 + batch_size, end)
                inputs = tokenizer(
                    comments[b0:b1], padding=True, truncation=True,
                    max_length=77, return_tensors="pt",
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                out = extract_feature_tensor(model.get_text_features(**inputs))
                emb[b0:b1] = out.to(torch.float16).cpu().numpy()

        meta.update({"num_done": end, "limit": limit})
        REVIEW_META_PATH.write_text(json.dumps(meta), encoding="utf-8")

    dt = time.perf_counter() - t0
    print(f"[ENCODE] Xong {n:,} reviews trong {dt/60:.1f} phút "
          f"({n/max(dt,1e-9):,.0f} reviews/s, device={device})")
    return emb, n


# ------------------------------------------------------------------
# 2) AGGREGATE (time-decay mean, cap review gần nhất)
# ------------------------------------------------------------------
def aggregate_profiles(df, emb_memmap, n_encoded, item_mapping):
    """
    Trả về:
      profile : (n_items, 512) float32 L2-norm
      counts  : (n_items,) int32
    """
    d = df.iloc[:n_encoded].copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    max_date = d["date"].max()

    # Trọng số time-decay theo NĂM tuổi review (tính từ review mới nhất của dataset)
    days = (max_date - d["date"]).dt.days.fillna(0).clip(lower=0)
    years = (days / 365.25).to_numpy(dtype=np.float32)
    weights = np.power(DEFAULT_CONFIG.review_decay, years)

    d = d.reset_index(drop=True)
    d["listing_id"] = d["listing_id"].astype(str).str.strip()
    item_mapping = item_mapping.copy()
    item_mapping["listing_id"] = item_mapping["listing_id"].astype(str).str.strip()

    lmap = dict(zip(item_mapping["listing_id"], item_mapping["item_idx"]))
    d["item_idx"] = d["listing_id"].map(lmap)
    d = d.dropna(subset=["item_idx"])
    d["item_idx"] = d["item_idx"].astype(np.int64)

    # Chỉ giữ review gần nhất mỗi listing (cap)
    d = d.sort_values(["item_idx", "date"], ascending=[True, False], kind="stable")
    d = d.groupby("item_idx", sort=False).head(DEFAULT_CONFIG.review_cap)

    n_items = len(item_mapping)
    profile = np.zeros((n_items, FEATURE_DIM), dtype=np.float32)
    counts = np.zeros(n_items, dtype=np.int32)

    grouped = d.groupby("item_idx", sort=True)
    for item_idx, group in tqdm(grouped, desc="Aggregate", total=grouped.ngroups):
        idx_rows = group.index.to_numpy()
        w = weights[idx_rows]
        w = w / max(w.sum(), 1e-12)
        e = np.asarray(emb_memmap[idx_rows], dtype=np.float32)
        profile[item_idx] = (e * w[:, None]).sum(axis=0)
        counts[item_idx] = len(idx_rows)

    norms = np.linalg.norm(profile, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    profile = profile / norms

    covered = int((counts > 0).sum())
    print(f"[AGGREGATE] Listings có review: {covered:,}/{n_items:,} "
          f"({covered/n_items:.1%}) | review dùng: {int(counts.sum()):,}")
    return profile, counts


# ------------------------------------------------------------------
# 3) GAP REPORT (bằng chứng hiện tượng thổi phồng)
# ------------------------------------------------------------------
def gap_report(profile, counts, out_path: Path):
    text = np.load(TEXT_FEATURES_PATH)
    image = np.load(IMAGE_FEATURES_PATH)

    cos_text = (profile * text).sum(axis=1)
    cos_image = (profile * image).sum(axis=1)

    rep = pd.DataFrame({
        "cos_review_vs_text": cos_text,
        "cos_review_vs_image": cos_image,
        "review_count": counts,
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rep.to_csv(out_path, index=False, encoding="utf-8-sig")

    valid = counts > 0
    print("\n=== SUPPLY-DEMAND GAP REPORT (bằng chứng mục tiêu #3) ===")
    for name, arr in [("review vs TEXT host", cos_text), ("review vs IMAGE host", cos_image)]:
        v = arr[valid]
        print(f"--- {name} ---")
        print(f"  mean={v.mean():.4f} median={np.median(v):.4f} "
              f"p10={np.percentile(v,10):.4f} p25={np.percentile(v,25):.4f} "
              f"p75={np.percentile(v,75):.4f}")
        print(f"  share cos<0.20: {(v<0.2).mean():.1%} | share cos<0.10: {(v<0.1).mean():.1%}")
    print(f"Đã lưu chi tiết: {out_path}")


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main(limit=0, resume=True, skip_gap=False):
    print("=== G1: BUILD REVIEW PROFILE (demand-side) ===")
    device = get_device()
    print(f"Device: {device}")

    item_mapping = pd.read_parquet(ITEM_MAPPING_PATH)
    n_items = len(item_mapping)

    reviews = pd.read_csv(REVIEWS_RAW_PATH)
    reviews["comments"] = reviews["comments"].fillna("").astype(str).str.strip()
    reviews = reviews[reviews["comments"].str.len() > 0].reset_index(drop=True)
    print(f"Reviews có nội dung: {len(reviews):,}")

    batch_size = 256 if device.type == "cuda" else 64
    chunk_size = DEFAULT_CONFIG.review_chunk

    emb, n_encoded = encode_reviews(
        reviews, device, batch_size, chunk_size,
        limit=limit or DEFAULT_CONFIG.limit_reviews, resume=resume,
    )

    print("\n=== AGGREGATE PROFILE (time-decay mean) ===")
    profile, counts = aggregate_profiles(reviews, emb, n_encoded, item_mapping)

    REVIEW_PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(REVIEW_PROFILE_PATH, profile.astype(np.float32))
    np.save(REVIEW_COUNT_PATH, counts)

    if not skip_gap and TEXT_FEATURES_PATH.exists() and IMAGE_FEATURES_PATH.exists():
        gap_report(profile, counts, REVIEW_GAP_REPORT_PATH)

    print(f"\nSUCCESS: review_profile -> {REVIEW_PROFILE_PATH}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--skip-gap", action="store_true")
    args = parser.parse_args()
    main(limit=args.limit, resume=not args.no_resume, skip_gap=args.skip_gap)
