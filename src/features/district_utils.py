# -*- coding: utf-8 -*-
"""
district_utils.py
=================
G2 - Tiện ích ánh xạ listing -> khu vực (district) cho audio/video cấp khu vực.

Dữ liệu Airbnb Bangkok có 50 khu vực (`neighbourhood_cleansed`) + lat/lng cho
từng listing. Audio ambience và video walking tour chỉ tồn tại ở cấp KHU VỰC
(50 thực thể) — mỗi listing kế thừa embedding của khu vực, tuỳ chọn trộn mượt
k=3 khu gần nhất theo khoảng cách haversine để giảm hiệu ứng ranh giới.

Đầu ra của build_district_table():
    district_names : list[str]           tên khu (theo dữ liệu)
    district_idx   : np.ndarray (n_items,)  khu của từng item (item_idx order)
    centroids      : np.ndarray (K, 2)      (lat, lng) trung bình từng khu
    lat_lng        : np.ndarray (n_items, 2)
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import csv
import gzip
from pathlib import Path

import numpy as np
import pandas as pd

from configs.config import RAW_DIR, ITEM_MAPPING_PATH

LISTINGS_RAW_PATH = RAW_DIR / "listings.csv.gz"


def _read_listing_geo():
    """Đọc (id -> district, lat, lng) từ listings.csv.gz bằng csv thuần."""
    table = {}
    with gzip.open(LISTINGS_RAW_PATH, "rt", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            listing_id = (row.get("id") or "").strip()
            if not listing_id:
                continue
            district = (row.get("neighbourhood_cleansed") or "").strip()
            lat = row.get("latitude") or ""
            lng = row.get("longitude") or ""
            try:
                lat_f, lng_f = float(lat), float(lng)
            except ValueError:
                lat_f = lng_f = float("nan")
            table[listing_id] = (district, lat_f, lng_f)
    return table


def build_district_table():
    """
    Trả về bảng ánh xạ item_idx -> district + toạ độ, theo THỨ TỰ item_mapping.
    """
    mapping = pd.read_parquet(ITEM_MAPPING_PATH)
    mapping = mapping.sort_values("item_idx").reset_index(drop=True)
    mapping["listing_id"] = mapping["listing_id"].astype(str).str.strip()

    geo = _read_listing_geo()

    n = len(mapping)
    district_names = np.empty(n, dtype=object)
    lat_lng = np.full((n, 2), np.nan, dtype=np.float32)

    missing_district = 0
    missing_geo = 0
    for i, listing_id in enumerate(mapping["listing_id"]):
        district, lat, lng = geo.get(listing_id, ("", np.nan, np.nan))
        if not district:
            missing_district += 1
        if np.isnan(lat) or np.isnan(lng):
            missing_geo += 1
        district_names[i] = district
        lat_lng[i] = (lat, lng)

    if missing_district:
        print(f"[DISTRICT] Cảnh báo: {missing_district}/{n} listing thiếu khu vực.")
    if missing_geo:
        print(f"[DISTRICT] Cảnh báo: {missing_geo}/{n} listing thiếu lat/lng.")

    unique_names = sorted({d for d in district_names if d})
    name_to_id = {name: k for k, name in enumerate(unique_names)}
    district_idx = np.array(
        [name_to_id.get(d, -1) for d in district_names], dtype=np.int64
    )

    # Tâm khu vực = trung bình lat/lng các listing thuộc khu (trên TOÀN BỘ raw
    # để khu ít listing trong mapping vẫn có tâm ổn định).
    counts = np.zeros(len(unique_names), dtype=np.int64)
    sums = np.zeros((len(unique_names), 2), dtype=np.float64)
    for listing_id, (district, lat, lng) in geo.items():
        k = name_to_id.get(district)
        if k is not None and np.isfinite(lat) and np.isfinite(lng):
            counts[k] += 1
            sums[k] += (lat, lng)
    centroids = np.where(
        counts[:, None] > 0, sums / np.maximum(counts[:, None], 1), np.nan
    ).astype(np.float32)

    print(f"[DISTRICT] Khu vực: {len(unique_names)} | "
          f"items: {n:,} | tâm khu có toạ độ: {(np.isfinite(centroids).all(axis=1)).sum()}")

    return {
        "district_names": unique_names,
        "district_idx": district_idx,
        "centroids": centroids,
        "lat_lng": lat_lng,
        "item_mapping": mapping,
    }


def haversine_matrix(lat_lng, centroids):
    """
    (n, 2) vs (K, 2) -> (n, K) khoảng cách km (haversine).
    Dùng cho trộn mượt k khu gần nhất.
    """
    lat1 = np.radians(lat_lng[:, 0])[:, None]
    lon1 = np.radians(lat_lng[:, 1])[:, None]
    lat2 = np.radians(centroids[None, :, 0])
    lon2 = np.radians(centroids[None, :, 1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    a = np.clip(a, 0.0, 1.0)
    return 6371.0 * 2.0 * np.arcsin(np.sqrt(a))


def district_to_item_features(district_emb, table, k_nearest=3):
    """
    Truyền embedding khu vực -> embedding từng item.

    district_emb: (K, 512) float32 (đã L2-norm từng khu) — NaN khu thiếu clip.
    k_nearest > 1: trộn mượt theo 1/distance tới k khu gần nhất CÓ embedding.
    Trả về (n_items, 512) float32 L2-norm; item không có khu nào -> vector 0.
    """
    district_idx = table["district_idx"]
    n_items = len(district_idx)
    feat_dim = district_emb.shape[1] if district_emb is not None else 512

    valid_district = np.isfinite(district_emb).all(axis=1)
    out = np.zeros((n_items, feat_dim), dtype=np.float32)

    if k_nearest <= 1:
        ok = valid_district[district_idx]
        out[ok] = district_emb[district_idx[ok]]
    else:
        centroids = table["centroids"]
        lat_lng = table["lat_lng"]
        dists = haversine_matrix(lat_lng, centroids)          # (n, K)
        for i in range(n_items):
            own = district_idx[i]
            d_row = dists[i]
            candidates = [k for k in np.argsort(d_row) if np.isfinite(d_row[k])]
            picked = []
            for k in candidates:
                if valid_district[k]:
                    picked.append(k)
                if len(picked) >= k_nearest:
                    break
            if not picked:
                continue
            # Ưu tiên khu riêng nếu nó có embedding
            if own >= 0 and valid_district[own] and own == picked[0]:
                out[i] = district_emb[own]
            else:
                w = np.array(
                    [1.0 / max(dists[i][k], 0.5) for k in picked], dtype=np.float32
                )
                v = (district_emb[picked] * w[:, None]).sum(axis=0)
                norm = np.linalg.norm(v)
                out[i] = v / norm if norm > 0 else v

    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (out / norms).astype(np.float32)
