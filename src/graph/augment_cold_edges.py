# -*- coding: utf-8 -*-
"""
augment_cold_edges.py
=====================
G5 - Mục tiêu #5: Bù liên kết bị thiếu trên đồ thị cực thưa cho item lạnh.

Nguyên lý:
  - Nội dung đa phương thức ĐÃ KHỬ NHIỄU bằng review (cùng công thức với
    DemandGate trong model, phiên bản tĩnh) tạo "chân dung nội dung" mỗi item.
  - Item có degree train thấp (<= ngưỡng) được nối top-k hàng xóm nội dung
    (cosine >= ngưỡng) bằng cạnh item-item có trọng số = cosine.
  - Đồ thị lan truyền: A' = [[0, R], [R^T, S]] — item-item block S trước đó rỗng.
  - Chuẩn hoá đối xứng lại như graph gốc và lưu norm_adj_aug.pt.

Đầu ra:
  - graph/norm_adj_aug.pt + results/cold_edges_report.csv (thống kê cạnh bù).
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from tqdm import tqdm

from configs.config import (
    GRAPH_DIR, NORM_ADJ_PATH, NORM_ADJ_AUG_PATH, COLD_EDGES_REPORT_PATH,
    ITEM_MAPPING_PATH, DEFAULT_CONFIG, save_json,
)
from model.mmgcf import load_modality_features


def build_static_content_embedding(cfg, num_items, demand_active):
    """
    Chân dung nội dung tĩnh của item = mean các modality (L2) có thể có,
    modality được scale theo mức review xác nhận (cosine trực tiếp trong
    không gian 512-d gốc — xấp xỉ tĩnh của DemandGate trong model).
    """
    feats = load_modality_features(cfg, num_items) if cfg.multimodal else {}
    host = [f for m, f in feats.items() if m in ("text", "image", "audio", "video")]
    if not host:
        return None

    host = [f.numpy() if hasattr(f, "numpy") else f for f in host]
    M = len(host)

    if demand_active and "review" in feats:
        r = feats["review"].numpy() if hasattr(feats["review"], "numpy") else feats["review"]
        # w_m = softmax(cosine(r, e_m)) trên các modality host
        cos = np.stack([(r * e).sum(axis=1) for e in host])      # (M, n_items)
        cos = cos - cos.max(axis=0, keepdims=True)
        w = np.exp(cos)
        w = w / np.maximum(w.sum(axis=0, keepdims=True), 1e-12)
        beta = cfg.beta_demand
        host = [
            e * (1.0 + beta * (M * w[m][:, None] - 1.0))
            for m, e in enumerate(host)
        ]

    content = np.stack(host).mean(axis=0)
    norms = np.linalg.norm(content, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (content / norms).astype(np.float32)


def add_cold_edges(content_emb, item_degrees, cfg):
    """Trả về S: scipy CSR (n_items x n_items) cạnh item-item cho item lạnh."""
    n_items = content_emb.shape[0]
    cold_mask = item_degrees <= cfg.cold_deg_threshold
    cold_items = np.where(cold_mask)[0]
    print(f"[COLD-AUG] Item lạnh (degree<={cfg.cold_deg_threshold}): {len(cold_items):,}")

    S = sp.lil_matrix((n_items, n_items), dtype=np.float32)
    rows = []

    # Cosine theo ma trận đầy đủ: (8308 x 512) nhỏ nên tính một lần là đủ.
    sims_full = content_emb @ content_emb.T
    np.fill_diagonal(sims_full, -1.0)

    added = 0
    for i in tqdm(cold_items, desc="Cold edges"):
        row = sims_full[i]
        # chọn hàng xóm cosine cao nhất, lọc ngưỡng
        order = np.argsort(-row)
        count = 0
        for j in order:
            if count >= cfg.cold_k:
                break
            if j == i:
                continue
            c = float(row[j])
            if c < cfg.cold_cos_threshold:
                break
            S[i, j] = c
            S[j, i] = c   # đối xứng: hàng xóm nhận cạnh ngược
            added += 1
            count += 1
            rows.append({"item_idx": i, "neighbor_item_idx": j, "cosine": c})

    print(f"[COLD-AUG] Cạnh item-item thêm: {added:,} (hướng đôi, k={cfg.cold_k}, "
          f"cos>={cfg.cold_cos_threshold})")
    return S.tocsr(), pd.DataFrame(rows)


def build_augmented_adjacency(S, train_df, num_users, num_items):
    """A' = [[0, R], [R^T, S]] -> chuẩn hoá đối xứng -> torch sparse COO."""
    users = train_df["user_idx"].to_numpy(dtype=np.int64)
    items = train_df["item_idx"].to_numpy(dtype=np.int64)
    R = sp.csr_matrix(
        (np.ones(len(users), dtype=np.float32), (users, items)),
        shape=(num_users, num_items), dtype=np.float32,
    )
    R.data[:] = 1.0
    R.eliminate_zeros()

    zero_users = sp.csr_matrix((num_users, num_users), dtype=np.float32)
    adjacency = sp.bmat(
        [[zero_users, R], [R.T.tocsr(), S]], format="csr", dtype=np.float32
    )

    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inv_sqrt = np.zeros_like(degree, dtype=np.float32)
    nz = degree > 0
    inv_sqrt[nz] = 1.0 / np.sqrt(degree[nz])
    D_inv = sp.diags(inv_sqrt, format="csr", dtype=np.float32)
    normalized = (D_inv @ adjacency @ D_inv).tocsr()

    diff = (normalized - normalized.T).tocsr()
    max_diff = float(np.abs(diff.data).max()) if diff.nnz else 0.0
    if max_diff > 1e-5:
        raise ValueError(f"norm_adj_aug không đối xứng: max_diff={max_diff}")
    if not np.isfinite(normalized.data).all():
        raise ValueError("norm_adj_aug chứa NaN/Inf.")

    coo = normalized.tocoo()
    tensor = torch.sparse_coo_tensor(
        torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64)),
        torch.from_numpy(coo.data),
        size=coo.shape, dtype=torch.float32,
    ).coalesce()
    return tensor


def main(cfg=None, force=False):
    cfg = cfg or DEFAULT_CONFIG
    print("=== G5: AUGMENT COLD EDGES (item-item nội dung) ===")

    if NORM_ADJ_AUG_PATH.exists() and not force:
        print(f"[COLD-AUG] Đã có {NORM_ADJ_AUG_PATH.name} — bỏ qua (dùng --force để tính lại).")
        return True

    item_mapping = pd.read_parquet(ITEM_MAPPING_PATH)
    num_items = len(item_mapping)
    train_df = pd.read_parquet(GRAPH_DIR / "train_interactions.parquet")

    train_users = train_df["user_idx"].to_numpy(dtype=np.int64)
    R_counts = sp.csr_matrix(
        (np.ones(len(train_users), dtype=np.float32),
         (train_users, train_df["item_idx"].to_numpy(dtype=np.int64))),
        shape=(train_df["user_idx"].max() + 1, num_items),
    )
    item_degrees = np.asarray(R_counts.sum(axis=0)).ravel()

    content_emb = build_static_content_embedding(cfg, num_items, cfg.use_demand)
    if content_emb is None:
        print("[COLD-AUG] Không có modality nào -> không thể augment. Bỏ qua.")
        return False

    S, report = add_cold_edges(content_emb, item_degrees, cfg)
    if report.empty:
        print("[COLD-AUG] Không thêm cạnh nào (ngưỡng cosine quá cao?)")
        return False

    num_users = num_users_path_len()
    tensor = build_augmented_adjacency(S, train_df, num_users, num_items)

    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, NORM_ADJ_AUG_PATH)
    COLD_EDGES_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(COLD_EDGES_REPORT_PATH, index=False, encoding="utf-8-sig")

    stats = {
        "cold_deg_threshold": cfg.cold_deg_threshold,
        "cold_items": int((item_degrees <= cfg.cold_deg_threshold).sum()),
        "edges_added": int(report.shape[0]),
        "cos_threshold": cfg.cold_cos_threshold,
        "k": cfg.cold_k,
        "avg_cos": float(report["cosine"].mean()) if not report.empty else 0.0,
    }
    save_json(GRAPH_DIR / "cold_edges_meta.json", stats)
    print(f"[COLD-AUG] Đã lưu norm_adj_aug.pt + thống kê: {stats}")
    return True


def num_users_path_len():
    """num_users lấy từ graph_meta.json (nguồn chính thức của đồ thị)."""
    from configs.config import GRAPH_DIR
    import json
    meta_path = GRAPH_DIR / "graph_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError("graph_meta.json không tồn tại — hãy chạy build_bipartite_graph.py trước.")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return int(meta["num_users"])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(force=args.force)
