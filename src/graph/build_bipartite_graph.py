# -*- coding: utf-8 -*-
"""
build_bipartite_graph.py
========================
Bước 3: Xây dựng Đồ thị 2 phía (User–Item Bipartite Graph) cho MMGCF.

Đầu vào:
  - train_interactions.parquet  (user_idx, item_idx)
  - graph_meta từ user/item mapping

Đầu ra (lưu tại data/processed/bangkok/graph/):
  - norm_adj.pt      : Ma trận kề chuẩn hóa D^(-1/2) * A * D^(-1/2)
                       dưới dạng torch.sparse_coo_tensor, kích thước (N x N)
                       N = num_users + num_items = 4411 + 8308 = 12719
  - edge_index.pt    : Cạnh 2 chiều dưới dạng LongTensor (2, 2 * num_train_edges)
  - graph_meta.json  : Metadata thống kê của đồ thị
"""

import json

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "processed" / "bangkok"
FEATURES_DIR = DATA_DIR / "features"
GRAPH_DIR = DATA_DIR / "graph"

TRAIN_PATH = GRAPH_DIR / "train_interactions.parquet"
USER_MAPPING_PATH = FEATURES_DIR / "user_mapping.parquet"
ITEM_MAPPING_PATH = FEATURES_DIR / "item_mapping.parquet"
TEXT_FEATURES_PATH = FEATURES_DIR / "text_features.npy"
IMAGE_FEATURES_PATH = FEATURES_DIR / "image_features.npy"

NORM_ADJ_OUTPUT = GRAPH_DIR / "norm_adj.pt"
EDGE_INDEX_OUTPUT = GRAPH_DIR / "edge_index.pt"
META_OUTPUT = GRAPH_DIR / "graph_meta.json"


# ─────────────────────────────────────────────────────────────
# Step helpers
# ─────────────────────────────────────────────────────────────

def load_inputs():
    for path, name in [
        (TRAIN_PATH, "Train interactions"),
        (USER_MAPPING_PATH, "User mapping"),
        (ITEM_MAPPING_PATH, "Item mapping"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")

    train = pd.read_parquet(TRAIN_PATH)
    user_map = pd.read_parquet(USER_MAPPING_PATH)
    item_map = pd.read_parquet(ITEM_MAPPING_PATH)

    num_users = len(user_map)
    num_items = len(item_map)

    print("=== INPUTS ===")
    print(f"Train interactions: {len(train)}")
    print(f"Num users (N_u): {num_users}")
    print(f"Num items (N_i): {num_items}")
    print(f"Total nodes (N): {num_users + num_items}")

    # Basic column validation
    if "user_idx" not in train.columns or "item_idx" not in train.columns:
        raise ValueError("Train interactions must contain 'user_idx' and 'item_idx'.")

    # Validate index ranges
    if train["user_idx"].min() < 0 or train["user_idx"].max() >= num_users:
        raise ValueError(
            f"user_idx out of range [0, {num_users - 1}]: "
            f"got [{train['user_idx'].min()}, {train['user_idx'].max()}]"
        )
    if train["item_idx"].min() < 0 or train["item_idx"].max() >= num_items:
        raise ValueError(
            f"item_idx out of range [0, {num_items - 1}]: "
            f"got [{train['item_idx'].min()}, {train['item_idx'].max()}]"
        )

    return train, num_users, num_items


def build_interaction_matrix(train_df, num_users, num_items):
    """
    Xây dựng ma trận tương tác R ∈ {0,1}^(num_users x num_items)
    dưới dạng scipy sparse CSR matrix (tiết kiệm bộ nhớ).
    """
    user_ids = train_df["user_idx"].to_numpy(dtype=np.int32)
    item_ids = train_df["item_idx"].to_numpy(dtype=np.int32)
    values = np.ones(len(train_df), dtype=np.float32)

    R = sp.csr_matrix(
        (values, (user_ids, item_ids)),
        shape=(num_users, num_items),
        dtype=np.float32,
    )

    print("\n=== INTERACTION MATRIX R ===")
    print(f"Shape: {R.shape}")
    print(f"Non-zero entries: {R.nnz}")
    print(f"Density: {R.nnz / (num_users * num_items):.6f}")

    return R


def build_full_adjacency(R, num_users, num_items):
    """
    Xây dựng ma trận kề toàn cục (block matrix):

        A = [  0   R  ]   kích thước (N x N)
            [ R^T  0  ]

    N = num_users + num_items
    """
    zero_uu = sp.csr_matrix((num_users, num_users), dtype=np.float32)
    zero_ii = sp.csr_matrix((num_items, num_items), dtype=np.float32)

    A = sp.bmat(
        [[zero_uu, R],
         [R.T,     zero_ii]],
        format="csr",
        dtype=np.float32,
    )

    N = num_users + num_items
    print("\n=== FULL ADJACENCY MATRIX A ===")
    print(f"Shape: {A.shape}  (expected: {N} x {N})")
    print(f"Non-zero entries: {A.nnz}  (expected: {2 * R.nnz})")
    assert A.shape == (N, N), "Adjacency matrix shape mismatch."
    assert A.nnz == 2 * R.nnz, "Adjacency matrix nnz mismatch."

    return A


def normalize_adjacency(A):
    """
    Chuẩn hóa đối xứng (Symmetric Laplacian Normalization):

        Ã = D^(-1/2) * A * D^(-1/2)

    Với D là ma trận bậc (degree matrix), D_{ii} = Σ_j A_{ij}.
    Các đỉnh có bậc = 0 được gán giá trị nghịch đảo = 0 (tránh chia cho 0).
    """
    # Tính bậc của từng đỉnh (số hàng xóm)
    degree = np.array(A.sum(axis=1)).flatten()  # shape (N,)

    # D^(-1/2): nghịch đảo căn bậc hai của degree, với degree=0 → 0
    d_inv_sqrt = np.zeros_like(degree, dtype=np.float32)
    nonzero_mask = degree > 0
    d_inv_sqrt[nonzero_mask] = 1.0 / np.sqrt(degree[nonzero_mask])

    # Chuyển thành ma trận đường chéo thưa (sparse diagonal matrix)
    D_inv_sqrt = sp.diags(d_inv_sqrt, format="csr", dtype=np.float32)

    # Ã = D^(-1/2) * A * D^(-1/2)
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt

    # Thống kê
    values = A_norm.data
    print("\n=== NORMALIZED ADJACENCY Ã ===")
    print(f"Shape: {A_norm.shape}")
    print(f"Non-zero entries: {A_norm.nnz}")
    print(f"Value range: [{values.min():.6f}, {values.max():.6f}]")
    print(f"Mean value (non-zero): {values.mean():.6f}")
    print(f"Isolated nodes (degree=0): {(~nonzero_mask).sum()}")

    # Kiểm tra không có NaN hoặc Inf
    if np.isnan(values).any():
        raise ValueError("NaN detected in normalized adjacency matrix.")
    if np.isinf(values).any():
        raise ValueError("Inf detected in normalized adjacency matrix.")

    return A_norm, degree


def sparse_to_torch_coo(sp_mat):
    """
    Chuyển scipy sparse matrix sang torch.sparse_coo_tensor.
    """
    sp_coo = sp_mat.tocoo().astype(np.float32)

    indices = torch.from_numpy(
        np.vstack([sp_coo.row, sp_coo.col]).astype(np.int64)
    )  # shape (2, nnz)
    values = torch.from_numpy(sp_coo.data)
    size = torch.Size(sp_coo.shape)

    return torch.sparse_coo_tensor(indices, values, size).coalesce()


def build_edge_index(train_df, num_users):
    """
    Xây dựng edge_index (COO format) với đánh số node toàn cục:

    - Node User:  u           ∈ [0, num_users - 1]
    - Node Item:  num_users + i ∈ [num_users, num_users + num_items - 1]

    Cạnh 2 chiều: (u → i_global) và (i_global → u)
    Output: LongTensor kích thước (2, 2 * num_train_edges)
    """
    user_nodes = train_df["user_idx"].to_numpy(dtype=np.int64)
    item_nodes = train_df["item_idx"].to_numpy(dtype=np.int64) + num_users

    # Cạnh xuôi: u → i
    src_fwd = user_nodes
    dst_fwd = item_nodes

    # Cạnh ngược: i → u
    src_bwd = item_nodes
    dst_bwd = user_nodes

    src = np.concatenate([src_fwd, src_bwd])
    dst = np.concatenate([dst_fwd, dst_bwd])

    edge_index = torch.from_numpy(np.vstack([src, dst]))

    print("\n=== EDGE INDEX ===")
    print(f"Shape: {edge_index.shape}  (2 x {edge_index.shape[1]})")
    print(f"Expected: 2 x {2 * len(train_df)}")
    assert edge_index.shape == (2, 2 * len(train_df)), "edge_index shape mismatch."

    return edge_index


def save_outputs(norm_adj_tensor, edge_index, graph_meta):
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    torch.save(norm_adj_tensor, NORM_ADJ_OUTPUT)
    torch.save(edge_index, EDGE_INDEX_OUTPUT)

    with open(META_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(graph_meta, f, indent=2)

    print("\n=== SAVED FILES ===")
    print(f"Normalized adjacency matrix : {NORM_ADJ_OUTPUT}")
    print(f"Edge index                  : {EDGE_INDEX_OUTPUT}")
    print(f"Graph metadata              : {META_OUTPUT}")


def main():
    # ── 1. Load inputs ──────────────────────────────────────────
    train_df, num_users, num_items = load_inputs()
    num_nodes = num_users + num_items
    num_train_edges = len(train_df)

    # ── 2. Ma trận tương tác R ──────────────────────────────────
    R = build_interaction_matrix(train_df, num_users, num_items)

    # ── 3. Ma trận kề toàn cục A ────────────────────────────────
    A = build_full_adjacency(R, num_users, num_items)

    # ── 4. Chuẩn hóa: Ã = D^(-1/2) A D^(-1/2) ─────────────────
    A_norm, degree = normalize_adjacency(A)

    # ── 5. Chuyển sang PyTorch sparse tensor ────────────────────
    print("\n=== CONVERTING TO TORCH SPARSE TENSOR ===")
    norm_adj_tensor = sparse_to_torch_coo(A_norm)
    print(f"Sparse tensor shape: {norm_adj_tensor.shape}")
    print(f"Sparse tensor nnz  : {norm_adj_tensor._nnz()}")

    # ── 6. Tạo edge_index 2 chiều ───────────────────────────────
    edge_index = build_edge_index(train_df, num_users)

    # ── 7. Kiểm tra tính đối xứng (spot-check) ──────────────────
    print("\n=== SYMMETRY CHECK (spot check) ===")
    coo = A_norm.tocoo()
    A_T = A_norm.T
    diff = (A_norm - A_T).data
    max_diff = np.abs(diff).max() if len(diff) > 0 else 0.0
    print(f"Max |A - A^T|: {max_diff:.2e}  (should be ~0)")
    if max_diff > 1e-5:
        raise ValueError(f"Normalized adjacency matrix is not symmetric! Max diff: {max_diff}")
    print("Symmetry check: PASSED")

    # ── 8. Lấy thông số đặc trưng đa phương thức ─────────────────
    text_dim, image_dim = 0, 0
    if TEXT_FEATURES_PATH.exists():
        text_feat = np.load(TEXT_FEATURES_PATH, mmap_mode="r")
        text_dim = text_feat.shape[1]
    if IMAGE_FEATURES_PATH.exists():
        img_feat = np.load(IMAGE_FEATURES_PATH, mmap_mode="r")
        image_dim = img_feat.shape[1]

    # ── 9. Lưu metadata ──────────────────────────────────────────
    graph_meta = {
        "num_users": num_users,
        "num_items": num_items,
        "num_nodes": num_nodes,
        "num_train_edges": num_train_edges,
        "num_edges_bidirectional": 2 * num_train_edges,
        "graph_density": round(num_train_edges / (num_users * num_items), 8),
        "text_feature_dim": text_dim,
        "image_feature_dim": image_dim,
        "user_node_range": [0, num_users - 1],
        "item_node_range": [num_users, num_nodes - 1],
        "norm_adj_nnz": int(A_norm.nnz),
        "isolated_nodes": int((degree == 0).sum()),
        "split_strategy": "leave_one_out",
        "split_counts": {
            "train": num_train_edges,
            "val": 4408,
            "test": 4411
        },
    }

    print("\n=== GRAPH METADATA ===")
    for k, v in graph_meta.items():
        print(f"  {k}: {v}")

    # ── 10. Lưu files ra đĩa ─────────────────────────────────────
    save_outputs(norm_adj_tensor, edge_index, graph_meta)


if __name__ == "__main__":
    main()
