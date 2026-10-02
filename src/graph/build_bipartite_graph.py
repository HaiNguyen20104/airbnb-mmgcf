# -*- coding: utf-8 -*-
"""
build_bipartite_graph.py
========================
Bước 3: Xây dựng User–Item Bipartite Graph cho MMGCF.

Graph được xây từ TRAIN interactions בלבד để tránh leakage từ validation/test.
Mỗi cặp (user_idx, item_idx) là một tương tác duy nhất và tạo một cạnh.
Nếu dữ liệu gốc có nhiều review cho cùng user-listing, cần gộp cặp đó thành
một tương tác trước bước split; không coi mỗi review là một cạnh riêng.

Đầu vào:
  - graph/train_interactions.parquet
  - graph/val_interactions.parquet
  - graph/test_interactions.parquet
  - features/user_mapping.parquet
  - features/item_mapping.parquet
  - features/text_features.npy
  - features/image_features.npy

Đầu ra:
  - graph/norm_adj.pt
  - graph/edge_index.pt
  - graph/graph_meta.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


# ---------------------------------------------------------------------
# Paths / configuration
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "processed" / "bangkok"
FEATURES_DIR = DATA_DIR / "features"
GRAPH_DIR = DATA_DIR / "graph"

TRAIN_PATH = GRAPH_DIR / "train_interactions.parquet"
VAL_PATH = GRAPH_DIR / "val_interactions.parquet"
TEST_PATH = GRAPH_DIR / "test_interactions.parquet"
USER_MAPPING_PATH = FEATURES_DIR / "user_mapping.parquet"
ITEM_MAPPING_PATH = FEATURES_DIR / "item_mapping.parquet"
TEXT_FEATURES_PATH = FEATURES_DIR / "text_features.npy"
IMAGE_FEATURES_PATH = FEATURES_DIR / "image_features.npy"

NORM_ADJ_OUTPUT = GRAPH_DIR / "norm_adj.pt"
EDGE_INDEX_OUTPUT = GRAPH_DIR / "edge_index.pt"
META_OUTPUT = GRAPH_DIR / "graph_meta.json"

EXPECTED_FEATURE_DIM = 512


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------
def validate_mapping(mapping, idx_col, id_col, name):
    """Validate one mapping table; row order need not match index order."""
    required = {idx_col, id_col}
    missing = required - set(mapping.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}.")
    if mapping.empty:
        raise ValueError(f"{name} is empty.")
    if mapping[[idx_col, id_col]].isna().any().any():
        raise ValueError(f"{name} contains missing values.")

    ids = mapping[id_col].astype("string").str.strip()
    if ids.eq("").any():
        raise ValueError(f"{name} contains empty IDs.")
    if ids.duplicated().any():
        raise ValueError(f"{name} contains duplicate {id_col}.")

    indices = pd.to_numeric(mapping[idx_col], errors="coerce")
    if indices.isna().any():
        raise ValueError(f"{name}.{idx_col} contains non-numeric values.")
    values = indices.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise ValueError(f"{name}.{idx_col} must contain finite integer values.")
    if mapping[idx_col].duplicated().any():
        raise ValueError(f"{name} contains duplicate {idx_col}.")

    expected = np.arange(len(mapping), dtype=np.int64)
    actual = np.sort(indices.to_numpy(dtype=np.int64))
    if not np.array_equal(actual, expected):
        raise ValueError(
            f"{name}.{idx_col} must contain each index from 0 to {len(mapping) - 1}."
        )


def validate_mappings(user_map, item_map):
    validate_mapping(user_map, "user_idx", "reviewer_id", "user_mapping")
    validate_mapping(item_map, "item_idx", "listing_id", "item_mapping")
    print(f"[OK] User mapping: {len(user_map):,} users")
    print(f"[OK] Item mapping: {len(item_map):,} items")


def validate_features(num_items):
    """Validate modality features align by item count and have finite values."""
    for name, path in [
        ("Text features", TEXT_FEATURES_PATH),
        ("Image features", IMAGE_FEATURES_PATH),
    ]:
        features = np.load(path, mmap_mode="r")
        if features.ndim != 2:
            raise ValueError(f"{name} must be 2D; got shape {features.shape}.")
        if features.shape != (num_items, EXPECTED_FEATURE_DIM):
            raise ValueError(
                f"{name} shape is {features.shape}; expected "
                f"({num_items}, {EXPECTED_FEATURE_DIM})."
            )
        if not np.isfinite(features).all():
            raise ValueError(f"{name} contains NaN or Inf.")
        print(f"[OK] {name}: shape {features.shape}, dtype={features.dtype}")


def validate_split(df, name, num_users, num_items, allow_empty=False):
    """Validate mapped user-item pairs for one split."""
    required = {"user_idx", "item_idx"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}.")
    if df.empty and not allow_empty:
        raise ValueError(f"{name} interactions are empty.")
    if df[["user_idx", "item_idx"]].isna().any().any():
        raise ValueError(f"{name} has missing user_idx/item_idx.")

    for col, upper in [("user_idx", num_users), ("item_idx", num_items)]:
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.isna().any():
            raise ValueError(f"{name}.{col} contains non-numeric values.")
        arr = numeric.to_numpy(dtype=np.float64)
        if not np.isfinite(arr).all() or not np.equal(arr, np.floor(arr)).all():
            raise ValueError(f"{name}.{col} must contain finite integers.")
        if (numeric < 0).any() or (numeric >= upper).any():
            raise ValueError(f"{name}.{col} values must be in [0, {upper - 1}].")

    duplicates = df.duplicated(["user_idx", "item_idx"])
    if duplicates.any():
        raise ValueError(
            f"{name} has {int(duplicates.sum())} duplicate user-item pairs. "
            "Deduplicate pairs before splitting; each pair should be one interaction."
        )


def load_inputs():
    required_files = [
        (TRAIN_PATH, "Train interactions"),
        (VAL_PATH, "Validation interactions"),
        (TEST_PATH, "Test interactions"),
        (USER_MAPPING_PATH, "User mapping"),
        (ITEM_MAPPING_PATH, "Item mapping"),
        (TEXT_FEATURES_PATH, "Text features"),
        (IMAGE_FEATURES_PATH, "Image features"),
    ]
    for path, name in required_files:
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")

    train = pd.read_parquet(TRAIN_PATH)
    val = pd.read_parquet(VAL_PATH)
    test = pd.read_parquet(TEST_PATH)
    user_map = pd.read_parquet(USER_MAPPING_PATH)
    item_map = pd.read_parquet(ITEM_MAPPING_PATH)

    validate_mappings(user_map, item_map)
    num_users, num_items = len(user_map), len(item_map)
    validate_features(num_items)

    validate_split(train, "Train", num_users, num_items)
    validate_split(val, "Val", num_users, num_items, allow_empty=True)
    validate_split(test, "Test", num_users, num_items, allow_empty=True)

    # Ensure no pair leaks across splits.
    train_pairs = set(map(tuple, train[["user_idx", "item_idx"]].to_numpy()))
    val_pairs = set(map(tuple, val[["user_idx", "item_idx"]].to_numpy()))
    test_pairs = set(map(tuple, test[["user_idx", "item_idx"]].to_numpy()))
    if train_pairs & val_pairs:
        raise ValueError("Leakage detected: Train and Val contain the same user-item pair.")
    if train_pairs & test_pairs:
        raise ValueError("Leakage detected: Train and Test contain the same user-item pair.")
    if val_pairs & test_pairs:
        raise ValueError("Leakage detected: Val and Test contain the same user-item pair.")

    print("\n=== SPLIT SUMMARY ===")
    print(f"Train interactions: {len(train):,}")
    print(f"Val interactions  : {len(val):,}")
    print(f"Test interactions : {len(test):,}")
    print(f"Users             : {num_users:,}")
    print(f"Items             : {num_items:,}")
    print(f"Total nodes       : {num_users + num_items:,}")

    return train, val, test, num_users, num_items


# ---------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------
def build_interaction_matrix(train, num_users, num_items):
    """Build sparse binary interaction matrix R with shape (num_users, num_items)."""
    user_ids = train["user_idx"].to_numpy(dtype=np.int64)
    item_ids = train["item_idx"].to_numpy(dtype=np.int64)

    # validate_split already rejects duplicate pairs; R is binary by definition.
    values = np.ones(len(train), dtype=np.float32)
    matrix = sp.csr_matrix(
        (values, (user_ids, item_ids)),
        shape=(num_users, num_items),
        dtype=np.float32,
    )
    matrix.data[:] = 1.0
    matrix.eliminate_zeros()

    print("\n=== INTERACTION MATRIX R ===")
    print(f"Shape       : {matrix.shape}")
    print(f"Nonzero edges: {matrix.nnz:,}")
    print(f"Density     : {matrix.nnz / (num_users * num_items):.8f}")
    return matrix


def build_full_adjacency(R, num_users, num_items):
    """Build A = [[0, R], [R.T, 0]]."""
    zero_users = sp.csr_matrix((num_users, num_users), dtype=np.float32)
    zero_items = sp.csr_matrix((num_items, num_items), dtype=np.float32)
    adjacency = sp.bmat(
        [[zero_users, R], [R.T, zero_items]],
        format="csr",
        dtype=np.float32,
    )

    expected_shape = (num_users + num_items, num_users + num_items)
    if adjacency.shape != expected_shape:
        raise ValueError(
            f"Adjacency shape {adjacency.shape} != expected {expected_shape}."
        )
    if adjacency.nnz != 2 * R.nnz:
        raise ValueError(
            f"Adjacency nnz {adjacency.nnz} != expected {2 * R.nnz}."
        )
    print("\n=== ADJACENCY MATRIX A ===")
    print(f"Shape        : {adjacency.shape}")
    print(f"Nonzero edges: {adjacency.nnz:,}")
    return adjacency


def normalize_adjacency(adjacency):
    """Compute symmetric normalized adjacency D^(-1/2) A D^(-1/2)."""
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inv_sqrt_degree = np.zeros_like(degree, dtype=np.float32)
    nonzero = degree > 0
    inv_sqrt_degree[nonzero] = 1.0 / np.sqrt(degree[nonzero])

    D_inv_sqrt = sp.diags(inv_sqrt_degree, format="csr", dtype=np.float32)
    normalized = (D_inv_sqrt @ adjacency @ D_inv_sqrt).tocsr()

    if normalized.nnz == 0:
        raise ValueError("Normalized adjacency has no edges.")
    if not np.isfinite(normalized.data).all():
        raise ValueError("Normalized adjacency contains NaN or Inf.")

    difference = (normalized - normalized.T).tocsr()
    max_difference = float(np.abs(difference.data).max()) if difference.nnz else 0.0
    if max_difference > 1e-5:
        raise ValueError(
            f"Normalized adjacency is not symmetric; max diff={max_difference}."
        )

    print("\n=== NORMALIZED ADJACENCY ===")
    print(f"Shape         : {normalized.shape}")
    print(f"Nonzero values: {normalized.nnz:,}")
    print(f"Isolated nodes: {int((~nonzero).sum()):,}")
    print(f"Symmetry max diff: {max_difference:.2e}")
    return normalized, degree


def sparse_to_torch_coo(matrix):
    """Convert SciPy sparse matrix to coalesced PyTorch sparse COO tensor."""
    coo = matrix.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((coo.row, coo.col)).astype(np.int64)
    )
    values = torch.from_numpy(coo.data)
    return torch.sparse_coo_tensor(
        indices, values, size=coo.shape, dtype=torch.float32
    ).coalesce()


def build_edge_index(train, num_users):
    """Build bidirectional edge_index; item node IDs are offset by num_users."""
    users = train["user_idx"].to_numpy(dtype=np.int64)
    items = train["item_idx"].to_numpy(dtype=np.int64) + num_users

    src = np.concatenate((users, items))
    dst = np.concatenate((items, users))
    edge_index = torch.from_numpy(np.vstack((src, dst))).long()

    expected_shape = (2, 2 * len(train))
    if tuple(edge_index.shape) != expected_shape:
        raise ValueError(
            f"edge_index shape {tuple(edge_index.shape)} != {expected_shape}."
        )
    print("\n=== EDGE INDEX ===")
    print(f"Shape: {tuple(edge_index.shape)}")
    return edge_index


def save_outputs(norm_adj_tensor, edge_index, metadata):
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(norm_adj_tensor, NORM_ADJ_OUTPUT)
    torch.save(edge_index, EDGE_INDEX_OUTPUT)
    with open(META_OUTPUT, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)

    print("\n=== SAVED FILES ===")
    print(f"norm_adj.pt    : {NORM_ADJ_OUTPUT}")
    print(f"edge_index.pt  : {EDGE_INDEX_OUTPUT}")
    print(f"graph_meta.json: {META_OUTPUT}")


def main():
    train, val, test, num_users, num_items = load_inputs()
    num_nodes = num_users + num_items

    # Graph is constructed from Train only. Val/Test are used for metadata/checks.
    R = build_interaction_matrix(train, num_users, num_items)
    adjacency = build_full_adjacency(R, num_users, num_items)
    normalized, degree = normalize_adjacency(adjacency)

    norm_adj_tensor = sparse_to_torch_coo(normalized)
    edge_index = build_edge_index(train, num_users)

    metadata = {
        "num_users": int(num_users),
        "num_items": int(num_items),
        "num_nodes": int(num_nodes),
        "num_train_interactions": int(len(train)),
        "num_train_edges": int(R.nnz),
        "num_bidirectional_edges": int(edge_index.shape[1]),
        "graph_density": round(float(R.nnz / (num_users * num_items)), 8),
        "text_feature_dim": EXPECTED_FEATURE_DIM,
        "image_feature_dim": EXPECTED_FEATURE_DIM,
        "user_node_range": [0, int(num_users - 1)],
        "item_node_range": [int(num_users), int(num_nodes - 1)],
        "norm_adj_nnz": int(normalized.nnz),
        "isolated_nodes": int((degree == 0).sum()),
        "split_strategy": "leave_one_out",
        "split_counts": {
            "train": int(len(train)),
            "val": int(len(val)),
            "test": int(len(test)),
        },
        "split_user_counts": {
            "train": int(train["user_idx"].nunique()),
            "val": int(val["user_idx"].nunique()),
            "test": int(test["user_idx"].nunique()),
        },
        "graph_source": "train_only",
        "interaction_semantics": "one_binary_edge_per_unique_user_item_pair",
    }

    print("\n=== GRAPH METADATA ===")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    save_outputs(norm_adj_tensor, edge_index, metadata)


if __name__ == "__main__":
    main()
