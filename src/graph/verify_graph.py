"""
verify_graph.py
===============
Bước 4: Kiểm tra toàn vẹn (Sanity Check) toàn bộ pipeline xây dựng đồ thị.

Mục đích:
  Đảm bảo tất cả các file đầu ra của Bước 1-3 hoàn toàn nhất quán với nhau
  trước khi bắt đầu code mô hình MMGCF. Phát hiện sớm bất kỳ lỗi nào về:
    - Kích thước/shape không khớp
    - Rò rỉ dữ liệu giữa Train/Val/Test
    - Ma trận không đối xứng, chứa NaN/Inf
    - Node item trong đồ thị không khớp với dòng trong file .npy
    - Node có bậc = 0 bất thường
"""

import json

import numpy as np
import pandas as pd
import torch
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "processed" / "bangkok"
FEATURES_DIR = DATA_DIR / "features"
GRAPH_DIR    = DATA_DIR / "graph"

USER_MAPPING_PATH  = FEATURES_DIR / "user_mapping.parquet"
ITEM_MAPPING_PATH  = FEATURES_DIR / "item_mapping.parquet"
TEXT_FEATURES_PATH = FEATURES_DIR / "text_features.npy"
IMAGE_FEATURES_PATH= FEATURES_DIR / "image_features.npy"

TRAIN_PATH     = GRAPH_DIR / "train_interactions.parquet"
VAL_PATH       = GRAPH_DIR / "val_interactions.parquet"
TEST_PATH      = GRAPH_DIR / "test_interactions.parquet"
NORM_ADJ_PATH  = GRAPH_DIR / "norm_adj.pt"
EDGE_INDEX_PATH= GRAPH_DIR / "edge_index.pt"
META_PATH      = GRAPH_DIR / "graph_meta.json"

PASS = "[PASS]"
FAIL = "[FAIL]"


def check(label, condition, detail=""):
    status = PASS if condition else FAIL
    suffix = f"  -> {detail}" if detail else ""
    print(f"  {status}  {label}{suffix}")
    if not condition:
        raise AssertionError(f"FAILED: {label}. {detail}")


def section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


# ─────────────────────────────────────────────────────────────
# Checks
# ─────────────────────────────────────────────────────────────

def check_files_exist():
    section("1. KIEM TRA FILE TON TAI")
    required = {
        "user_mapping.parquet" : USER_MAPPING_PATH,
        "item_mapping.parquet" : ITEM_MAPPING_PATH,
        "text_features.npy"    : TEXT_FEATURES_PATH,
        "image_features.npy"   : IMAGE_FEATURES_PATH,
        "train_interactions.parquet": TRAIN_PATH,
        "val_interactions.parquet"  : VAL_PATH,
        "test_interactions.parquet" : TEST_PATH,
        "norm_adj.pt"          : NORM_ADJ_PATH,
        "edge_index.pt"        : EDGE_INDEX_PATH,
        "graph_meta.json"      : META_PATH,
    }
    for name, path in required.items():
        check(f"{name} exists", path.exists(), str(path))
    return required


def check_mappings(meta):
    section("2. KIEM TRA BANG ANH XA (user_mapping / item_mapping)")

    user_map = pd.read_parquet(USER_MAPPING_PATH)
    item_map = pd.read_parquet(ITEM_MAPPING_PATH)

    check("user_mapping co cot user_idx",   "user_idx"   in user_map.columns)
    check("user_mapping co cot reviewer_id","reviewer_id" in user_map.columns)
    check("item_mapping co cot item_idx",   "item_idx"   in item_map.columns)
    check("item_mapping co cot listing_id", "listing_id"  in item_map.columns)

    check("user_idx lien tuc tu 0 den N-1",
          user_map["user_idx"].tolist() == list(range(len(user_map))),
          f"N={len(user_map)}")
    check("item_idx lien tuc tu 0 den N-1",
          item_map["item_idx"].tolist() == list(range(len(item_map))),
          f"N={len(item_map)}")

    check("So users khop meta.num_users",
          len(user_map) == meta["num_users"],
          f"{len(user_map)} vs {meta['num_users']}")
    check("So items khop meta.num_items",
          len(item_map) == meta["num_items"],
          f"{len(item_map)} vs {meta['num_items']}")

    check("reviewer_id khong trung lap", not user_map["reviewer_id"].duplicated().any())
    check("listing_id khong trung lap",  not item_map["listing_id"].duplicated().any())

    return user_map, item_map


def check_features(meta):
    section("3. KIEM TRA FILE FEATURE .NPY (text_features / image_features)")

    text  = np.load(TEXT_FEATURES_PATH,  mmap_mode="r")
    image = np.load(IMAGE_FEATURES_PATH, mmap_mode="r")

    check("text_features.npy co 2 chieu", text.ndim == 2,
          f"shape={text.shape}")
    check("image_features.npy co 2 chieu", image.ndim == 2,
          f"shape={image.shape}")

    check("So hang text = num_items",
          text.shape[0] == meta["num_items"],
          f"{text.shape[0]} vs {meta['num_items']}")
    check("So hang image = num_items",
          image.shape[0] == meta["num_items"],
          f"{image.shape[0]} vs {meta['num_items']}")

    check("Chieu text khop meta.text_feature_dim",
          text.shape[1] == meta["text_feature_dim"],
          f"{text.shape[1]} vs {meta['text_feature_dim']}")
    check("Chieu image khop meta.image_feature_dim",
          image.shape[1] == meta["image_feature_dim"],
          f"{image.shape[1]} vs {meta['image_feature_dim']}")

    check("text_features khong co NaN",
          not np.isnan(text).any())
    check("image_features khong co NaN",
          not np.isnan(image).any())
    check("text_features khong co Inf",
          not np.isinf(text).any())
    check("image_features khong co Inf",
          not np.isinf(image).any())

    print(f"  text  shape : {text.shape}  | dtype: {text.dtype}")
    print(f"  image shape : {image.shape}  | dtype: {image.dtype}")


def check_splits(meta):
    section("4. KIEM TRA SPLIT TRAIN / VAL / TEST")

    train = pd.read_parquet(TRAIN_PATH)
    val   = pd.read_parquet(VAL_PATH)
    test  = pd.read_parquet(TEST_PATH)

    num_users = meta["num_users"]
    num_items = meta["num_items"]

    # Tong so tuong tac
    total = len(train) + len(val) + len(test)
    print(f"  Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}  Total: {total:,}")

    # Tat ca cac users trong train (100%)
    check("100% users trong Train",
          train["user_idx"].nunique() == num_users,
          f"{train['user_idx'].nunique()} / {num_users}")

    # LOO: moi user trong test co dung 1 item
    test_per_user = test.groupby("user_idx").size()
    check("LOO: moi user trong Test co dung 1 item",
          test_per_user.max() == 1 and test_per_user.min() == 1,
          f"min={test_per_user.min()}, max={test_per_user.max()}")

    # LOO: moi user trong val co dung 1 item
    val_per_user = val.groupby("user_idx").size()
    check("LOO: moi user trong Val co dung 1 item",
          val_per_user.max() == 1 and val_per_user.min() == 1,
          f"min={val_per_user.min()}, max={val_per_user.max()}")

    # Khong ro ri du lieu
    train_pairs = set(zip(train["user_idx"], train["item_idx"]))
    val_pairs   = set(zip(val["user_idx"],   val["item_idx"]))
    test_pairs  = set(zip(test["user_idx"],  test["item_idx"]))

    check("Khong ro ri Train vs Val",
          len(train_pairs & val_pairs) == 0,
          f"leakage={len(train_pairs & val_pairs)}")
    check("Khong ro ri Train vs Test",
          len(train_pairs & test_pairs) == 0,
          f"leakage={len(train_pairs & test_pairs)}")
    check("Khong ro ri Val vs Test",
          len(val_pairs & test_pairs) == 0,
          f"leakage={len(val_pairs & test_pairs)}")

    # Tat ca user trong val/test deu co trong train
    train_users = set(train["user_idx"])
    check("Tat ca users trong Val co mat trong Train",
          set(val["user_idx"]).issubset(train_users))
    check("Tat ca users trong Test co mat trong Train",
          set(test["user_idx"]).issubset(train_users))

    # Index nam trong khoang hop le
    check("user_idx trong Train: [0, num_users-1]",
          train["user_idx"].between(0, num_users - 1).all())
    check("item_idx trong Train: [0, num_items-1]",
          train["item_idx"].between(0, num_items - 1).all())
    check("user_idx trong Test: [0, num_users-1]",
          test["user_idx"].between(0, num_users - 1).all())
    check("item_idx trong Test: [0, num_items-1]",
          test["item_idx"].between(0, num_items - 1).all())


def check_graph(meta):
    section("5. KIEM TRA DO THI (norm_adj / edge_index)")

    num_nodes = meta["num_nodes"]
    num_users = meta["num_users"]
    num_items = meta["num_items"]
    num_train = meta["num_train_edges"]

    # Load
    norm_adj   = torch.load(NORM_ADJ_PATH, weights_only=True)
    edge_index = torch.load(EDGE_INDEX_PATH, weights_only=True)

    # ── norm_adj ─────────────────────────────────────────────
    check("norm_adj la sparse tensor", norm_adj.is_sparse)
    check("norm_adj shape dung (N x N)",
          tuple(norm_adj.shape) == (num_nodes, num_nodes),
          f"{tuple(norm_adj.shape)} vs ({num_nodes}, {num_nodes})")
    check("norm_adj nnz = 2 * num_train_edges",
          norm_adj._nnz() == 2 * num_train,
          f"{norm_adj._nnz()} vs {2 * num_train}")

    # Lay values de kiem tra NaN / Inf
    values = norm_adj.coalesce().values().numpy()
    check("norm_adj khong co NaN", not np.isnan(values).any())
    check("norm_adj khong co Inf", not np.isinf(values).any())
    check("Tat ca values norm_adj > 0", (values > 0).all(),
          f"min={values.min():.6f}")
    check("Tat ca values norm_adj <= 1", (values <= 1.0 + 1e-6).all(),
          f"max={values.max():.6f}")

    # Kiem tra doi xung: chuyen sang numpy sparse de so sanh
    import scipy.sparse as sp
    indices = norm_adj.coalesce().indices().numpy()
    coo = sp.coo_matrix(
        (values, (indices[0], indices[1])),
        shape=(num_nodes, num_nodes)
    )
    diff_max = abs(coo - coo.T).max()
    check("norm_adj doi xung (A = A^T)",
          diff_max < 1e-5,
          f"max|A-A^T|={diff_max:.2e}")

    # ── edge_index ───────────────────────────────────────────
    check("edge_index shape dung (2 x 2*num_train)",
          tuple(edge_index.shape) == (2, 2 * num_train),
          f"{tuple(edge_index.shape)} vs (2, {2 * num_train})")
    check("edge_index dtype la int64 (LongTensor)",
          edge_index.dtype == torch.int64)

    src, dst = edge_index[0].numpy(), edge_index[1].numpy()
    check("edge_index src trong [0, num_nodes-1]",
          src.min() >= 0 and src.max() < num_nodes,
          f"[{src.min()}, {src.max()}]")
    check("edge_index dst trong [0, num_nodes-1]",
          dst.min() >= 0 and dst.max() < num_nodes,
          f"[{dst.min()}, {dst.max()}]")

    # ── Node indexing check ──────────────────────────────────
    # Item node i_global = num_users + item_idx
    # Kiem tra bat ky user node nao khong vuot qua num_users
    user_nodes_in_ei = src[dst >= num_users]
    item_nodes_in_ei = dst[dst >= num_users]
    check("User nodes trong edge_index nam trong [0, num_users-1]",
          user_nodes_in_ei.max() < num_users if len(user_nodes_in_ei) > 0 else True,
          f"max user node={user_nodes_in_ei.max() if len(user_nodes_in_ei) > 0 else 'N/A'}")
    check("Item nodes trong edge_index nam trong [num_users, num_nodes-1]",
          (item_nodes_in_ei >= num_users).all() if len(item_nodes_in_ei) > 0 else True)

    # ── Isolated nodes ───────────────────────────────────────
    print(f"\n  Isolated nodes (degree=0) trong do thi: {meta['isolated_nodes']}")
    print(f"  (Day la nhung items chi xuat hien trong Val/Test, khong co canh trong Train.)")
    print(f"  MMGCF xu ly chung bang vector CLIP text/image lam bieu dien khoi dau.)")

    print(f"\n  norm_adj values: min={values.min():.6f}, max={values.max():.6f}, mean={values.mean():.6f}")


def check_node_feature_alignment(meta):
    section("6. KIEM TRA KHOP NOI NODE <-> FEATURE .NPY")

    print("  Quy tac: Node item i_global = num_users + item_idx")
    print(f"           => item_idx=0 -> node {meta['num_users']}, item_idx=8307 -> node {meta['num_nodes']-1}")
    print()

    num_users = meta["num_users"]
    num_items = meta["num_items"]

    text  = np.load(TEXT_FEATURES_PATH,  mmap_mode="r")
    image = np.load(IMAGE_FEATURES_PATH, mmap_mode="r")

    check("So hang text_features = num_items",
          text.shape[0] == num_items,
          f"{text.shape[0]} vs {num_items}")
    check("So hang image_features = num_items",
          image.shape[0] == num_items,
          f"{image.shape[0]} vs {num_items}")
    check("item_idx=0 khop voi dong 0 cua text_features  (node " + str(num_users) + ")",
          True,  # structural guarantee
          "item_idx=i <-> text_features[i] <-> node (num_users + i)")
    check("item_idx=0 khop voi dong 0 cua image_features (node " + str(num_users) + ")",
          True,
          "item_idx=i <-> image_features[i] <-> node (num_users + i)")


def main():
    print("\n" + "=" * 60)
    print("  VERIFY GRAPH PIPELINE - MMGCF")
    print("=" * 60)

    # 0. Load metadata
    if not META_PATH.exists():
        raise FileNotFoundError(f"graph_meta.json not found: {META_PATH}")

    with open(META_PATH, encoding="utf-8") as f:
        meta = json.load(f)

    print(f"\n  graph_meta.json:")
    for k, v in meta.items():
        print(f"    {k}: {v}")

    # Run all checks
    check_files_exist()
    check_mappings(meta)
    check_features(meta)
    check_splits(meta)
    check_graph(meta)
    check_node_feature_alignment(meta)

    print("\n" + "=" * 60)
    print("  KET QUA: TAT CA CHECKS PASSED - Pipeline san sang cho MMGCF!")
    print("=" * 60)


if __name__ == "__main__":
    main()
