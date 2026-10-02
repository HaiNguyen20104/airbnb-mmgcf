"""
split_dataset.py
================
Bước 2: Chuyển đổi ID và chia tập dữ liệu theo chiến lược Leave-One-Out (LOO).

Chiến lược LOO (chuẩn của các paper LightGCN, NGCF, MMGCF):
  - 1 tương tác (ngẫu nhiên) của mỗi user -> Test
  - 1 tương tác khác (ngẫu nhiên) của mỗi user -> Val  (chỉ với user có >= 3 tuong tac)
  - Phần còn lại -> Train

Ưu điểm:
  - 100% người dùng đều xuất hiện trong Test -> đánh giá công bằng toàn diện
  - Phù hợp với metrics chuẩn HR@K và NDCG@K (mỗi user luôn có đúng 1 ground-truth)
  - Khớp với thiết lập thực tế: "Dự đoán căn phòng kế tiếp user sẽ đặt"

Đầu vào:
  - interactions_final.parquet (reviewer_id, listing_id)
  - user_mapping.parquet      (user_idx, reviewer_id)
  - item_mapping.parquet      (item_idx, listing_id)

Đầu ra (lưu tại data/processed/bangkok/graph/):
  - interactions_mapped.parquet  : Toàn bộ tương tác đã map sang idx
  - train_interactions.parquet   : Train set
  - val_interactions.parquet     : Val set   (1 item/user với user >= 3 tuong tac)
  - test_interactions.parquet    : Test set  (1 item/user, 100% users)
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "processed" / "bangkok"
FEATURES_DIR = DATA_DIR / "features"
GRAPH_DIR = DATA_DIR / "graph"

INTERACTIONS_PATH = DATA_DIR / "interactions_final.parquet"
USER_MAPPING_PATH = FEATURES_DIR / "user_mapping.parquet"
ITEM_MAPPING_PATH = FEATURES_DIR / "item_mapping.parquet"

OUTPUT_ALL_PATH   = GRAPH_DIR / "interactions_mapped.parquet"
TRAIN_OUTPUT_PATH = GRAPH_DIR / "train_interactions.parquet"
VAL_OUTPUT_PATH   = GRAPH_DIR / "val_interactions.parquet"
TEST_OUTPUT_PATH  = GRAPH_DIR / "test_interactions.parquet"

SEED = 42


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def load_and_validate_inputs():
    for path, name in [
        (INTERACTIONS_PATH, "Interactions"),
        (USER_MAPPING_PATH, "User mapping"),
        (ITEM_MAPPING_PATH, "Item mapping"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{name} file not found: {path}")

    interactions = pd.read_parquet(INTERACTIONS_PATH)
    user_mapping = pd.read_parquet(USER_MAPPING_PATH)
    item_mapping = pd.read_parquet(ITEM_MAPPING_PATH)

    print("=== INPUTS ===")
    print(f"Interactions  : {len(interactions):,}")
    print(f"Unique users  : {interactions['reviewer_id'].nunique():,}")
    print(f"Unique items  : {interactions['listing_id'].nunique():,}")
    print(f"User mapping  : {len(user_mapping):,} users")
    print(f"Item mapping  : {len(item_mapping):,} items")

    return interactions, user_mapping, item_mapping


def map_interactions(interactions, user_mapping, item_mapping):
    """Chuyển reviewer_id / listing_id sang user_idx / item_idx."""

    # Chuẩn hoá kiểu chuỗi
    interactions = interactions.copy()
    interactions["reviewer_id"] = interactions["reviewer_id"].astype("string").str.strip()
    interactions["listing_id"]  = interactions["listing_id"].astype("string").str.strip()
    user_mapping = user_mapping.copy()
    user_mapping["reviewer_id"] = user_mapping["reviewer_id"].astype("string").str.strip()
    item_mapping = item_mapping.copy()
    item_mapping["listing_id"]  = item_mapping["listing_id"].astype("string").str.strip()

    mapped = (
        interactions
        .merge(user_mapping, on="reviewer_id", how="inner")
        .merge(item_mapping,  on="listing_id",  how="inner")
    )

    if len(mapped) != len(interactions):
        raise ValueError(
            f"Mapping dropped rows! Original: {len(interactions)}, Mapped: {len(mapped)}"
        )

    # Sắp xếp cột cho gọn
    mapped = mapped[["user_idx", "item_idx", "reviewer_id", "listing_id"]].copy()

    if mapped[["user_idx", "item_idx"]].isna().any().any():
        raise ValueError("Mapped interactions contain null index values.")

    if mapped.duplicated(subset=["user_idx", "item_idx"]).any():
        raise ValueError("Duplicate (user_idx, item_idx) pairs found after mapping.")

    print("\n=== MAPPED INTERACTIONS ===")
    print(f"Total: {len(mapped):,} | Users: {mapped['user_idx'].nunique():,} | Items: {mapped['item_idx'].nunique():,}")

    return mapped


def split_leave_one_out(mapped_df, seed=42):
    """
    Leave-One-Out split (chuẩn của LightGCN / MMGCF paper):

      - Với mỗi user có >= 3 tuong tac:
          * 1 tương tác ngẫu nhiên  -> Test
          * 1 tương tác khác        -> Val
          * Phần còn lại            -> Train

      - Với user có đúng 2 tuong tac:
          * 1 -> Test
          * 1 -> Train  (không có Val vì không đủ)

      - Với user có đúng 1 tuong tac:
          * 1 -> Train  (không thể giấu đi, không có gì để test)

    Kết quả:
      - Test : 1 item mỗi user với 100% users có >= 2 tuong tac
      - Val  : 1 item mỗi user với users có >= 3 tuong tac (~99.9%)
    """
    np.random.seed(seed)

    train_rows, val_rows, test_rows = [], [], []

    for _, group in mapped_df.groupby("user_idx"):
        idx = group.index.to_numpy(copy=True)
        np.random.shuffle(idx)
        n = len(idx)

        if n >= 3:
            # idx[0] -> Test | idx[1] -> Val | idx[2:] -> Train
            test_rows.append(idx[0])
            val_rows.append(idx[1])
            train_rows.extend(idx[2:])
        elif n == 2:
            # idx[0] -> Test | idx[1] -> Train
            test_rows.append(idx[0])
            train_rows.append(idx[1])
        else:
            # n == 1: không đủ để giấu, đưa vào Train
            train_rows.extend(idx)

    train_df = (
        mapped_df.loc[train_rows]
        .sort_values(["user_idx", "item_idx"], kind="stable")
        .reset_index(drop=True)
    )
    val_df = (
        mapped_df.loc[val_rows]
        .sort_values(["user_idx", "item_idx"], kind="stable")
        .reset_index(drop=True)
    )
    test_df = (
        mapped_df.loc[test_rows]
        .sort_values(["user_idx", "item_idx"], kind="stable")
        .reset_index(drop=True)
    )

    # ── Kiểm tra toàn vẹn ────────────────────────────────────────
    total = len(train_df) + len(val_df) + len(test_df)
    assert total == len(mapped_df), f"Total mismatch: {total} vs {len(mapped_df)}"

    # Không được rò rỉ dữ liệu chéo
    train_pairs = set(zip(train_df["user_idx"], train_df["item_idx"]))
    val_pairs   = set(zip(val_df["user_idx"],   val_df["item_idx"]))
    test_pairs  = set(zip(test_df["user_idx"],  test_df["item_idx"]))

    assert len(train_pairs & val_pairs)  == 0, "Data leakage: Train vs Val!"
    assert len(train_pairs & test_pairs) == 0, "Data leakage: Train vs Test!"
    assert len(val_pairs   & test_pairs) == 0, "Data leakage: Val vs Test!"

    # Mọi user trong Val/Test phải có mặt trong Train
    train_users = set(train_df["user_idx"])
    val_unseen  = set(val_df["user_idx"])  - train_users
    test_unseen = set(test_df["user_idx"]) - train_users

    # Chỉ có những user đúng 1 tuong tac mới không trong Train, nhưng
    # họ cũng không có trong Val/Test, nên assert này luôn pass
    assert len(val_unseen)  == 0, f"Unseen users in Val:  {val_unseen}"
    assert len(test_unseen) == 0, f"Unseen users in Test: {test_unseen}"

    return train_df, val_df, test_df


def print_split_report(mapped_df, train_df, val_df, test_df):
    n = len(mapped_df)
    num_users = mapped_df["user_idx"].nunique()
    train_items = set(train_df["item_idx"])

    # LOO: test/val moi user co dung 1 item
    test_per_user = test_df.groupby("user_idx").size()
    val_per_user  = val_df.groupby("user_idx").size()

    cold_val  = len(set(val_df["item_idx"])  - train_items)
    cold_test = len(set(test_df["item_idx"]) - train_items)

    print("\n=== SPLIT SUMMARY (Leave-One-Out) ===")
    print(f"{'Tap':<8} {'Rows':>7}  {'%':>6}  {'Users':>6}  {'User%':>7}  {'Items':>6}")
    print("-" * 55)
    print(f"{'Train':<8} {len(train_df):>7,}  {len(train_df)/n:>6.1%}  {train_df['user_idx'].nunique():>6,}  {train_df['user_idx'].nunique()/num_users:>7.1%}  {train_df['item_idx'].nunique():>6,}")
    print(f"{'Val':<8} {len(val_df):>7,}  {len(val_df)/n:>6.1%}  {val_df['user_idx'].nunique():>6,}  {val_df['user_idx'].nunique()/num_users:>7.1%}  {val_df['item_idx'].nunique():>6,}")
    print(f"{'Test':<8} {len(test_df):>7,}  {len(test_df)/n:>6.1%}  {test_df['user_idx'].nunique():>6,}  {test_df['user_idx'].nunique()/num_users:>7.1%}  {test_df['item_idx'].nunique():>6,}")
    print(f"{'Total':<8} {n:>7,}  {'100.0%':>6}")

    print(f"\n--- Kiem tra chat luong ---")
    print(f"Items per user trong Test : min={test_per_user.min()}, max={test_per_user.max()}, (LOO: luon = 1)")
    print(f"Items per user trong Val  : min={val_per_user.min()},  max={val_per_user.max()},  (LOO: luon = 1)")
    print(f"Cold-start items trong Val : {cold_val:,}  (items chi xuat hien trong val, khong trong train)")
    print(f"Cold-start items trong Test: {cold_test:,}  (items chi xuat hien trong test, khong trong train)")

    # Users khong trong val (user chi co 2 tuong tac)
    users_no_val = num_users - val_df["user_idx"].nunique()
    print(f"Users khong co trong Val   : {users_no_val} (users chi co 2 tuong tac, khong du de tach Val)")


def main():
    # 1. Load
    interactions, user_mapping, item_mapping = load_and_validate_inputs()

    # 2. Map sang idx
    mapped = map_interactions(interactions, user_mapping, item_mapping)

    # 3. LOO Split
    print(f"\n=== SPLITTING (Leave-One-Out, seed={SEED}) ===")
    train_df, val_df, test_df = split_leave_one_out(mapped, seed=SEED)

    # 4. Bao cao
    print_split_report(mapped, train_df, val_df, test_df)

    # 5. Luu file
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    mapped.to_parquet(OUTPUT_ALL_PATH,   index=False)
    train_df.to_parquet(TRAIN_OUTPUT_PATH, index=False)
    val_df.to_parquet(VAL_OUTPUT_PATH,   index=False)
    test_df.to_parquet(TEST_OUTPUT_PATH,  index=False)

    print("\n=== SAVED FILES ===")
    print(f"All mapped : {OUTPUT_ALL_PATH}")
    print(f"Train      : {TRAIN_OUTPUT_PATH}")
    print(f"Val        : {VAL_OUTPUT_PATH}")
    print(f"Test       : {TEST_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
