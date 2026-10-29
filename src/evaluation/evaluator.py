# -*- coding: utf-8 -*-
"""
evaluator.py
============
Đánh giá xếp hạng toàn cục (full ranking) cho LOO protocol chuẩn LightGCN/MMGCF:

  - Với mỗi user: chấm điểm TẤT CẢ items, che các item đã interact trong
    Train (khi đánh giá Val) hoặc Train+Val (khi đánh giá Test) — khớp
    protocol test() trong utils.py của tác giả gốc.
  - HR@K == Precision@K == Recall@K vì LOO có đúng 1 ground truth/user.
  - NDCG@K = 1/log2(rank+2) khi ground truth nằm trong Top-K.
  - Hỗ trợ đánh giá phân tầng theo mức độ "lạnh" của item (G5).
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import torch


def _gather_csr_cols(csr, user_start, user_end, device):
    """Vector hoá lấy toàn bộ (row_in_batch, col) của dải user trong CSR."""
    indptr = csr.indptr
    indices = csr.indices
    # lens[k] = indptr[u+1] - indptr[u] với u = user_start + k;
    # hai slice luôn cùng độ dài, kể cả batch cuối (user_end = num_users).
    lens = indptr[user_start + 1 : user_end + 1] - indptr[user_start : user_end]
    total = int(lens.sum())

    if total == 0:
        rows = np.empty(0, dtype=np.int64)
        cols = np.empty(0, dtype=np.int64)
        return torch.from_numpy(rows).to(device), torch.from_numpy(cols).to(device)

    base = indptr[user_start:user_end]
    rows = np.repeat(np.arange(len(lens)), lens)                # chỉ số user trong batch
    offsets = np.repeat(base, lens)                             # đầu mỗi dải indices
    within = np.arange(total) - np.repeat(np.cumsum(lens) - lens, lens)
    cols = indices[offsets + within]
    return (
        torch.from_numpy(rows).to(device),
        torch.from_numpy(cols.astype(np.int64)).to(device),
    )


@torch.no_grad()
def evaluate_ranking(user_emb, item_emb, excl_csr, gt_df, ks, device,
                     batch_users=2048, max_k_cap=None, extras=None,
                     return_detail=False):
    """
    Args:
        user_emb, item_emb: Tensor (num_users, d), (num_items, d) — bất kỳ device.
        excl_csr: scipy CSR — item cần che khi xếp hạng (train hoặc train+val).
        gt_df: DataFrame(user_idx, item_idx) — ground truth cần đánh giá.
        ks: tuple các K.
        device: torch.device tính toán.
        extras: dict tùy chọn — kênh nội dung trực tiếp cho item lạnh (P1):
            {"user_profile": (num_users,d) normed, "content_item": (num_items,d) normed,
             "cold_mask": bool (num_items,), "gamma": float}
            -> cộng gamma·cos(profile_u, content_i) vào cột item lạnh.
        return_detail: trả về mảng per-user phục vụ đánh giá phân tầng.

    Returns:
        metrics: {k: {"hr": float, "ndcg": float, "n_users": int}}
        detail (nếu return_detail): dict các mảng per-user:
            {"gt_item": int64[num_users], "hit@k": bool[num_users],
             "ndcg@k": float[num_users]} cho từng k.
    """
    ks = sorted({int(k) for k in ks})
    num_users, num_items = user_emb.shape[0], item_emb.shape[0]
    k_max = max(ks) if max_k_cap is None else max(max(ks), max_k_cap)
    k_max = min(k_max, num_items)

    user_emb = user_emb.to(device)
    item_emb = item_emb.to(device)

    # Chế độ chấm cos (khi có kênh nội dung): đưa cả hai kênh về cùng thang
    if extras is not None and extras.get("cos_eval"):
        user_emb = torch.nn.functional.normalize(user_emb, dim=-1)
        item_emb = torch.nn.functional.normalize(item_emb, dim=-1)

    gt_mat = _build_gt_sparse(gt_df, num_users, num_items)

    hit_count = {k: 0 for k in ks}
    ndcg_sum = {k: 0.0 for k in ks}
    n_eval = 0

    detail = {
        "gt_item": np.full(num_users, -1, dtype=np.int64),
        **{f"hit@{k}": np.zeros(num_users, dtype=bool) for k in ks},
        **{f"ndcg@{k}": np.zeros(num_users, dtype=np.float32) for k in ks},
    }

    dcg_base = 1.0 / np.log2(np.arange(k_max) + 2.0)  # vị trí 0..k_max-1

    for u_start in range(0, num_users, batch_users):
        u_end = min(u_start + batch_users, num_users)
        b = u_end - u_start

        scores = user_emb[u_start:u_end] @ item_emb.t()      # (b, num_items)

        if extras is not None:
            channel = float(extras.get("gamma", 0.5)) * (
                extras["user_profile"].to(device)[u_start:u_end]
                @ extras["content_item"].to(device).t()
            )
            cold_f = extras["cold_mask"].to(device).to(channel.dtype)
            if extras.get("fusion") == "z":
                # Z-fusion scale-invariant theo user:
                #   cột warm giữ nguyên dot (thứ hạng train không đổi);
                #   cột cold thay bằng z_id (điểm id chuẩn hoá theo phân phối
                #   catalog của user) + gamma·z_prof (kênh cos chuẩn hoá trên
                #   tập cold của user). Không phụ thuộc độ lớn <u,·>.
                prof = extras["user_profile"].to(device)[u_start:u_end]
                content = extras["content_item"].to(device)
                cos_ch = prof @ content.t()                      # (b, num_items) ∈ [-1,1]
                finite = torch.isfinite(scores)
                n_fin = finite.sum(1).clamp(min=1).to(scores.dtype)
                mu = (scores.masked_fill(~finite, 0.0).sum(1) / n_fin)
                var = (scores.masked_fill(~finite, 0.0).pow(2).sum(1) / n_fin) - mu.pow(2)
                sd = var.clamp(min=1e-12).sqrt()
                fin_cold = cold_f.bool() & finite
                n_c = fin_cold.sum(1).clamp(min=1).to(scores.dtype)
                mu_c = (cos_ch.masked_fill(~fin_cold, 0.0).sum(1) / n_c)
                var_c = (cos_ch.masked_fill(~fin_cold, 0.0).pow(2).sum(1) / n_c) - mu_c.pow(2)
                sd_c = var_c.clamp(min=1e-12).sqrt()
                g = float(extras.get("gamma", 0.5))
                fused = ((scores - mu[:, None]) / sd[:, None]) + g * (
                    (cos_ch - mu_c[:, None]) / sd_c[:, None]
                )
                scores = torch.where(cold_f.bool() & finite, fused, scores)
            else:
                scores = scores + cold_f * channel

        rows, cols = _gather_csr_cols(excl_csr, u_start, u_end, device)
        if rows.numel() > 0:
            scores[rows, cols] = float("-inf")

        topk = torch.topk(scores, k=k_max, dim=1).indices     # (b, k_max)

        # Ground truth dải user này (LOO: tối đa 1 item/user)
        sub = gt_mat[u_start:u_end]
        gt_users, gt_items = sub.nonzero()
        gt_dense = torch.zeros((b, num_items), dtype=torch.bool, device=device)
        if len(gt_users) > 0:
            gt_dense[
                torch.from_numpy(gt_users).long().to(device),
                torch.from_numpy(gt_items).long().to(device),
            ] = True

        isin = torch.gather(gt_dense, 1, topk)                # (b, k_max) bool

        isin_np = isin.cpu().numpy()
        for i in range(b):
            gt_items_u = gt_items[gt_users == i]
            if len(gt_items_u) == 0:
                continue
            if return_detail:
                detail["gt_item"][u_start + i] = gt_items_u[0]
            row = isin_np[i]
            for k in ks:
                if row[:k].any():
                    pos = int(np.argmax(row[:k]))             # vị trí hit đầu tiên
                    hit_count[k] += 1
                    ndcg_sum[k] += float(dcg_base[pos])
                    if return_detail:
                        detail[f"hit@{k}"][u_start + i] = True
                        detail[f"ndcg@{k}"][u_start + i] = dcg_base[pos]
        n_eval += int((gt_dense.any(dim=1)).sum().item())

    metrics = {
        k: {
            "hr": hit_count[k] / max(n_eval, 1),
            "ndcg": ndcg_sum[k] / max(n_eval, 1),
            "n_users": n_eval,
        }
        for k in ks
    }
    if return_detail:
        return metrics, detail
    return metrics, None


def _build_gt_sparse(gt_df, num_users, num_items):
    import scipy.sparse as sp
    users = gt_df["user_idx"].to_numpy(dtype=np.int64)
    items = gt_df["item_idx"].to_numpy(dtype=np.int64)
    return sp.csr_matrix(
        (np.ones(len(users), dtype=np.int8), (users, items)),
        shape=(num_users, num_items),
    )


# ------------------------------------------------------------------
# Đánh giá phân tầng theo độ lạnh của item (G5 - mục tiêu #5)
# ------------------------------------------------------------------
BUCKET_LABELS = ("cold(0)", "low(1-2)", "mid(3-10)", "head(>10)")


def degree_bucket(degrees):
    """degrees: np.ndarray số cạnh train của item -> nhãn bucket."""
    return np.select(
        [degrees <= 0, degrees <= 2, degrees <= 10],
        ["cold(0)", "low(1-2)", "mid(3-10)"],
        default="head(>10)",
    )


def stratified_metrics(detail, item_train_degrees, ks=(5, 10, 20)):
    """
    Từ detail per-user (gt_item, hit@k, ndcg@k) -> metrics theo bucket độ lạnh.
    """
    gt_item = detail["gt_item"]
    valid = gt_item >= 0
    buckets = degree_bucket(item_train_degrees[gt_item[valid]])

    result = {}
    for label in BUCKET_LABELS:
        mask = buckets == label
        n = int(mask.sum())
        entry = {"n_items": n}
        for k in ks:
            if n > 0:
                entry[f"hr@{k}"] = float(detail[f"hit@{k}"][valid][mask].mean())
                entry[f"ndcg@{k}"] = float(detail[f"ndcg@{k}"][valid][mask].mean())
            else:
                entry[f"hr@{k}"] = None
                entry[f"ndcg@{k}"] = None
        result[label] = entry
    return result


def format_metrics_row(name, metrics, stratified=None):
    """Chuỗi một dòng kết quả cho bảng tổng hợp."""
    parts = [f"{name:<24}"]
    for k in sorted(metrics):
        m = metrics[k]
        parts.append(f"HR@{k}={m['hr']:.4f} NDCG@{k}={m['ndcg']:.4f}")
    line = " | ".join(parts)
    if stratified:
        cold = stratified.get("cold(0)", {})
        low = stratified.get("low(1-2)", {})
        h10 = cold.get("hr@10")
        l10 = low.get("hr@10")
        line += f" || cold0 HR@10={h10 if h10 is None else round(h10, 4)}" \
                f" low12 HR@10={l10 if l10 is None else round(l10, 4)}"
    return line
