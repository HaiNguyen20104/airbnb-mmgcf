# -*- coding: utf-8 -*-
"""
train.py
========
Vòng lặp training cho MMGCF (thuần PyTorch):

  - Loss = BPR (bắt chước PyG LightGCN.recommendation_loss, lambda_reg=1e-4)
           + lambda_contrast * InfoNCE hai chiều (G4) khi bật.
  - Âm mẫu: uniform random trên toàn bộ items (đúng protocol utils.py của tác giả).
  - Chọn model tốt nhất theo Val NDCG@10, có early-stop theo patience.
  - Mỗi epoch tính lại lan truyền LightGCN theo từng batch (đạo hàm đúng chuẩn).
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import random
import time
from copy import deepcopy

import numpy as np
import torch

from evaluation.evaluator import evaluate_ranking


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(cfg, model, train_edges, eval_data, log_fn=print):
    """
    Args:
        cfg: Config.
        model: MMGCF (chưa .to(device)).
        train_edges: np.ndarray (E, 2) [user_idx, item_idx] — tập TRAIN.
        eval_data: dict với các khóa:
            val_df        : DataFrame(user_idx, item_idx) — có thể rỗng
            excl_csr      : scipy CSR (num_users x num_items) — các item cần che
                            khi xếp hạng trên Val (= TRAIN graph)
        log_fn: hàm ghi log.

    Returns:
        dict: {
            "best_state": state_dict (CPU) của epoch tốt nhất,
            "best_epoch": int,
            "history": list[dict],
            "val_metrics": metrics của epoch tốt nhất,
        }
    """
    device = torch.device(cfg.device) if cfg.device != "auto" else None
    if device is None:
        from configs.config import resolve_device
        device = resolve_device("auto")

    model = model.to(device)
    set_seed(cfg.seed)

    num_users, num_items = model.num_users, model.num_items
    edges = torch.from_numpy(train_edges.astype(np.int64)).to(device)
    num_edges = edges.shape[0]
    if num_edges == 0:
        raise ValueError("Train interactions rỗng.")

    val_df = eval_data["val_df"]
    excl_csr = eval_data["excl_csr"]
    eval_ks = (10,)  # metric chọn model

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    generator = torch.Generator().manual_seed(cfg.seed + 1)

    # Âm mẫu đúng protocol bản gốc (feat/mmgcf - data_loader.NegativeSampler):
    # chỉ chọn từ các item XUẤT HIỆN trong tập train, và loại trừ các item
    # mà user đã tương tác (resample khi trùng — tránh false negative).
    catalog = torch.unique(edges[:, 1])
    seen_mask = torch.zeros(num_users, num_items, dtype=torch.bool, device=device)
    seen_mask[edges[:, 0], edges[:, 1]] = True

    best = {"val_ndcg": -1.0, "state": None, "epoch": -1, "metrics": None}
    no_improve = 0
    history = []

    log_fn(f"[TRAIN] {cfg.run_name}: edges={num_edges:,} | device={device} | "
           f"epochs={cfg.epochs} | dim={cfg.embedding_dim} | layers={cfg.n_layers} | "
           f"fusion={cfg.fusion_modalities} | weighting={cfg.weighting} | "
           f"freeze_mm={cfg.freeze_mm} | demand={model.use_demand} | "
           f"lambda_c={cfg.lambda_contrast} | cold_aug={cfg.cold_aug}")

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.perf_counter()
        model.train()
        perm = torch.randperm(num_edges, generator=generator).to(device)

        total_loss = total_examples = 0
        for start in range(0, num_edges, cfg.batch_size):
            batch_idx = perm[start:start + cfg.batch_size]
            users, items = edges[batch_idx, 0], edges[batch_idx, 1]

            # Âm mẫu: catalogue train-only, loại trừ item user đã tương tác
            # (đúng NegativeSampler của bản gốc); resample tối đa 32 vòng.
            with torch.no_grad():
                idx = torch.randint(0, len(catalog), (len(batch_idx),),
                                    device="cpu", generator=generator).to(device)
                neg_items = catalog[idx]
                for _ in range(32):
                    bad = seen_mask[users, neg_items]
                    if not bool(bad.any()):
                        break
                    idx_bad = torch.randint(0, len(catalog), (int(bad.sum()),),
                                            device="cpu", generator=generator).to(device)
                    neg_items[bad] = catalog[idx_bad]

            optimizer.zero_grad(set_to_none=True)

            user_emb, final_item, graph_view, content_view = model.get_embedding()
            loss, _, _ = model.recommendation_loss(
                user_emb, final_item, users, items, neg_items, cfg.lambda_reg
            )

            c_loss = model.contrastive_loss(graph_view, content_view, items)
            if c_loss is not None:
                loss = loss + cfg.lambda_contrast * c_loss

            loss.backward()
            optimizer.step()

            bs = len(batch_idx)
            total_loss += float(loss.detach()) * bs
            total_examples += bs

        train_loss = total_loss / max(total_examples, 1)
        elapsed = time.perf_counter() - t0

        row = {"epoch": epoch, "train_loss": round(train_loss, 5),
               "time_s": round(elapsed, 2)}

        # ---- Đánh giá trên Val ----
        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
            model.eval()
            with torch.no_grad():
                user_emb, final_item, _, content_view = model.get_embedding()
            if getattr(model, "use_cold_channel", False):
                extras = model.cold_channel_extras(content_view)
            else:
                extras = None
            if len(val_df) > 0:
                val_metrics, _ = evaluate_ranking(
                    user_emb, final_item, excl_csr, val_df, eval_ks, device,
                    extras=extras,
                )
                val_ndcg = val_metrics[10]["ndcg"]
                val_hr = val_metrics[10]["hr"]
            else:
                # Không có Val (user chỉ có <=2 tương tác toàn bộ) -> dùng loss
                val_ndcg, val_hr = None, None

            row["val_hr@10"] = val_hr
            row["val_ndcg@10"] = val_ndcg
            history.append(row)
            log_fn(
                f"  [epoch {epoch:4d}] loss={train_loss:.4f} | "
                f"val HR@10={val_hr if val_hr is None else round(val_hr, 4)} | "
                f"val NDCG@10={val_ndcg if val_ndcg is None else round(val_ndcg, 4)} | "
                f"{elapsed:.1f}s"
            )

            if val_ndcg is not None and val_ndcg > best["val_ndcg"]:
                best = {
                    "val_ndcg": val_ndcg,
                    "state": deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()}),
                    "epoch": epoch,
                    "metrics": {"hr@10": val_hr, "ndcg@10": val_ndcg},
                }
                no_improve = 0
            elif val_ndcg is not None:
                no_improve += 1
                if no_improve >= cfg.patience_evals:
                    log_fn(f"  [EARLY STOP] Không cải thiện sau {cfg.patience_evals} lần đánh giá.")
                    break
        else:
            history.append(row)

    if best["state"] is None:
        # Không có Val để chọn — dùng trạng thái cuối
        best["state"] = deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        best["epoch"] = epoch

    log_fn(f"[TRAIN] {cfg.run_name}: best epoch={best['epoch']} "
           f"(val NDCG@10={best['val_ndcg']:.4f})")

    return {
        "best_state": best["state"],
        "best_epoch": best["epoch"],
        "history": history,
        "val_metrics": best["metrics"],
    }
