"""Step 5.3: train MMGCF with BPR and evaluate full-catalog Top-K ranking.

Run from the project root or from src/model::

    python src/model/train_eval.py --city bangkok --device cpu

The adjacency and training negatives use train interactions only. Validation
selects the best epoch; test is evaluated once after that epoch is restored.
No model is trained when this module is imported.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch

if __package__:
    from .data_loader import load_model_inputs
    from .late_fusion_mmgcf import LateFusionMMGCF
    from .mmgcf import MMGCF
else:
    from data_loader import load_model_inputs
    from late_fusion_mmgcf import LateFusionMMGCF
    from mmgcf import MMGCF


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate MMGCF (Step 5.3).")
    parser.add_argument("--city", default="bangkok")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, cuda:0, or auto")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--negatives", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--l2-weight", type=float, default=1e-4)
    parser.add_argument("--model", choices=["late_fusion", "attention"], default="late_fusion")
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--attention-dim", type=int, default=32)
    parser.add_argument("--id-weight", type=float, default=0.25,
                        help="Late-fusion ID weight for warm items, in [0, 1).")
    parser.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--select-k", type=int, default=10)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-only", action="store_true",
                        help="Save a validation checkpoint without evaluating test.")
    parser.add_argument("--evaluate-checkpoint", type=Path, default=None,
                        help="Evaluate an existing checkpoint without training.")
    args = parser.parse_args()

    if min(args.epochs, args.batch_size, args.eval_batch_size, args.negatives, args.patience) <= 0:
        parser.error("epochs, batch sizes, negatives, and patience must be positive")
    if args.learning_rate <= 0 or args.l2_weight < 0:
        parser.error("learning-rate must be positive and l2-weight must be nonnegative")
    if any(k <= 0 for k in args.ks) or args.select_k not in args.ks:
        parser.error("ks must be positive and select-k must be one of ks")
    if args.embedding_dim <= 0 or args.attention_dim <= 0 or args.num_layers < 0:
        parser.error("invalid model dimensions or layer count")
    if not 0 <= args.id_weight < 1:
        parser.error("id-weight must be in [0, 1)")
    args.ks = sorted(set(args.ks))
    return args


def seed_random_generators(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed)


def interaction_sets(edges, num_users: int) -> list[set[int]]:
    """Local item IDs observed by each user in one or more given splits."""
    seen = [set() for _ in range(num_users)]
    for user, item in edges[["user_idx", "item_idx"]].itertuples(index=False, name=None):
        seen[int(user)].add(int(item))
    return seen


class NegativeSampler:
    """Sample train-observed items that a user has not interacted with."""

    def __init__(
        self,
        train_seen: list[set[int]],
        num_items: int,
        rng: np.random.Generator,
        candidate_items: np.ndarray,
    ) -> None:
        self.train_seen = train_seen
        self.rng = rng
        self.dense_candidates: dict[int, np.ndarray] = {}
        catalog = np.unique(np.asarray(candidate_items, dtype=np.int64))
        if catalog.size == 0 or catalog.min() < 0 or catalog.max() >= num_items:
            raise ValueError("Negative candidate pool must contain valid train item IDs.")
        self.catalog = catalog
        for user, seen in enumerate(train_seen):
            if len(seen) >= len(catalog):
                raise ValueError(f"User {user} has no available negative item.")
            if len(seen) > len(catalog) // 2:
                self.dense_candidates[user] = np.setdiff1d(
                    catalog, np.fromiter(seen, dtype=np.int64), assume_unique=True
                )

    def sample(self, users: np.ndarray, count: int) -> np.ndarray:
        negatives = np.empty((len(users), count), dtype=np.int64)
        for row, raw_user in enumerate(users):
            user = int(raw_user)
            seen = self.train_seen[user]
            candidates = self.dense_candidates.get(user)
            for col in range(count):
                if candidates is not None:
                    negatives[row, col] = self.rng.choice(candidates)
                else:
                    item = int(self.rng.choice(self.catalog))
                    while item in seen:
                        item = int(self.rng.choice(self.catalog))
                    negatives[row, col] = item
        return negatives[:, 0] if count == 1 else negatives


def metric_bucket(ranks: np.ndarray, ks: list[int]) -> dict[str, int | float | None]:
    """One held-out item per row; HR@K equals Recall@K in this protocol."""
    result: dict[str, int | float | None] = {"count": int(len(ranks))}
    for k in ks:
        found = ranks <= k
        result[f"HR@{k}"] = float(found.mean()) if len(ranks) else None
        result[f"NDCG@{k}"] = (
            float((1.0 / np.log2(ranks[found] + 1)).sum() / len(ranks))
            if len(ranks) else None
        )
    return result


@torch.inference_mode()
def evaluate(
    model: MMGCF | LateFusionMMGCF,
    user_idx: torch.Tensor,
    item_idx: torch.Tensor,
    seen: list[set[int]],
    ks: list[int],
    batch_size: int,
) -> dict[str, dict[str, int | float | None]]:
    """Rank each held-out item against the full catalog after seen-item masking."""
    if user_idx.shape != item_idx.shape or user_idx.ndim != 1 or user_idx.numel() == 0:
        raise ValueError("Evaluation expects nonempty aligned 1D user/item tensors.")
    if max(ks) > model.num_items:
        raise ValueError("Largest K exceeds the catalog size.")

    model.eval()
    output = model()
    cold_mask = model.cold_item_mask.cpu().numpy()
    all_ranks: list[np.ndarray] = []
    all_cold: list[np.ndarray] = []
    max_k = max(ks)

    for start in range(0, user_idx.numel(), batch_size):
        users = user_idx[start : start + batch_size]
        targets = item_idx[start : start + batch_size]
        scores = model.score_all_items(users, output=output)
        user_ids = users.cpu().tolist()
        target_ids = targets.cpu().numpy()

        # Build one indexing tensor per batch rather than moving each user's
        # seen list separately to the GPU.
        rows, cols = [], []
        for row, user in enumerate(user_ids):
            if int(target_ids[row]) in seen[user]:
                raise ValueError("Held-out positive is already masked as seen.")
            items = seen[user]
            rows.extend([row] * len(items))
            cols.extend(items)
        if rows:
            scores[
                torch.tensor(rows, dtype=torch.long, device=scores.device),
                torch.tensor(cols, dtype=torch.long, device=scores.device),
            ] = -torch.inf

        top_items = torch.topk(scores, k=max_k, dim=1).indices
        matches = top_items.eq(targets[:, None]).cpu().numpy()
        ranks = np.where(matches.any(axis=1), matches.argmax(axis=1) + 1, max_k + 1)
        all_ranks.append(ranks)
        all_cold.append(cold_mask[target_ids])

    ranks = np.concatenate(all_ranks)
    cold = np.concatenate(all_cold)
    return {
        "all": metric_bucket(ranks, ks),
        "cold": metric_bucket(ranks[cold], ks),
        "warm": metric_bucket(ranks[~cold], ks),
    }


def train_epoch(
    model: MMGCF | LateFusionMMGCF,
    optimizer: torch.optim.Optimizer,
    train_users: torch.Tensor,
    train_items: torch.Tensor,
    sampler: NegativeSampler,
    batch_size: int,
    negatives: int,
    l2_weight: float,
    device: torch.device,
) -> float:
    model.train()
    order = torch.randperm(train_users.numel())
    total_loss = 0.0

    for positions in order.split(batch_size):
        users_cpu = train_users[positions]
        items_cpu = train_items[positions]
        sampled = sampler.sample(users_cpu.numpy(), negatives)
        users = users_cpu.to(device)
        positives = items_cpu.to(device)
        negative_items = torch.from_numpy(sampled).to(device)

        optimizer.zero_grad(set_to_none=True)
        loss = model.bpr_loss(users, positives, negative_items, l2_weight=l2_weight)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite BPR loss encountered.")
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().item()) * len(positions)

    return total_loss / train_users.numel()


def evaluate_checkpoint(args: argparse.Namespace, data) -> None:
    path = args.evaluate_checkpoint.expanduser().resolve()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    if config["city"] != args.city or config["num_users"] != data.num_users or config["num_items"] != data.num_items:
        raise ValueError("Checkpoint city or user/item counts do not match the loaded dataset.")
    if config["model"] == "late_fusion":
        model = LateFusionMMGCF(
            data,
            embedding_dim=config["embedding_dim"],
            num_layers=config["num_layers"],
            id_weight=config["id_weight"],
        ).to(data.device)
    elif config["model"] == "attention":
        model = MMGCF(
            data,
            embedding_dim=config["embedding_dim"],
            num_layers=config["num_layers"],
            attention_dim=config["attention_dim"],
        ).to(data.device)
    else:
        raise ValueError(f"Unknown checkpoint model: {config['model']}")
    model.load_state_dict(checkpoint["model_state"])

    train_seen = interaction_sets(data.train_edges, data.num_users)
    test_seen = [set(items) for items in train_seen]
    for user, item in data.val_edges[["user_idx", "item_idx"]].itertuples(index=False, name=None):
        test_seen[int(user)].add(int(item))
    validation = evaluate(
        model, data.val_user_idx, data.val_item_idx,
        train_seen, args.ks, args.eval_batch_size,
    )
    test = evaluate(
        model, data.test_user_idx, data.test_item_idx,
        test_seen, args.ks, args.eval_batch_size,
    )
    report_path = path.parent / "mmgcf_metrics.json"
    report = {
        "best_epoch": checkpoint["best_epoch"],
        "selection_metric": "validation NDCG@10",
        "validation": validation,
        "test": test,
        "evaluation": {
            "ranking": "full item catalog",
            "validation_seen_mask": "train",
            "test_seen_mask": "train + validation",
            "negative_candidate_pool": "items observed in train",
            "k_values": args.ks,
        },
        "config": config,
    }
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
    print("Validation:", json.dumps(validation, ensure_ascii=False))
    print("Test:", json.dumps(test, ensure_ascii=False))
    print(f"Metrics: {report_path}")


def main() -> None:
    args = parse_args()
    rng = seed_random_generators(args.seed)
    data = load_model_inputs(
        city=args.city,
        project_root=args.project_root,
        device=args.device,
        save_metadata=False,
    )
    if max(args.ks) > data.num_items:
        raise ValueError("Largest K exceeds the number of items.")
    if args.evaluate_checkpoint is not None:
        evaluate_checkpoint(args, data)
        return

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path(data.paths["train_interactions"]).parent.parent / "model" / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.model == "late_fusion":
        model = LateFusionMMGCF(
            data,
            embedding_dim=args.embedding_dim,
            num_layers=args.num_layers,
            id_weight=args.id_weight,
        ).to(data.device)
    else:
        model = MMGCF(
            data,
            embedding_dim=args.embedding_dim,
            num_layers=args.num_layers,
            attention_dim=args.attention_dim,
        ).to(data.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    train_seen = interaction_sets(data.train_edges, data.num_users)
    train_users = data.train_user_idx.detach().cpu()
    train_items = data.train_item_idx.detach().cpu()
    # Cold items have no train positives. Sampling them only as negatives
    # teaches BPR to suppress their content embeddings at recommendation time.
    sampler = NegativeSampler(
        train_seen, data.num_items, rng, np.unique(train_items.numpy())
    )

    # Validation candidates exclude train positives. Test candidates exclude
    # train and validation positives, as validation is known by test time.
    test_seen = [set(items) for items in train_seen]
    for user, item in data.val_edges[["user_idx", "item_idx"]].itertuples(index=False, name=None):
        test_seen[int(user)].add(int(item))

    best_score = -math.inf
    best_epoch = 0
    best_state = None
    best_validation = None
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(
            model, optimizer, train_users, train_items, sampler,
            args.batch_size, args.negatives, args.l2_weight, data.device,
        )
        validation = evaluate(
            model, data.val_user_idx, data.val_item_idx,
            train_seen, args.ks, args.eval_batch_size,
        )
        score = validation["all"][f"NDCG@{args.select_k}"]
        if score is None:
            raise ValueError("Validation split contains no examples.")
        print(
            f"Epoch {epoch:03d} | BPR loss {loss:.5f} | "
            f"val HR@{args.select_k} {validation['all'][f'HR@{args.select_k}']:.5f} | "
            f"val NDCG@{args.select_k} {score:.5f}"
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_validation = validation
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after {args.patience} epochs without improvement.")
                break

    if best_state is None or best_validation is None:
        raise RuntimeError("Training did not produce a validation checkpoint.")
    model.load_state_dict(best_state)
    test = None
    if not args.validation_only:
        test = evaluate(
            model, data.test_user_idx, data.test_item_idx,
            test_seen, args.ks, args.eval_batch_size,
        )

    config = {
        "city": args.city,
        "model": args.model,
        "embedding_dim": args.embedding_dim,
        "num_layers": args.num_layers,
        "attention_dim": args.attention_dim,
        "id_weight": args.id_weight,
        "seed": args.seed,
        "num_users": data.num_users,
        "num_items": data.num_items,
    }
    checkpoint_path = output_dir / "mmgcf_best.pt"
    metrics_path = output_dir / "mmgcf_metrics.json"
    torch.save(
        {"model_state": best_state, "config": config, "best_epoch": best_epoch},
        checkpoint_path,
    )
    report = {
        "best_epoch": best_epoch,
        "selection_metric": f"validation NDCG@{args.select_k}",
        "validation": best_validation,
        "test": test,
        "evaluation": {
            "ranking": "full item catalog",
            "validation_seen_mask": "train",
            "test_seen_mask": "train + validation",
            "negative_sampling_exclusion": "train only",
            "negative_candidate_pool": "items observed in train",
            "k_values": args.ks,
        },
        "config": config,
    }
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print(f"Best epoch: {best_epoch}")
    print("Validation:", json.dumps(best_validation, ensure_ascii=False))
    if test is not None:
        print("Test:", json.dumps(test, ensure_ascii=False))
    else:
        print("Test was not evaluated (--validation-only).")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
