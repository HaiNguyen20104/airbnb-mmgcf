"""Step 6: rank unseen Bangkok listings for an existing reviewer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

if __package__:
    from .data_loader import find_project_root, load_model_inputs
    from .late_fusion_mmgcf import LateFusionMMGCF
    from .mmgcf import MMGCF
else:
    from data_loader import find_project_root, load_model_inputs
    from late_fusion_mmgcf import LateFusionMMGCF
    from mmgcf import MMGCF


def resolve_user_idx(data, user_id: str | None, user_idx: int | None) -> int:
    if user_idx is not None:
        if not 0 <= user_idx < data.num_users:
            raise ValueError(f"user_idx must be between 0 and {data.num_users - 1}.")
        return user_idx
    if user_id is None:
        raise ValueError("Provide either user_id or user_idx.")
    mapping = data.user_mapping
    matches = mapping.loc[mapping["reviewer_id"].astype(str) == str(user_id), "user_idx"]
    if matches.empty:
        raise ValueError(f"Unknown reviewer_id: {user_id}")
    return int(matches.iloc[0])


@torch.inference_mode()
def recommend_user(model, data, user_idx: int, top_k: int = 10) -> pd.DataFrame:
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if not 0 <= user_idx < data.num_users:
        raise ValueError("Unknown user_idx.")

    seen = set()
    for edges in (data.train_edges, data.val_edges, data.test_edges):
        seen.update(edges.loc[edges["user_idx"] == user_idx, "item_idx"].astype(int))

    model.eval() if hasattr(model, "eval") else None
    user = torch.tensor([user_idx], dtype=torch.long, device=data.device)
    scores = model.score_all_items(user, output=model()).squeeze(0).clone()
    if seen:
        scores[list(seen)] = -torch.inf
    count = min(top_k, data.num_items - len(seen))
    if count == 0:
        return pd.DataFrame(columns=["item_idx", "listing_id", "score", "cold_start"])
    values, indices = torch.topk(scores, k=count)
    ranking = pd.DataFrame({
        "item_idx": indices.cpu().tolist(),
        "score": values.cpu().tolist(),
        "cold_start": model.cold_item_mask[indices].cpu().tolist(),
    })
    return ranking.merge(
        data.item_mapping[["item_idx", "listing_id"]],
        on="item_idx", how="left", validate="one_to_one", sort=False,
    )[["item_idx", "listing_id", "score", "cold_start"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommend unseen listings to an existing user.")
    user_group = parser.add_mutually_exclusive_group(required=True)
    user_group.add_argument("--user-id")
    user_group.add_argument("--user-idx", type=int)
    parser.add_argument("--city", default="bangkok")
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, help="Optional .csv or .json output path.")
    args = parser.parse_args()

    root = find_project_root(args.city, args.project_root)
    checkpoint_path = args.checkpoint or (
        root / "data" / "processed" / args.city / "model" / "tune_id025_150" / "mmgcf_best.pt"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    data = load_model_inputs(
        city=args.city, project_root=root, device=args.device, save_metadata=False,
    )
    if (config["city"], config["num_users"], config["num_items"]) != (
        args.city, data.num_users, data.num_items
    ):
        raise ValueError("Checkpoint does not match this dataset.")
    if config["model"] == "late_fusion":
        model = LateFusionMMGCF(
            data, embedding_dim=config["embedding_dim"],
            num_layers=config["num_layers"], id_weight=config["id_weight"],
        )
    elif config["model"] == "attention":
        model = MMGCF(
            data, embedding_dim=config["embedding_dim"],
            num_layers=config["num_layers"], attention_dim=config["attention_dim"],
        )
    else:
        raise ValueError(f"Unsupported model: {config['model']}")
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(data.device)

    index = resolve_user_idx(data, args.user_id, args.user_idx)
    ranking = recommend_user(model, data, index, args.top_k)
    listings_path = root / "data" / "processed" / args.city / "listings_final.parquet"
    if listings_path.exists():
        listings = pd.read_parquet(listings_path)
        columns = [c for c in ("id", "name", "price", "neighbourhood_cleansed", "listing_url") if c in listings]
        if "id" in columns:
            listings = listings[columns].rename(columns={"id": "listing_id"})
            ranking["listing_id"] = ranking["listing_id"].astype(str)
            listings["listing_id"] = listings["listing_id"].astype(str)
            ranking = ranking.merge(
                listings.drop_duplicates("listing_id"), on="listing_id", how="left", sort=False,
            )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.suffix.lower() == ".csv":
            ranking.to_csv(args.output, index=False)
        elif args.output.suffix.lower() == ".json":
            args.output.write_text(
                json.dumps(ranking.to_dict(orient="records"), ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        else:
            parser.error("--output must end in .csv or .json")
        print(f"Saved recommendations: {args.output}")
    print(ranking.to_string(index=False))


if __name__ == "__main__":
    main()
