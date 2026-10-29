"""Step 5.2: multimodal graph collaborative filtering model.

Example (run from the project root in the next training step)::

    from src.model.data_loader import load_model_inputs
    from src.model.mmgcf import MMGCF

    data = load_model_inputs(device="cpu")
    model = MMGCF(data, embedding_dim=64, num_layers=2).to(data.device)
    output = model()
    # output.user_embeddings: [num_users, 64]
    # output.item_embeddings: [num_items, 64]

Only the train adjacency participates in message passing. Validation and test
interactions must be used outside this module, during evaluation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from .data_loader import ModelInputs


@dataclass
class MMGCFOutput:
    """Fused embeddings and per-node weights for [ID, text, image]."""

    user_embeddings: torch.Tensor
    item_embeddings: torch.Tensor
    view_weights: torch.Tensor


class MMGCF(nn.Module):
    """Graph propagation and attention fusion of ID, text and image views.

    The three views use the same train-only normalized adjacency, but have
    independent initial node embeddings. Text and image item embeddings start
    from CLIP features; their user embeddings are learned through training.
    Cold-start items receive content embeddings and have no ID contribution.
    """

    def __init__(
        self,
        inputs: ModelInputs,
        embedding_dim: int = 64,
        num_layers: int = 2,
        attention_dim: int = 32,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0 or attention_dim <= 0 or num_layers < 0:
            raise ValueError("embedding_dim and attention_dim must be positive; num_layers must be nonnegative.")

        self.num_users = int(inputs.num_users)
        self.num_items = int(inputs.num_items)
        self.num_layers = int(num_layers)

        if tuple(inputs.normalized_adj.shape) != (
            self.num_users + self.num_items,
            self.num_users + self.num_items,
        ):
            raise ValueError("normalized_adj shape does not match user/item counts.")
        if inputs.normalized_adj.layout != torch.sparse_coo:
            raise ValueError("normalized_adj must be a sparse COO tensor.")
        if inputs.text_features.ndim != 2 or inputs.text_features.shape[0] != self.num_items:
            raise ValueError("text_features must have one row per item.")
        if inputs.image_features.ndim != 2 or inputs.image_features.shape[0] != self.num_items:
            raise ValueError("image_features must have one row per item.")
        if inputs.node_degree.shape != (self.num_users + self.num_items,):
            raise ValueError("node_degree shape does not match the graph.")

        self.register_buffer("normalized_adj", inputs.normalized_adj.coalesce(), persistent=False)
        self.register_buffer("text_features", inputs.text_features.detach().clone(), persistent=False)
        self.register_buffer("image_features", inputs.image_features.detach().clone(), persistent=False)
        self.register_buffer(
            "cold_item_mask",
            inputs.node_degree[self.num_users :].eq(0),
            persistent=False,
        )
        self.register_buffer(
            "isolated_node_mask",
            inputs.node_degree.eq(0),
            persistent=False,
        )

        self.user_id = nn.Embedding(self.num_users, embedding_dim)
        self.item_id = nn.Embedding(self.num_items, embedding_dim)
        self.text_user = nn.Embedding(self.num_users, embedding_dim)
        self.image_user = nn.Embedding(self.num_users, embedding_dim)
        self.text_projection = nn.Linear(inputs.text_features.shape[1], embedding_dim)
        self.image_projection = nn.Linear(inputs.image_features.shape[1], embedding_dim)

        # Shared scoring network: attention is normalized separately for each node.
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1, bias=False),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for embedding in (self.user_id, self.item_id, self.text_user, self.image_user):
            nn.init.normal_(embedding.weight, std=0.1)
        for projection in (self.text_projection, self.image_projection):
            nn.init.xavier_uniform_(projection.weight)
            nn.init.zeros_(projection.bias)
        for layer in self.attention:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def _propagate(self, initial: torch.Tensor) -> torch.Tensor:
        """Average graph layers without shrinking isolated content nodes."""
        current = initial
        summed = initial
        for _ in range(self.num_layers):
            current = torch.sparse.mm(self.normalized_adj, current)
            summed = summed + current
        average = summed / (self.num_layers + 1)
        # Every propagated layer of an isolated node is zero. Averaging those
        # zeros would divide its usable text/image embedding by num_layers+1.
        return torch.where(self.isolated_node_mask[:, None], initial, average)

    def forward(self) -> MMGCFOutput:
        id_nodes = torch.cat((self.user_id.weight, self.item_id.weight), dim=0)
        text_nodes = torch.cat(
            (self.text_user.weight, self.text_projection(self.text_features)), dim=0
        )
        image_nodes = torch.cat(
            (self.image_user.weight, self.image_projection(self.image_features)), dim=0
        )

        views = torch.stack(
            (
                self._propagate(id_nodes),
                self._propagate(text_nodes),
                self._propagate(image_nodes),
            ),
            dim=1,
        )  # [num_nodes, 3, embedding_dim]
        logits = self.attention(views).squeeze(-1)

        # A cold item's ID is random at initialization and never receives a
        # train edge. Its text/image views remain valid for recommendation.
        cf_available = torch.cat(
            (
                torch.ones(self.num_users, dtype=torch.bool, device=views.device),
                ~self.cold_item_mask,
            )
        )
        logits = torch.cat(
            (logits[:, :1].masked_fill(~cf_available[:, None], -torch.inf), logits[:, 1:]),
            dim=1,
        )
        weights = torch.softmax(logits, dim=1)
        fused = (views * weights.unsqueeze(-1)).sum(dim=1)

        return MMGCFOutput(
            user_embeddings=fused[: self.num_users],
            item_embeddings=fused[self.num_users :],
            view_weights=weights,
        )

    def score_pairs(
        self,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
        output: MMGCFOutput | None = None,
    ) -> torch.Tensor:
        """Dot-product scores for aligned user/item index tensors."""
        if output is None:
            output = self()
        return (
            output.user_embeddings[user_idx]
            * output.item_embeddings[item_idx]
        ).sum(dim=-1)

    def score_all_items(
        self,
        user_idx: torch.Tensor,
        output: MMGCFOutput | None = None,
    ) -> torch.Tensor:
        """Scores [selected_users, all_items], including cold-start items."""
        if output is None:
            output = self()
        return output.user_embeddings[user_idx] @ output.item_embeddings.T

    def bpr_loss(
        self,
        user_idx: torch.Tensor,
        positive_item_idx: torch.Tensor,
        negative_item_idx: torch.Tensor,
        l2_weight: float = 0.0,
    ) -> torch.Tensor:
        """Pairwise BPR loss for sampled negatives; sampling belongs to Step 5.3.

        negative_item_idx can have shape [batch] or [batch, negatives].
        Call this once per training batch and provide negatives absent from the
        user's known positive interactions.
        """
        if l2_weight < 0:
            raise ValueError("l2_weight must be nonnegative.")
        if user_idx.ndim != 1 or positive_item_idx.shape != user_idx.shape:
            raise ValueError("user_idx and positive_item_idx must have shape [batch].")
        if negative_item_idx.ndim not in (1, 2) or negative_item_idx.shape[0] != user_idx.shape[0]:
            raise ValueError("negative_item_idx must have shape [batch] or [batch, negatives].")
        if user_idx.numel() == 0 or negative_item_idx.numel() == 0:
            raise ValueError("BPR batch must be nonempty.")

        output = self()
        users = output.user_embeddings[user_idx]
        positives = output.item_embeddings[positive_item_idx]
        negatives = output.item_embeddings[negative_item_idx]
        positive_scores = (users * positives).sum(dim=-1)

        if negative_item_idx.ndim == 1:
            negative_scores = (users * negatives).sum(dim=-1)
        else:
            negative_scores = (users[:, None, :] * negatives).sum(dim=-1)
            positive_scores = positive_scores[:, None]

        loss = F.softplus(negative_scores - positive_scores).mean()
        if l2_weight:
            regularizer = (
                self.user_id(user_idx).square().sum(dim=-1).mean()
                + self.item_id(positive_item_idx).square().sum(dim=-1).mean()
                + self.item_id(negative_item_idx).square().sum(dim=-1).mean()
            ) / 2
            loss = loss + l2_weight * regularizer
        return loss


def main() -> None:
    """Build and inspect the untrained Step 5.2 model from Step 5.1 inputs."""
    parser = argparse.ArgumentParser(description="Build the MMGCF model (Step 5.2).")
    parser.add_argument("--city", default="bangkok")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, cuda:0, or auto")
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--attention-dim", type=int, default=32)
    args = parser.parse_args()

    if __package__:
        from .data_loader import load_model_inputs
    else:
        from data_loader import load_model_inputs

    data = load_model_inputs(
        city=args.city,
        project_root=args.project_root,
        device=args.device,
        save_metadata=False,
    )
    model = MMGCF(
        data,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        attention_dim=args.attention_dim,
    ).to(data.device)

    model.eval()
    with torch.inference_mode():
        output = model()

    print("=" * 72)
    print("STEP 5.2 - MMGCF MODEL CONSTRUCTION")
    print("=" * 72)
    print(f"Device: {data.device}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"User embeddings: {tuple(output.user_embeddings.shape)}")
    print(f"Item embeddings: {tuple(output.item_embeddings.shape)}")
    print(f"Modality weights [ID, text, image]: {tuple(output.view_weights.shape)}")
    print(f"Cold-start items using text/image: {int(model.cold_item_mask.sum().item())}")
    print("Model initialized. Parameters have not been trained.")


if __name__ == "__main__":
    main()
