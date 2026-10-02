"""Late-fusion MMGCF adapted to the Airbnb ModelInputs contract.

Inspired by https://github.com/giuspillo/MMGCF/blob/main/mmgcf/src/mmgcf.py
This is an independent implementation using PyTorch sparse COO tensors; it
does not copy the upstream source or require PyTorch Geometric.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import functional as F

if __package__:
    from .mmgcf import MMGCFOutput
else:
    from mmgcf import MMGCFOutput

if TYPE_CHECKING:
    from .data_loader import ModelInputs


class LateFusionMMGCF(nn.Module):
    """LightGCN on train edges, then ID/text/image item fusion.

    Content features are projected into the collaborative space. Every item
    representation is L2-normalized before scoring, so zero-degree items do
    not win or lose solely because their vector norm differs from warm items.
    """

    def __init__(
        self,
        inputs: ModelInputs,
        embedding_dim: int = 64,
        num_layers: int = 2,
        id_weight: float = 0.5,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0 or num_layers < 0:
            raise ValueError("embedding_dim must be positive and num_layers nonnegative.")
        if not 0.0 <= id_weight < 1.0:
            raise ValueError("id_weight must be in [0, 1) so content can be learned.")

        self.num_users = int(inputs.num_users)
        self.num_items = int(inputs.num_items)
        self.num_layers = int(num_layers)
        self.id_weight = float(id_weight)
        num_nodes = self.num_users + self.num_items
        if tuple(inputs.normalized_adj.shape) != (num_nodes, num_nodes):
            raise ValueError("normalized_adj shape does not match the user/item counts.")
        if inputs.normalized_adj.layout != torch.sparse_coo:
            raise ValueError("normalized_adj must be sparse COO.")
        if inputs.text_features.ndim != 2 or inputs.text_features.shape[0] != self.num_items:
            raise ValueError("text_features must contain one row per item.")
        if inputs.image_features.ndim != 2 or inputs.image_features.shape[0] != self.num_items:
            raise ValueError("image_features must contain one row per item.")
        if tuple(inputs.node_degree.shape) != (num_nodes,):
            raise ValueError("node_degree shape does not match graph size.")

        self.register_buffer("normalized_adj", inputs.normalized_adj.coalesce(), persistent=False)
        self.register_buffer("text_features", inputs.text_features.detach().clone(), persistent=False)
        self.register_buffer("image_features", inputs.image_features.detach().clone(), persistent=False)
        self.register_buffer("cold_item_mask", inputs.node_degree[self.num_users:].eq(0), persistent=False)

        self.id_embedding = nn.Embedding(num_nodes, embedding_dim)
        self.text_projection = nn.Linear(inputs.text_features.shape[1], embedding_dim, bias=False)
        self.image_projection = nn.Linear(inputs.image_features.shape[1], embedding_dim, bias=False)
        nn.init.normal_(self.id_embedding.weight, std=0.1)
        nn.init.xavier_uniform_(self.text_projection.weight)
        nn.init.xavier_uniform_(self.image_projection.weight)

    def forward(self) -> MMGCFOutput:
        current = self.id_embedding.weight
        summed = current
        for _ in range(self.num_layers):
            current = torch.sparse.mm(self.normalized_adj, current)
            summed = summed + current
        graph_embeddings = summed / (self.num_layers + 1)
        user_embeddings = graph_embeddings[:self.num_users]
        item_id = F.normalize(graph_embeddings[self.num_users:], dim=-1)

        text = F.normalize(self.text_projection(self.text_features), dim=-1)
        image = F.normalize(self.image_projection(self.image_features), dim=-1)
        content = F.normalize((text + image) / 2, dim=-1)
        warm = F.normalize(
            self.id_weight * item_id + (1.0 - self.id_weight) * content,
            dim=-1,
        )
        item_embeddings = torch.where(self.cold_item_mask[:, None], content, warm)

        weights = torch.empty(
            (self.num_users + self.num_items, 3),
            dtype=user_embeddings.dtype,
            device=user_embeddings.device,
        )
        weights[:self.num_users] = torch.tensor([1.0, 0.0, 0.0], device=weights.device)
        weights[self.num_users:] = torch.tensor(
            [self.id_weight, (1.0 - self.id_weight) / 2, (1.0 - self.id_weight) / 2],
            device=weights.device,
        )
        weights[self.num_users:][self.cold_item_mask] = torch.tensor(
            [0.0, 0.5, 0.5], device=weights.device
        )
        return MMGCFOutput(user_embeddings, item_embeddings, weights)

    def score_all_items(
        self, user_idx: torch.Tensor, output: MMGCFOutput | None = None
    ) -> torch.Tensor:
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
        if user_idx.ndim != 1 or positive_item_idx.shape != user_idx.shape:
            raise ValueError("user_idx and positive_item_idx must have shape [batch].")
        if negative_item_idx.ndim not in (1, 2) or negative_item_idx.shape[0] != len(user_idx):
            raise ValueError("negative_item_idx must have shape [batch] or [batch, negatives].")
        if user_idx.numel() == 0 or negative_item_idx.numel() == 0:
            raise ValueError("BPR batch must not be empty.")
        if l2_weight < 0:
            raise ValueError("l2_weight must be nonnegative.")

        output = self()
        users = output.user_embeddings[user_idx]
        positive = output.item_embeddings[positive_item_idx]
        negative = output.item_embeddings[negative_item_idx]
        pos_scores = (users * positive).sum(-1)
        if negative_item_idx.ndim == 1:
            neg_scores = (users * negative).sum(-1)
        else:
            neg_scores = (users[:, None, :] * negative).sum(-1)
            pos_scores = pos_scores[:, None]
        loss = F.softplus(neg_scores - pos_scores).mean()
        if l2_weight:
            raw_user = self.id_embedding(user_idx)
            raw_pos = self.id_embedding(self.num_users + positive_item_idx)
            raw_neg = self.id_embedding(self.num_users + negative_item_idx)
            loss = loss + 0.5 * l2_weight * (
                raw_user.square().sum(-1).mean()
                + raw_pos.square().sum(-1).mean()
                + raw_neg.square().sum(-1).mean()
            )
        return loss
