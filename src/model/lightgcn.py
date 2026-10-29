# -*- coding: utf-8 -*-
"""
lightgcn.py
===========
Backbone LightGCN thuần PyTorch, tái tạo đúng công thức bài báo gốc:

    E^(k+1) = A_hat @ E^(k),  E_final = mean(E^(0..K))

Không dùng torch_geometric — tự lan truyền bằng norm_adj (sparse COO) đã chuẩn
bị sẵn ở src/graph/build_bipartite_graph.py (A_hat = D^(-1/2) A D^(-1/2)).
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch
import torch.nn as nn


class LightGCNBackbone(nn.Module):
    """Embedding table + lan truyền LightGCN trên đồ thị user-item đã chuẩn hoá."""

    def __init__(self, num_nodes, num_users, embedding_dim, n_layers, norm_adj):
        super().__init__()
        if norm_adj is None:
            raise ValueError("norm_adj là bắt buộc (sparse COO tensor).")
        if norm_adj.layout != torch.sparse_coo:
            raise ValueError("norm_adj phải là sparse COO tensor.")
        if norm_adj.shape != (num_nodes, num_nodes):
            raise ValueError(
                f"norm_adj shape {tuple(norm_adj.shape)} != ({num_nodes}, {num_nodes})."
            )

        self.num_nodes = num_nodes
        self.num_users = num_users
        self.n_layers = n_layers

        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        nn.init.xavier_uniform_(self.embedding.weight)

        # Đồ thị là hằng số theo training -> lưu làm buffer (di chuyển theo device).
        self.register_buffer("norm_adj", norm_adj.coalesce())

    def propagate(self):
        """Lan truyền đầy đủ: trả (num_nodes, d) = mean của E^0..E^n_layers."""
        emb = self.embedding.weight
        layers = [emb]
        for _ in range(self.n_layers):
            emb = torch.sparse.mm(self.norm_adj, emb)
            layers.append(emb)
        return torch.stack(layers, dim=0).mean(dim=0)

    def get_user_item(self):
        """Tách (user_emb, item_emb) từ embedding sau lan truyền."""
        all_emb = self.propagate()
        return all_emb[: self.num_users], all_emb[self.num_users :]

    def layer_wise(self):
        """Trả toàn bộ danh sách layers (dùng cho debug/đo độ sâu)."""
        emb = self.embedding.weight
        layers = [emb]
        for _ in range(self.n_layers):
            emb = torch.sparse.mm(self.norm_adj, emb)
            layers.append(emb)
        return layers
