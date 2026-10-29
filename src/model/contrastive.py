# -*- coding: utf-8 -*-
"""
contrastive.py
==============
Học tương phản tự giám sát đa góc nhìn (G4 - mục tiêu #4).

InfoNCE hai chiều (NT-Xent) để gióng hàng giữa:
  - góc nhìn hành vi đồ thị: embedding item sau lan truyền LightGCN,
  - góc nhìn đa phương thức: embedding item hợp nhất từ nội dung.
Cặp dương = chính item đó qua 2 góc nhìn; âm = các item khác trong batch.
Khắc phục hạn chế late fusion + hàm mất mát đơn nhiệm (BPR) của MMGCF gốc.
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch
import torch.nn.functional as F


def info_nce_loss(z_graph, z_mm, temperature=0.2):
    """
    z_graph, z_mm: (B, d) — hai góc nhìn của CÙNG các item, đã cùng thứ tự.
    Trả về scalar loss (trung bình 2 chiều G->M và M->G).
    """
    if z_graph.shape != z_mm.shape or z_graph.dim() != 2:
        raise ValueError(f"z shape không khớp: {z_graph.shape} vs {z_mm.shape}.")
    if z_graph.shape[0] < 2:
        return z_graph.new_zeros(())

    z1 = F.normalize(z_graph, dim=-1)
    z2 = F.normalize(z_mm, dim=-1)

    logits = z1 @ z2.t() / temperature   # (B, B)
    labels = torch.arange(z1.shape[0], device=z1.device)

    loss_g2m = F.cross_entropy(logits, labels)
    loss_m2g = F.cross_entropy(logits.t(), labels)
    return (loss_g2m + loss_m2g) / 2.0
