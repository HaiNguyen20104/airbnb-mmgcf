# -*- coding: utf-8 -*-
"""
fusion.py
=========
Các cơ chế kết hợp đa phương thức của MMGCF.

Hai nhóm:
  1) Bản gốc MMGCF (UMAP'26):
       - weighting: alpha (scalar toàn cục) | normalized | equal
       - fusion:    mean | sum | prod | concat
     -> tái tạo 1-1 từ mmgcf.py chính thức của tác giả.
  2) Bản cải tiến:
       - DemandGate: khuếch/kỳch từng modality theo mức được review xác nhận (G1, mục tiêu #3)
       - AttentionFusion: chú ý thích ứng THEO TỪNG THỰC THỂ (item) + gate ID↔MM (G3, mục tiêu #2)
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------
# Fusion nguyên thuỷ (tái tạo từ code gốc swapUniba/MMGCF)
# ------------------------------------------------------------------
def fuse_mean(stacked):
    return torch.mean(stacked, dim=0)


def fuse_sum(stacked):
    return torch.sum(stacked, dim=0)


def fuse_prod(stacked):
    return torch.prod(stacked, dim=0)


def fuse_concat(stacked, layer):
    """concat rồi chiếu về d bằng Linear (đối ứng LazyLinear trong bản gốc).

    stacked: (M, n_items, d) -> flat (n_items, M*d) -> Linear -> (n_items, d)
    """
    M = stacked.shape[0]
    flat = stacked.transpose(0, 1).reshape(stacked.shape[1], M * stacked.shape[2])
    return layer(flat)


FUSION_FN = {
    "mean": fuse_mean,
    "sum": fuse_sum,
    "prod": fuse_prod,
}


class OriginalFusion(nn.Module):
    """
    Late Fusion kiểu MMGCF gốc:
      - weighting ∈ {alpha, normalized, equal} áp lên stack [id_emb, mm_1..mm_M]
      - fusion ∈ {mean, sum, prod, concat}
    """

    def __init__(self, embedding_dim, num_modalities, weighting, fusion):
        super().__init__()
        if weighting not in {"alpha", "normalized", "equal"}:
            raise ValueError(f"weighting không hợp lệ: {weighting}")
        if fusion not in {"mean", "sum", "prod", "concat"}:
            raise ValueError(f"fusion không hợp lệ: {fusion}")

        self.weighting = weighting
        self.fusion_name = fusion
        self.num_modalities = num_modalities

        if fusion == "concat":
            self.reduce = nn.Linear((num_modalities + 1) * embedding_dim, embedding_dim)

        # alpha: 1 scalar toàn cục cho TOÀN BỘ item (đúng bản gốc)
        if weighting == "alpha":
            self.mm_weight = nn.Parameter(torch.zeros(1))

    def _fuse(self, stacked):
        if self.fusion_name == "concat":
            return fuse_concat(stacked, self.reduce)
        return FUSION_FN[self.fusion_name](stacked)

    def forward(self, id_emb, mm_list):
        if not mm_list:
            return id_emb, None

        M = len(mm_list)

        if self.weighting == "alpha":
            alpha = torch.sigmoid(self.mm_weight)
            stacked = torch.stack(
                [id_emb * alpha] + [emb * (1 - alpha) for emb in mm_list]
            )
            return self._fuse(stacked), None

        if self.weighting == "normalized":
            # Đúng công thức bản gốc: norm(id)*M, còn lại norm(mm)
            stacked = torch.stack(
                [F.normalize(id_emb) * M] + [F.normalize(emb) for emb in mm_list]
            )
            return self._fuse(stacked), None

        # equal: gộp mm trước bằng cùng fusion, rồi gộp [id, fused_mm]
        fused_mm = self._fuse(torch.stack(mm_list)) if M > 1 else mm_list[0]
        stacked = torch.stack([id_emb, fused_mm])
        return self._fuse(stacked), None


# ------------------------------------------------------------------
# Demand gate (G1): review "bên cầu" làm bộ lọc nhiễu "bên cung"
# ------------------------------------------------------------------
class DemandGate(nn.Module):
    """
    Cross-attention 1 lớp: Query = profile review của listing,
    Keys = từng modality bên cung (host). Modality nào KHÔNG được nội dung
    review thực tế của khách xác nhận -> bị hạ trọng số (khử "thổi phồng").

    Trả về list modality embedding ĐÃ scale: e_m * (1 + beta*(M*w_m - 1))
    (trọng số w có tổng 1 nên hệ số nằm trong [1-beta, 1+beta], vừa khuếch vừa kỳch),
    sau đó L2-chuẩn hoá lại từng vector.
    """

    def __init__(self, embedding_dim, beta=0.5):
        super().__init__()
        if not 0.0 <= beta <= 1.0:
            raise ValueError(f"beta_demand phải trong [0,1], nhận {beta}.")
        self.beta = beta
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.scale = 1.0 / math.sqrt(embedding_dim)

    def forward(self, review_emb, modality_embs):
        if not modality_embs:
            return modality_embs

        q = self.query(review_emb)                       # (n_items, d)
        keys = torch.stack([self.key(emb) for emb in modality_embs])  # (M, n_items, d)
        logits = (q.unsqueeze(0) * keys).sum(-1) * self.scale        # (M, n_items)
        weights = F.softmax(logits, dim=0)                            # (M, n_items)

        M = len(modality_embs)
        out = []
        for m, emb in enumerate(modality_embs):
            factor = 1.0 + self.beta * (M * weights[m].unsqueeze(-1) - 1.0)
            out.append(F.normalize(emb * factor, dim=-1))
        return out


# ------------------------------------------------------------------
# Attention fusion theo thực thể (G3)
# ------------------------------------------------------------------
class AttentionFusion(nn.Module):
    """
    Thay phép cộng trung bình cào bằng của MMGCF gốc bằng:

      1) Chú ý đa phương thức THEO TỪNG ITEM:
         ctx = [ê_id ‖ mean(ê_m)] -> MLP -> logits -> softmax (có mask)
         e_mm_fused(i) = Σ α_m(i) · ê_m(i)
      2) Gate ID↔MM theo từng item (vector theo chiều d):
         g(i) = σ(MLP([ê_id ‖ e_mm_fused]))
         e_final(i) = g·ê_id + (1-g)·e_mm_fused

    Khi không còn modality nào (mất hết), tự động rơi về embedding ID.
    """

    def __init__(self, embedding_dim, num_modalities, hidden_dim=None):
        super().__init__()
        if num_modalities <= 0:
            raise ValueError("Cần ít nhất 1 modality.")
        hidden_dim = hidden_dim or max(embedding_dim // 2, 16)

        self.num_modalities = num_modalities
        self.attn = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_modalities),
        )
        self.gate = nn.Linear(embedding_dim * 2, embedding_dim)
        self.last_attention = None  # để xuất thống kê diễn giải sau training

    def forward(self, id_emb, mm_list):
        if not mm_list:
            return id_emb, None

        mm_mean = torch.stack(mm_list, dim=0).mean(dim=0)
        ctx = torch.cat([id_emb, mm_mean], dim=1)
        logits = self.attn(ctx)                       # (n_items, M)
        alpha = F.softmax(logits, dim=1)              # softmax có cùng mask theo dataset

        stacked = torch.stack(mm_list, dim=0)         # (M, n_items, d)
        e_mm = (stacked * alpha.t().unsqueeze(-1)).sum(dim=0)  # (n_items, d)

        g = torch.sigmoid(self.gate(torch.cat([id_emb, e_mm], dim=1)))
        final = g * id_emb + (1.0 - g) * e_mm

        # Lưu alpha (detach) chỉ để thống kê diễn giải; e_mm GIỮ gradient
        # để InfoNCE có thể kéo cả hai góc nhìn về nhau (G4).
        self.last_attention = alpha.detach()
        return final, e_mm
