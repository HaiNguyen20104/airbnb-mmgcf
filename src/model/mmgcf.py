# -*- coding: utf-8 -*-
"""
mmgcf.py
========
Mô hình MMGCF cải tiến cho domain Airbnb (thuần PyTorch).

Cấu trúc:
  - Backbone LightGCN (src/model/lightgcn.py) cho embedding ID user/item.
  - Nhánh đa phương thức cho ITEM: text/image (CLIP), audio (CLAP), video (X-CLIP).
  - Demand modality: profile review của khách (bên cầu) tham gia fusion như
    một modality thứ 5 + DemandGate khử nhiễu modality bên cung (G1).
  - Fusion:
      * "gốc":   OriginalFusion (weighting alpha/normalized/equal x mean/sum/prod/concat)
      * "cải tiến": AttentionFusion (chú ý thích ứng theo item + gate ID↔MM)
  - InfoNCE hai chiều giữa góc nhìn đồ thị và góc nhìn đa phương thức (G4).

Hỗ trợ missing modality: modality không có dữ liệu KHÔNG được đưa vào fusion
(mask) — model chạy đúng với số phương thức có mặt (thiết kế modality-agnostic).
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import Config
from model.lightgcn import LightGCNBackbone
from model.fusion import OriginalFusion, AttentionFusion, DemandGate
from model.contrastive import info_nce_loss


HOST_MODALITY_ORDER = ["text", "image", "audio", "video"]
FEATURE_DIM = 512


def load_modality_features(cfg: Config, num_items: int):
    """
    Nạp feature các modality theo cfg.modalities.

    Trả về dict {name: (n_items, 512) float32 L2-norm} chỉ chứa modality CÓ file.
    "review" được nạp riêng khi cfg.use_demand.
    """
    from configs.config import (
        TEXT_FEATURES_PATH, IMAGE_FEATURES_PATH,
        AUDIO_FEATURES_PATH, VIDEO_FEATURES_PATH,
        REVIEW_PROFILE_PATH,
    )

    paths = {
        "text": TEXT_FEATURES_PATH,
        "image": IMAGE_FEATURES_PATH,
        "audio": AUDIO_FEATURES_PATH,
        "video": VIDEO_FEATURES_PATH,
    }

    feats = {}
    for name in cfg.modalities:
        path = paths.get(name)
        if path is None or not path.exists():
            print(f"[MMGCF] Bỏ qua modality '{name}' (không có file: {path}).")
            continue
        arr = np.load(path)
        if arr.ndim != 2 or arr.shape != (num_items, FEATURE_DIM):
            raise ValueError(
                f"{name} features shape {arr.shape} != ({num_items}, {FEATURE_DIM})."
            )
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} features chứa NaN/Inf.")
        feats[name] = torch.from_numpy(arr.astype(np.float32))
        print(f"[MMGCF] Nạp modality '{name}': {tuple(arr.shape)}")

    if cfg.use_demand:
        if REVIEW_PROFILE_PATH.exists():
            profile = np.load(REVIEW_PROFILE_PATH)
            if profile.shape != (num_items, FEATURE_DIM):
                raise ValueError(
                    f"review_profile shape {profile.shape} != ({num_items}, {FEATURE_DIM})."
                )
            if not np.isfinite(profile).all():
                raise ValueError("review_profile chứa NaN/Inf.")
            feats["review"] = torch.from_numpy(profile.astype(np.float32))
            print(f"[MMGCF] Nạp demand modality 'review': {tuple(profile.shape)}")
        else:
            print("[MMGCF] Chưa có review_profile.npy -> tắt demand modality.")

    return feats


class MMGCF(nn.Module):
    """
    MMGCF tổng quát: chạy được cả ở chế độ "gốc" (tái tạo baseline) lẫn "cải tiến".

    Args:
        cfg: Config.
        num_users, num_items: kích thước đồ thị.
        norm_adj: sparse COO (num_users+num_items, num_users+num_items) đã chuẩn hoá.
        modality_feats: dict {name: Tensor (n_items, 512)}; "review" tách riêng.
    """

    def __init__(self, cfg: Config, num_users, num_items, norm_adj, modality_feats,
                 item_degrees=None, train_edges=None):
        super().__init__()
        self.cfg = cfg
        self.num_users = num_users
        self.num_items = num_items

        host_feats = {m: modality_feats[m] for m in HOST_MODALITY_ORDER if m in modality_feats}
        self.multimodal = cfg.multimodal and len(host_feats) > 0
        if not self.multimodal:
            host_feats = {}

        self.backbone = LightGCNBackbone(
            num_users + num_items, num_users, cfg.embedding_dim, cfg.n_layers, norm_adj
        )

        self.host_names = list(host_feats.keys())
        self.improved = cfg.fusion_modalities == "attention"

        # Chế độ tái tạo TRUNG THỰC bản gốc (branch feat/mmgcf của dự án):
        # weighting="original" -> công thức late-fusion hằng số
        #   final = normalize(id_weight·norm(id) + (1-id_weight)·norm(content))
        #   item lạnh (degree=0 trong train) dùng content thuần.
        self.is_original_faithful = (not self.improved) and cfg.weighting == "original"

        # Kênh nội dung trực tiếp cho item lạnh (P1, inference-time):
        # cần item_degrees (để tạo cold mask) và train_edges (để dựng profile user).
        self.use_cold_channel = (
            bool(cfg.cold_content_channel)
            and item_degrees is not None
            and train_edges is not None
            and self.multimodal
        )

        if self.is_original_faithful:
            # Bản gốc khởi tạo id embedding bằng normal(std=0.1), không phải xavier
            nn.init.normal_(self.backbone.embedding.weight, std=0.1)
            if item_degrees is None:
                raise ValueError(
                    "weighting='original' cần item_degrees (độ bậc train của item) "
                    "để tạo cold_item_mask."
                )
            cold = torch.as_tensor(np.asarray(item_degrees)).long().eq(0)
            self.register_buffer("cold_item_mask", cold, persistent=False)
        elif self.use_cold_channel:
            cold = torch.as_tensor(np.asarray(item_degrees)).long().eq(0)
            self.register_buffer("cold_item_mask", cold, persistent=False)

        if self.use_cold_channel:
            cc_edges = torch.as_tensor(np.asarray(train_edges, dtype=np.int64))
            self.register_buffer("cc_edges", cc_edges, persistent=False)
            counts = torch.zeros(num_users, dtype=torch.float32)
            counts.index_add_(0, cc_edges[:, 0], torch.ones(len(cc_edges)))
            self.register_buffer("cc_user_count", counts, persistent=False)

        # Demand modality quyết định TRƯỚC khi dựng fusion (số modality tổng)
        self.use_demand = cfg.use_demand and "review" in modality_feats and bool(self.host_names)

        if self.multimodal:
            # Projection 512 -> d cho từng modality
            # (chế độ faithful: Linear không bias + xavier, đúng bản gốc)
            proj_bias = not self.is_original_faithful
            self.host_proj = nn.ModuleList(
                [nn.Linear(FEATURE_DIM, cfg.embedding_dim, bias=proj_bias)
                 for _ in self.host_names]
            )
            if self.is_original_faithful:
                for proj in self.host_proj:
                    nn.init.xavier_uniform_(proj.weight)
            # Feature gốc đóng băng (buffer)
            for name in self.host_names:
                self.register_buffer(f"feat_{name}", host_feats[name])

            n_mm_total = len(self.host_names) + (1 if self.use_demand else 0)

            if self.improved:
                self.fusion = AttentionFusion(cfg.embedding_dim, n_mm_total)
            else:
                # tái tạo bản gốc: projection LazyLinear chạy 1 lần lúc init rồi freeze
                if cfg.freeze_mm:
                    with torch.no_grad():
                        frozen = [
                            proj(host_feats[name]) for proj, name in
                            zip(self.host_proj, self.host_names)
                        ]
                    self.host_proj = None  # không train projection nữa
                    self.frozen_host = nn.ModuleList(
                        [nn.Embedding.from_pretrained(emb, freeze=True) for emb in frozen]
                    )
                if self.is_original_faithful:
                    # Dùng nhánh late-fusion faithful trong get_embedding,
                    # không dựng OriginalFusion (không có alpha học được).
                    self.fusion = None
                else:
                    self.fusion = OriginalFusion(
                        cfg.embedding_dim, n_mm_total, cfg.weighting, cfg.fusion_modalities
                    )

            if self.use_demand:
                self.register_buffer("review_profile", modality_feats["review"])
                self.review_proj = nn.Linear(FEATURE_DIM, cfg.embedding_dim)
                self.demand_gate = DemandGate(cfg.embedding_dim, cfg.beta_demand)
        else:
            self.host_names = []
            self.host_proj = None
            self.fusion = None
            self.use_demand = False

    # ------------------------------------------------------------------
    # Lan truyền + hợp nhất
    # ------------------------------------------------------------------
    def get_embedding(self):
        """
        Trả về:
          user_emb     : (num_users, d)
          final_item   : (num_items, d) — embedding item dùng để xếp hạng
          graph_view   : (num_items, d) — góc nhìn hành vi đồ thị (pre-fusion)
          content_view : (num_items, d) — góc nhìn đa phương thức (pre-gate)
        """
        user_emb, id_item_emb = self.backbone.get_user_item()

        if not self.multimodal:
            return user_emb, id_item_emb, id_item_emb, None

        if self.is_original_faithful:
            # Công thức late fusion của bản gốc (late_fusion_mmgcf.py):
            #   item_id = norm(graph_emb);  mm_m = norm(proj(feat_m))
            #   content  = norm(mean(mm_m));
            #   warm     = norm(id_weight·item_id + (1-id_weight)·content)
            #   item lạnh (degree=0) -> content thuần. Giữ nguyên gradient cho proj.
            item_id = F.normalize(id_item_emb, dim=-1)
            mm_normed = [
                F.normalize(proj(getattr(self, f"feat_{name}")), dim=-1)
                for proj, name in zip(self.host_proj, self.host_names)
            ]
            content = F.normalize(torch.stack(mm_normed, dim=0).mean(dim=0), dim=-1)
            warm = F.normalize(
                self.cfg.id_weight * item_id + (1.0 - self.cfg.id_weight) * content,
                dim=-1,
            )
            final_item = torch.where(self.cold_item_mask[:, None], content, warm)
            return user_emb, final_item, id_item_emb, content

        if self.cfg.freeze_mm:
            mm_list = [e.weight for e in self.frozen_host]
        else:
            mm_list = [proj(getattr(self, f"feat_{name}")) for proj, name in zip(self.host_proj, self.host_names)]

        content_view = torch.stack(mm_list, dim=0).mean(dim=0)

        if self.use_demand:
            review_emb = self.review_proj(self.review_profile)
            # 1) Gate khử nhiễu bên cung theo review (G1)
            mm_list = self.demand_gate(review_emb, mm_list)
            # 2) Review tham gia fusion như modality "demand"
            mm_list = mm_list + [review_emb]
            content_view = torch.stack(mm_list, dim=0).mean(dim=0)

        final_item, e_mm_fused = self.fusion(id_item_emb, mm_list)
        if e_mm_fused is not None:
            content_view = e_mm_fused

        # P1 (tuỳ chọn): item lạnh dùng content thuần, hiệu chuẩn về độ lớn
        # trung bình của warm final (đưa kênh chấm dot về cùng thang —
        # không đụng thứ hạng warm, tránh flooding/thủng các kênh độ lớn lệch)
        if (self.use_cold_channel and self.cfg.cold_content_fallback
                and not self.is_original_faithful):
            content_n = F.normalize(content_view, dim=-1)
            with torch.no_grad():
                target = final_item[~self.cold_item_mask].norm(dim=-1).median()
            rescaled = content_n * target
            final_item = torch.where(self.cold_item_mask[:, None], rescaled, final_item)

        return user_emb, final_item, id_item_emb, content_view

    # ------------------------------------------------------------------
    # Kênh nội dung trực tiếp cho item lạnh (P1 - mục tiêu #5)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def cold_channel_extras(self, content_view):
        """
        Trạng thái phục vụ kênh nội dung lúc suy luận:
          profile_u  = normalize(mean(content_j), j ∈ train(u))  — "gu" của user
          content_i  = normalize(content_view[i])
        Evaluator cộng gamma·cos(profile_u, content_i) vào cột item lạnh.

        Train KHÔNG bị ảnh hưởng: item lạnh (degree=0) không bao giờ xuất
        hiện ở positi/negative của BPR (âm mẫu chỉ từ catalog train).
        """
        if not getattr(self, "use_cold_channel", False):
            return None
        if content_view is None:
            raise ValueError("use_cold_channel cần content_view khác None.")
        content_n = F.normalize(content_view, dim=-1)
        prof = torch.zeros(
            self.num_users, content_view.shape[1],
            dtype=content_view.dtype, device=content_view.device,
        )
        prof = prof.index_add(0, self.cc_edges[:, 0], content_n[self.cc_edges[:, 1]])
        prof = prof / self.cc_user_count.clamp(min=1.0).unsqueeze(-1)
        prof = F.normalize(prof, dim=-1)
        return {
            "user_profile": prof,
            "content_item": content_n,
            "cold_mask": self.cold_item_mask,
            "gamma": self.cfg.cold_gamma,
            "cos_eval": bool(getattr(self.cfg, "cold_cos_eval", False)),
            "fusion": getattr(self.cfg, "cold_channel_fusion", "add"),
        }

    # ------------------------------------------------------------------
    # BPR scoring
    # ------------------------------------------------------------------
    def score_pairs(self, user_emb, item_emb, users, items):
        """score(u,i) = <user_emb[u], item_emb[i]>."""
        return (user_emb[users] * item_emb[items]).sum(dim=-1)

    def recommendation_loss(self, user_emb, item_emb, pos_users, pos_items, neg_items, lambda_reg):
        """BPR + L2 reg trên các embedding tham gia batch (chuẩn LightGCN)."""
        pos_score = self.score_pairs(user_emb, item_emb, pos_users, pos_items)
        neg_score = self.score_pairs(user_emb, item_emb, pos_users, neg_items)
        loss = (-F.logsigmoid(pos_score - neg_score)).mean()

        if self.is_original_faithful:
            # Bản gốc reg trên RAW id embeddings (trước lan truyền) với hệ số 0.5:
            #   loss + 0.5·λ·[mean(||raw_u||²) + mean(||raw_pos||²) + mean(||raw_neg||²)]
            raw = self.backbone.embedding.weight
            reg = (
                raw[pos_users].pow(2).sum(-1).mean()
                + raw[self.num_users + pos_items].pow(2).sum(-1).mean()
                + raw[self.num_users + neg_items].pow(2).sum(-1).mean()
            )
            return loss + 0.5 * lambda_reg * reg, pos_score, neg_score

        reg = (
            user_emb[pos_users].pow(2).sum(-1)
            + item_emb[pos_items].pow(2).sum(-1)
            + item_emb[neg_items].pow(2).sum(-1)
        ).mean()
        return loss + lambda_reg * reg, pos_score, neg_score

    def contrastive_loss(self, graph_view, content_view, batch_items):
        """InfoNCE giữa 2 góc nhìn trên tập item của batch (in-batch negatives)."""
        if self.cfg.lambda_contrast <= 0 or content_view is None:
            return None
        items = torch.unique(batch_items)
        return info_nce_loss(
            graph_view[items], content_view[items], self.cfg.temperature
        )

    # ------------------------------------------------------------------
    # Thống kê diễn giải (cho báo cáo G3)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def attention_summary(self):
        """Trung bình alpha theo modality (chỉ khi dùng AttentionFusion)."""
        if not self.multimodal or not self.improved or not isinstance(self.fusion, AttentionFusion):
            return None
        # Chạy 1 lần lan truyền để cập nhật last_attention
        self.get_embedding()
        if self.fusion.last_attention is None:
            return None
        alpha = self.fusion.last_attention.mean(dim=0)  # (M,)
        names = self.host_names + (["review"] if self.use_demand else [])
        return dict(zip(names, [float(v) for v in alpha]))
