# -*- coding: utf-8 -*-
"""
config.py
=========
Cấu hình trung tâm cho toàn bộ pipeline MMGCF cải tiến (G0 -> G6).

Mọi script mới (model/train/eval/features/graph) đều import từ đây để:
  - dùng chung đường dẫn thư mục,
  - dùng chung bộ hyper-parameter mặc định,
  - ghi đè bằng override dict khi chạy ablation từ run_all.py.

Không phụ thuộc TOML ngoài; config là dataclass thuần Python.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import sys


# Console Windows có thể dùng mã hóa cp1258/cp936 — ép UTF-8 để in được tiếng Việt.
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# ------------------------------------------------------------------
# Đường dẫn chung
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw" / "bangkok"
PROCESSED_DIR = DATA_DIR / "processed" / "bangkok"
FEATURES_DIR = PROCESSED_DIR / "features"
GRAPH_DIR = PROCESSED_DIR / "graph"
EXTERNAL_DIR = DATA_DIR / "external" / "bangkok"          # audio/video theo khu vực
RESULTS_DIR = BASE_DIR / "results"                        # metrics/checkpoint/ablation

LISTINGS_FINAL_PATH = PROCESSED_DIR / "listings_final.parquet"
REVIEWS_RAW_PATH = RAW_DIR / "reviews.csv.gz"
ITEM_MAPPING_PATH = FEATURES_DIR / "item_mapping.parquet"
USER_MAPPING_PATH = FEATURES_DIR / "user_mapping.parquet"

TEXT_FEATURES_PATH = FEATURES_DIR / "text_features.npy"
IMAGE_FEATURES_PATH = FEATURES_DIR / "image_features.npy"
AUDIO_FEATURES_PATH = FEATURES_DIR / "audio_features.npy"
VIDEO_FEATURES_PATH = FEATURES_DIR / "video_features.npy"

REVIEW_EMB_PATH = FEATURES_DIR / "review_embeddings.f16.npy"   # 693k x 512, fp16, có resume
REVIEW_META_PATH = FEATURES_DIR / "review_embeddings_meta.json"
REVIEW_PROFILE_PATH = FEATURES_DIR / "review_profile.npy"      # (n_items, 512) đã time-decay
REVIEW_COUNT_PATH = FEATURES_DIR / "review_count.npy"
REVIEW_GAP_REPORT_PATH = RESULTS_DIR / "review_gap_report.csv"

AUDIO_DISTRICT_PATH = FEATURES_DIR / "audio_district_features.npy"
VIDEO_DISTRICT_PATH = FEATURES_DIR / "video_district_features.npy"
MODALITY_STATUS_PATH = FEATURES_DIR / "modality_status.json"

NORM_ADJ_PATH = GRAPH_DIR / "norm_adj.pt"
NORM_ADJ_AUG_PATH = GRAPH_DIR / "norm_adj_aug.pt"
COLD_EDGES_REPORT_PATH = RESULTS_DIR / "cold_edges_report.csv"


# ------------------------------------------------------------------
# Cấu hình model / training
# ------------------------------------------------------------------
@dataclass
class Config:
    # --- Chung ---
    seed: int = 42
    device: str = "auto"                 # auto | cuda | mps | cpu
    run_name: str = "run"

    # --- Model ---
    embedding_dim: int = 64
    n_layers: int = 2                    # đúng bản gốc feat/mmgcf (2 lớp lan truyền)
    multimodal: bool = True              # False -> LightGCN thuần
    modalities: tuple = ("text", "image", "audio", "video")  # danh sách KHẢ DỤNG cần nạp
    freeze_mm: bool = False              # True = tái tạo gốc (projection ngẫu nhiên đóng băng)

    # --- Fusion / weighting ---
    fusion_modalities: str = "attention"  # mean | sum | prod | concat | attention
    weighting: str = "alpha"              # alpha | normalized | equal | original (original = tái tạo trung thực bản gốc)
    id_weight: float = 0.25               # trọng số id trong công thức gốc: id·w + (1-w)·content (chỉ mode "original")

    # --- Demand-side (G1 - mục tiêu #3) ---
    use_demand: bool = True               # review profile làm modality "bên cầu" + gate
    beta_demand: float = 0.5              # biên độ khuếch/kỳch modality theo review (0 = tắt gate)

    # --- Contrastive (G4 - mục tiêu #4) ---
    lambda_contrast: float = 0.0          # trọng số InfoNCE; 0 = tắt
    temperature: float = 0.2

    # --- Cold-start (G5 - mục tiêu #5) ---
    cold_aug: bool = False                # thêm cạnh item-item nội dung cho item lạnh
    cold_deg_threshold: int = 2           # item có degree train <= ngưỡng coi là lạnh
    cold_k: int = 5
    cold_cos_threshold: float = 0.5
    # Kênh nội dung trực tiếp cho item lạnh (thời điểm suy luận - P1):
    # score(u,i) = <u, final_i> + gamma·1[cold(i)]·cos(profile_u, content_i)
    # với profile_u = trung bình content của item trong lịch sử train của user.
    cold_content_channel: bool = False
    cold_gamma: float = 0.5               # trọng số kênh nội dung
    cold_channel_fusion: str = "add"      # add (cộng thẳng) | z (z-score theo user, scale-invariant)
    cold_content_fallback: bool = False   # item lạnh dùng content thuần trong final_item (mang cơ chế gốc vào mode attention)
    cold_cos_eval: bool = False           # chấm điểm theo cos (chuẩn hoá user/item) khi có kênh nội dung — cân bằng thang với kênh cos

    # --- Training ---
    # (siêu tham số khớp protocol bản gốc feat/mmgcf: 150 epochs, patience 12,
    #  l2 1e-4, lr 1e-3, batch 1024, negatives 1, eval mỗi epoch trên Val NDCG@10)
    epochs: int = 150
    learning_rate: float = 1e-3
    batch_size: int = 1024
    lambda_reg: float = 1e-4
    eval_every: int = 1                   # đánh giá trên Val mỗi N epoch
    patience_evals: int = 12              # dừng sớm nếu không cải thiện Val NDCG@10
    top_k: tuple = (5, 10, 20)

    # --- Reviews (G1) ---
    review_decay: float = 0.9             # trọng số giảm theo năm tuổi review
    review_cap: int = 30                  # tối đa review gần nhất mỗi listing
    review_chunk: int = 100_000           # số review mỗi lần encode
    limit_reviews: int = 0                # >0 = chỉ encode N review đầu (smoke test)

    # --- External audio/video (G2) ---
    audio_k_nearest: int = 3              # trộn mượt embedding 3 khu gần nhất theo lat/lng
    freesound_token: str = ""             # để trống -> đọc env FREESOUND_API_TOKEN

    # --- Kết xuất ---
    results_subdir: str = ""              # set lúc chạy; mặc định theo run_name


# ------------------------------------------------------------------
# Thiết bị
# ------------------------------------------------------------------
def resolve_device(name: str = "auto"):
    import torch

    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    return torch.device(name)


# ------------------------------------------------------------------
# Tiện ích
# ------------------------------------------------------------------
def override_config(base: Config, overrides: dict) -> Config:
    """Tạo bản sao Config và ghi đè các trường theo overrides (ablation)."""
    import dataclasses

    cfg = dataclasses.replace(base)
    for key, value in (overrides or {}).items():
        if not hasattr(cfg, key):
            raise KeyError(f"Config không có trường '{key}'.")
        if key == "modalities" and isinstance(value, list):
            value = tuple(value)
        if key == "top_k" and isinstance(value, list):
            value = tuple(value)
        setattr(cfg, key, value)
    return cfg


def config_to_dict(cfg: Config) -> dict:
    d = asdict(cfg)
    d["modalities"] = list(cfg.modalities)
    d["top_k"] = list(cfg.top_k)
    return d


# Cấu hình mặc định dùng chung cho các script feature (import trực tiếp)
DEFAULT_CONFIG = Config()


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False, default=str)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)
