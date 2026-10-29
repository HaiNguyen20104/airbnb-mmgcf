# -*- coding: utf-8 -*-
"""
run_all.py
==========
LỆNH CHẠY DUY NHẤT cho toàn bộ pipeline MMGCF cải tiến (G0 -> G6):

    python src/run_all.py                  # chạy đầy đủ
    python src/run_all.py --smoke          # chạy nhanh kiểm tra (giảm quy mô)
    python src/run_all.py --force          # tính lại mọi bước kể cả đã có kết quả

Các bước theo thứ tự:
  0) Preflight   : kiểm tra gói/phần cứng + dữ liệu thô data/raw/bangkok.
  0.5) Data prep : 11 script gốc (lọc listings -> k-core -> tải ảnh CLIP ->
                   split train/val/test -> dựng norm_adj), tự bỏ qua bước
                   đã có output. Chỉ chạy lần đầu từ dữ liệu thô.
  1) G1 Reviews  : encode 693k comments bằng CLIP text (có resume), gộp
                   time-decay profile, báo cáo gap supply-demand.
  2) G2 External : audio (Freesound CC0 -> CLAP) và video (YouTube -> X-CLIP)
                   cấp khu vực — tự bỏ qua lịch sự khi thiếu token/công cụ.
  3) G5 Cold-aug : cạnh item-item nội dung (đã khử nhiễu review) cho item lạnh.
  4) Training    : chuỗi ablation — LightGCN, MMGCF gốc (late-fusion 2 mod),
                   +demand, +attention fusion, +InfoNCE, +cold-aug,
                   +video, 4-modal đầy đủ.
  5) Tổng hợp    : bảng ablation (HR/NDCG tổng + phân tầng cold) + manifest.

Kết quả lưu tại results/ (metrics.json mỗi config, ablation_matrix.csv,
run_manifest.json, review_gap_report.csv, cold_edges_report.csv).
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd
import scipy.sparse as sp
import torch

from configs.config import (
    Config, override_config, config_to_dict, resolve_device, save_json,
    GRAPH_DIR, FEATURES_DIR, RESULTS_DIR, NORM_ADJ_PATH, NORM_ADJ_AUG_PATH,
    REVIEW_PROFILE_PATH, REVIEW_META_PATH, MODALITY_STATUS_PATH, TEXT_FEATURES_PATH,
    IMAGE_FEATURES_PATH, AUDIO_FEATURES_PATH, VIDEO_FEATURES_PATH,
)
from model.mmgcf import MMGCF
from train.train import train_model, set_seed
from evaluation.evaluator import evaluate_ranking, stratified_metrics

BASE_DIR = SRC_DIR.parent

# ------------------------------------------------------------------
# Chuỗi ablation (G6)
# ------------------------------------------------------------------
ABLATION_DEFS = [
    {"name": "lightgcn",
     "desc": "LightGCN thuần (không multimodal) — mốc so sánh CF",
     "overrides": dict(multimodal=False)},
    {"name": "mmgcf_orig",
     "desc": "MMGCF gốc trung thực (late-fusion, id_weight=0.25): 0.25·id + 0.375·text + 0.375·image, projection có học, item lạnh dùng content thuần",
     "overrides": dict(modalities=("text", "image"), fusion_modalities="mean",
                       weighting="original", freeze_mm=False, use_demand=False,
                       lambda_contrast=0.0, cold_aug=False)},
    {"name": "mmgcf_demand",
     "desc": "+ Demand modality & gate review (G1, mục tiêu #3)",
     "overrides": dict(modalities=("text", "image"), fusion_modalities="mean",
                       weighting="alpha", freeze_mm=False, use_demand=True,
                       lambda_contrast=0.0, cold_aug=False)},
    {"name": "mmgcf_attn",
     "desc": "+ Fusion chú ý thích ứng theo thực thể (G3, mục tiêu #2)",
     "overrides": dict(modalities=("text", "image"), fusion_modalities="attention",
                       freeze_mm=False, use_demand=True, lambda_contrast=0.0,
                       cold_aug=False)},
    {"name": "mmgcf_attn_infonce",
     "desc": "+ InfoNCE hai chiều đồ thị↔multimodal (G4, mục tiêu #4)",
     "overrides": dict(modalities=("text", "image"), fusion_modalities="attention",
                       freeze_mm=False, use_demand=True, lambda_contrast=0.01,
                       cold_aug=False)},
    {"name": "mmgcf_coldaug",
     "desc": "+ Cạnh nội dung item-item cho item lạnh (G5, mục tiêu #5)",
     "overrides": dict(modalities=("text", "image"), fusion_modalities="attention",
                       freeze_mm=False, use_demand=True, lambda_contrast=0.01,
                       cold_aug=True)},
    {"name": "mmgcf_coldaug_video",
     "desc": "+ Video khu vực (X-CLIP) vào coldaug — text+image+video",
     "overrides": dict(modalities=("text", "image", "video"),
                       fusion_modalities="attention", freeze_mm=False,
                       use_demand=True, lambda_contrast=0.01, cold_aug=True),
     "needs": ("video",)},
    {"name": "mmgcf_4mod_full",
     "desc": "4 phương thức đầy đủ (audio/video) + demand + attention + InfoNCE + cold-aug",
     "overrides": dict(modalities=("text", "image", "audio", "video"),
                       fusion_modalities="attention", freeze_mm=False,
                       use_demand=True, lambda_contrast=0.01, cold_aug=True),
     "needs": ("audio", "video")},
    {"name": "mmgcf_coldcc",
     "desc": "MMGCF gốc faithful + kênh nội dung trực tiếp cho item lạnh (P1, inference-time)",
     "overrides": dict(modalities=("text", "image"), fusion_modalities="mean",
                       weighting="original", freeze_mm=False, use_demand=False,
                       lambda_contrast=0.0, cold_aug=False,
                       cold_content_channel=True)},
]

MANIFEST_PATH = RESULTS_DIR / "run_manifest.json"
ABLATION_CSV_PATH = RESULTS_DIR / "ablation_matrix.csv"

RAW_BANGKOK_DIR = BASE_DIR / "data" / "raw" / "bangkok"

# ------------------------------------------------------------------
# Chuỗi data-prep gốc (raw -> graph/features) cho lần chạy từ đầu
# Mỗi bước: script chạy với cwd = repo root (các script gốc dùng đường
# dẫn tương đối) + danh sách output để bỏ qua khi đã có sẵn.
# ------------------------------------------------------------------
DATA_PREP_STEPS = [
    {"name": "filter_listings",
     "script": "src/data/filter_listings.py",
     "desc": "Lọc listings dùng được (giá, hình ảnh, vị trí) từ listings.csv.gz",
     "outputs": ["data/processed/bangkok/listings_image_filtered.parquet"]},
    {"name": "filter_missing",
     "script": "src/data/filter_missing.py",
     "desc": "Loại listings thiếu cột quan trọng",
     "outputs": ["data/processed/bangkok/listings_cleaned.parquet"]},
    {"name": "filter_kcore",
     "script": "src/data/filter_kcore.py",
     "desc": "k-core lọc tương tác user-item sạch",
     "outputs": ["data/processed/bangkok/interactions_kcore.parquet",
                 "data/processed/bangkok/listings_kcore.parquet"]},
    {"name": "download_images",
     "script": "src/features/download_images.py",
     "desc": "Tải ảnh listings về features/images/ (bước nặng nhất, có resume)",
     "outputs": ["data/processed/bangkok/features/image_download_status.parquet"]},
    {"name": "filter_invalid_images",
     "script": "src/features/filter_invalid_images.py",
     "desc": "Loại listing không có ảnh hợp lệ -> listings_final/interactions_final",
     "outputs": ["data/processed/bangkok/listings_final.parquet",
                 "data/processed/bangkok/interactions_final.parquet"]},
    {"name": "build_item_mapping",
     "script": "src/features/build_item_mapping.py",
     "desc": "Sinh item_mapping.parquet (listing_id -> item_idx)",
     "outputs": ["data/processed/bangkok/features/item_mapping.parquet"]},
    {"name": "build_user_mapping",
     "script": "src/graph/build_user_mapping.py",
     "desc": "Sinh user_mapping.parquet (reviewer_id -> user_idx)",
     "outputs": ["data/processed/bangkok/features/user_mapping.parquet"]},
    {"name": "encode_text_clip",
     "script": "src/features/encode_text_clip.py",
     "desc": "Encode mô tả listings bằng CLIP text -> text_features.npy",
     "outputs": ["data/processed/bangkok/features/text_features.npy"]},
    {"name": "encode_image_clip",
     "script": "src/features/encode_image_clip.py",
     "desc": "Encode ảnh listings bằng CLIP image -> image_features.npy",
     "outputs": ["data/processed/bangkok/features/image_features.npy"]},
    {"name": "split_dataset",
     "script": "src/graph/split_dataset.py",
     "desc": "Chia train/val/test theo user (protocol tác giả)",
     "outputs": ["data/processed/bangkok/graph/train_interactions.parquet",
                 "data/processed/bangkok/graph/val_interactions.parquet",
                 "data/processed/bangkok/graph/test_interactions.parquet"]},
    {"name": "build_bipartite_graph",
     "script": "src/graph/build_bipartite_graph.py",
     "desc": "Dựng norm_adj.pt / edge_index.pt / graph_meta.json",
     "outputs": ["data/processed/bangkok/graph/norm_adj.pt",
                 "data/processed/bangkok/graph/edge_index.pt",
                 "data/processed/bangkok/graph/graph_meta.json"]},
]


# ------------------------------------------------------------------
# Preflight
# ------------------------------------------------------------------
def preflight():
    print("=== BƯỚC 0: PREFLIGHT ===")
    versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }
    device = resolve_device("auto")
    versions["device"] = str(device)

    # Dữ liệu thô bắt buộc để data-prep chạy được
    raw_required = [RAW_BANGKOK_DIR / "listings.csv.gz", RAW_BANGKOK_DIR / "reviews.csv.gz"]
    missing_raw = [str(p.relative_to(BASE_DIR)) for p in raw_required if not p.exists()]
    if missing_raw:
        raise FileNotFoundError(
            "Thiếu dữ liệu thô (dataset Airbnb Bangkok):\n  " + "\n  ".join(missing_raw)
        )

    print(f"[OK] torch {versions['torch']} | CUDA={versions['cuda_available']} | device={device}")
    print("[OK] dữ liệu thô data/raw/bangkok: listings.csv.gz + reviews.csv.gz")
    token = os.environ.get("FREESOUND_API_TOKEN")
    token_file = SRC_DIR / "configs" / "freesound_token.txt"
    if not token and not token_file.exists():
        print("[WARN] chưa có FREESOUND_API_TOKEN -> audio sẽ bị mask (các config khác vẫn chạy).")
    return versions, device


def _run_script(script_relpath):
    """Chạy 1 script data-prep gốc bằng chính interpreter hiện tại.

    cwd = repo root vì các script gốc dùng đường dẫn tương đối
    (Path("data/processed/...")). PYTHONIOENCODING=utf-8 để tránh lỗi
    khi script in dấu tiếng Việt trên console CP1258 của Windows.
    """
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        [sys.executable, script_relpath],
        cwd=str(BASE_DIR), env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Script {script_relpath} thất bại (exit {result.returncode}) — "
            "xem log phía trên."
        )


def step_data_prep(args, statuses):
    print("\n=== BƯỚC 0.5: DATA PREP (raw -> graph/features) ===")
    if args.smoke:
        statuses["data_prep"] = "skipped (smoke mode)"
        print("[SMOKE] bỏ qua data prep (dùng dữ liệu đã chuẩn bị từ lần chạy đầy đủ).")
        return

    status = statuses.setdefault("data_prep_steps", {})
    total = len(DATA_PREP_STEPS)
    for i, step in enumerate(DATA_PREP_STEPS, 1):
        outputs = [BASE_DIR / rel for rel in step["outputs"]]
        if (not args.force) and args.skip_existing and all(p.exists() for p in outputs):
            print(f"[{i:>2}/{total}] SKIP   {step['name']} (đã có output)")
            status[step["name"]] = "skipped (exists)"
            continue
        desc = step["desc"]
        if step["name"] == "download_images":
            desc += " — 30-90 phút lần đầu, nghỉ giữa chừng rồi chạy lại sẽ resume"
        print(f"[{i:>2}/{total}] RUN    {step['name']}: {desc}")
        t0 = time.perf_counter()
        try:
            _run_script(step["script"])
        except Exception:
            status[step["name"]] = "failed"
            raise
        status[step["name"]] = f"ok ({time.perf_counter() - t0:.0f}s)"
    statuses["data_prep"] = "ok"


def ensure_pipeline_inputs(smoke=False):
    """Chốt chặn sau data-prep: mọi file cho training phải tồn tại."""
    required = [
        GRAPH_DIR / "norm_adj.pt",
        GRAPH_DIR / "train_interactions.parquet",
        GRAPH_DIR / "val_interactions.parquet",
        GRAPH_DIR / "test_interactions.parquet",
        FEATURES_DIR / "item_mapping.parquet",
        TEXT_FEATURES_PATH,
        IMAGE_FEATURES_PATH,
    ]
    missing = [str(p.relative_to(BASE_DIR)) for p in required if not p.exists()]
    if missing:
        hint = (
            "Đang chạy --smoke nhưng chưa có dữ liệu đã chuẩn bị. "
            "Hãy chạy đầy đủ trước: python src/run_all.py\n  "
            if smoke
            else "Thiếu file đầu vào sau data-prep (xem log bước 0.5 phía trên):\n  "
        )
        raise FileNotFoundError(hint + "\n  ".join(missing))


# ------------------------------------------------------------------
# Bước 1 — G1: review profile
# ------------------------------------------------------------------
def step_reviews(args, statuses):
    """
    Skip CHỈ khi profile hiện có khớp quy mô mong muốn (meta.limit == target).
    Profile dở từ smoke (limit>0) sẽ được build lại khi chạy đầy đủ.
    """
    target_limit = args.limit_reviews or (15_000 if args.smoke else 0)

    if REVIEW_PROFILE_PATH.exists() and not args.force:
        limit_meta = None
        if REVIEW_META_PATH.exists():
            try:
                limit_meta = json.loads(
                    REVIEW_META_PATH.read_text(encoding="utf-8")
                ).get("limit")
            except (json.JSONDecodeError, OSError):
                limit_meta = None
        if limit_meta is not None and limit_meta == target_limit:
            print("\n=== BƯỚC 1 (G1): review profile đã có và khớp quy mô — bỏ qua ===")
            statuses["g1_reviews"] = "skipped (exists)"
            return
        print("\n[REVIEW] Profile cũ không khớp quy mô "
              f"(meta.limit={limit_meta}, cần {target_limit}) — build lại.")
    if not REVIEW_PROFILE_PATH.exists():
        print("\n=== BƯỚC 1 (G1): BUILD REVIEW PROFILE ===")

    from features.build_review_profile import main as build_reviews
    build_reviews(limit=target_limit, resume=not args.force, skip_gap=False)
    statuses["g1_reviews"] = "ok"


# ------------------------------------------------------------------
# Bước 2 — G2: audio/video ngoại vi (best effort)
# ------------------------------------------------------------------
def step_external(args, statuses):
    if args.no_external:
        print("\n=== BƯỚC 2 (G2): bỏ qua audio/video theo yêu cầu --no-external ===")
        statuses["g2_external"] = "disabled by flag"
        return

    print("\n=== BƯỚC 2 (G2): EXTERNAL AUDIO/VIDEO (best effort) ===")

    # --- Audio: cần token Freesound ---
    try:
        from features.fetch_audio_freesound import fetch_all as fetch_audio, get_token
        if not get_token():
            statuses["g2_audio_fetch"] = "skipped (no FREESOUND_API_TOKEN)"
            print("[AUDIO] Không có token -> bỏ qua.")
        else:
            fetch_audio(limit_districts=3 if args.smoke else None,
                        resume=not args.force)
            statuses["g2_audio_fetch"] = "ok"
        from features.encode_audio_clap import main as encode_audio
        encode_audio()
        statuses["g2_audio_encode"] = "ok"
    except Exception as exc:
        statuses["g2_audio"] = f"failed: {exc}"
        traceback.print_exc()

    # --- Video: cần yt-dlp + ffmpeg ---
    try:
        import yt_dlp  # noqa: F401
        from features.fetch_video_districts import fetch_all as fetch_video, ffmpeg_available
        if ffmpeg_available() is None:
            statuses["g2_video_fetch"] = "skipped (no ffmpeg)"
        else:
            fetch_video(limit_districts=3 if args.smoke else None,
                        resume=not args.force)
            statuses["g2_video_fetch"] = "ok"
        from features.encode_video_xclip import main as encode_video
        encode_video()
        statuses["g2_video_encode"] = "ok"
    except ImportError:
        statuses["g2_video"] = "skipped (no yt-dlp)"
        print("[VIDEO] Chưa cài yt-dlp -> bỏ qua.")
    except Exception as exc:
        statuses["g2_video"] = f"failed: {exc}"
        traceback.print_exc()


# ------------------------------------------------------------------
# Bước 3 — G5: cold-edge augmentation
# ------------------------------------------------------------------
def step_cold_aug(cfg_base, statuses, force=False):
    print("\n=== BƯỚC 3 (G5): COLD-EDGE AUGMENTATION ===")
    try:
        from graph.augment_cold_edges import main as augment
        ok = augment(cfg=cfg_base, force=force)
        statuses["g5_cold_aug"] = "ok" if ok else "skipped (no modalities)"
    except Exception as exc:
        statuses["g5_cold_aug"] = f"failed: {exc}"
        traceback.print_exc()


# ------------------------------------------------------------------
# Bước 4 — training các cấu hình ablation
# ------------------------------------------------------------------
def load_graph_data():
    meta = json.loads((GRAPH_DIR / "graph_meta.json").read_text(encoding="utf-8"))
    num_users, num_items = int(meta["num_users"]), int(meta["num_items"])

    train_df = pd.read_parquet(GRAPH_DIR / "train_interactions.parquet")
    val_df = pd.read_parquet(GRAPH_DIR / "val_interactions.parquet")
    test_df = pd.read_parquet(GRAPH_DIR / "test_interactions.parquet")

    norm_adj = torch.load(NORM_ADJ_PATH, weights_only=False)
    if norm_adj.dtype != torch.float32:
        norm_adj = norm_adj.to(torch.float32)

    train_csr = _edges_to_csr(train_df, num_users, num_items)
    val_csr = _edges_to_csr(val_df, num_users, num_items)
    excl_test_csr = (train_csr + val_csr).tocsr()

    return {
        "meta": meta, "num_users": num_users, "num_items": num_items,
        "train_df": train_df, "val_df": val_df, "test_df": test_df,
        "norm_adj": norm_adj,
        "excl_val_csr": train_csr,
        "excl_test_csr": excl_test_csr,
        "item_degrees": np.asarray(train_csr.sum(axis=0)).ravel().astype(np.int64),
    }


def _edges_to_csr(df, num_users, num_items):
    users = df["user_idx"].to_numpy(dtype=np.int64)
    items = df["item_idx"].to_numpy(dtype=np.int64)
    csr = sp.csr_matrix(
        (np.ones(len(users), dtype=np.float32), (users, items)),
        shape=(num_users, num_items), dtype=np.float32,
    )
    csr.data[:] = 1.0
    csr.eliminate_zeros()
    return csr


def norm_adj_for_cfg(data, cfg):
    """Chọn đồ thị lan truyền: augment khi cfg.cold_aug và file tồn tại."""
    if cfg.cold_aug and NORM_ADJ_AUG_PATH.exists():
        return torch.load(NORM_ADJ_AUG_PATH, weights_only=False).to(torch.float32)
    return data["norm_adj"]


def run_one_config(name, desc, overrides, base_cfg, data, args, statuses):
    out_dir = RESULTS_DIR / "runs" / name
    metrics_path = out_dir / "metrics.json"
    t0 = time.perf_counter()

    cfg = override_config(base_cfg, overrides)
    cfg.run_name = name
    if args.smoke:
        cfg.epochs = args.epochs or 2
        cfg.eval_every = 1
        cfg.patience_evals = 3
        cfg.batch_size = min(cfg.batch_size, 1024)
    else:
        if args.epochs:
            cfg.epochs = args.epochs
        if args.eval_every:
            cfg.eval_every = args.eval_every

    if metrics_path.exists() and args.skip_existing and not args.smoke and not args.force:
        print(f"\n=== {name}: đã có kết quả — bỏ qua (--force để chạy lại) ===")
        print(f"    ({desc})")
        statuses.setdefault("runs", {})[name] = "skipped (exists)"
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    print(f"\n=== TRAIN: {name} ===")
    print(f"    {desc}")

    feats = {}
    if cfg.multimodal:
        from model.mmgcf import load_modality_features
        feats = load_modality_features(cfg, data["num_items"])
        if not feats:
            print(f"    Không có modality nào -> trả về LightGCN thuần.")
            cfg.multimodal = False

    # Seed TRƯỚC khi khởi tạo model (Xavier init + frozen projection)
    from train.train import set_seed
    set_seed(cfg.seed)

    train_edges = data["train_df"][["user_idx", "item_idx"]].to_numpy()
    model = MMGCF(cfg, data["num_users"], data["num_items"],
                  norm_adj_for_cfg(data, cfg), feats,
                  item_degrees=data["item_degrees"],
                  train_edges=train_edges)

    result = train_model(cfg, model, data["train_df"][["user_idx", "item_idx"]].to_numpy(),
                         {"val_df": data["val_df"], "excl_csr": data["excl_val_csr"]})

    # Nạp model tốt nhất và đánh giá TEST (che train+val, đúng protocol tác giả)
    model.load_state_dict(result["best_state"])
    model.eval()
    with torch.no_grad():
        user_emb, final_item, _, content_view = model.get_embedding()
    extras = (
        model.cold_channel_extras(content_view)
        if getattr(model, "use_cold_channel", False) else None
    )
    test_metrics, detail = evaluate_ranking(
        user_emb, final_item, data["excl_test_csr"], data["test_df"],
        tuple(cfg.top_k), torch.device(cfg.device) if cfg.device != "auto"
        else resolve_device("auto"),
        return_detail=True, extras=extras,
    )
    stratified = stratified_metrics(detail, data["item_degrees"], ks=(10,))

    attention = None
    try:
        attention = model.attention_summary()
    except Exception:
        attention = None

    wall = time.perf_counter() - t0
    metrics = {
        "name": name,
        "desc": desc,
        "config": config_to_dict(cfg),
        "best_epoch": result["best_epoch"],
        "val": result["val_metrics"],
        "test": test_metrics,
        "stratified_test": stratified,
        "attention_weights": attention,
        "wall_time_s": round(wall, 1),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(metrics_path, metrics)
    pd.DataFrame(result["history"]).to_csv(out_dir / "history.csv", index=False)
    torch.save(result["best_state"], out_dir / "best_model.pt")

    from evaluation.evaluator import format_metrics_row
    print("[RESULT] " + format_metrics_row(name, test_metrics, stratified))
    if attention:
        print(f"         attention: " + ", ".join(f"{k}={v:.3f}" for k, v in attention.items()))

    statuses.setdefault("runs", {})[name] = "ok"
    return metrics


def summarize(all_metrics):
    """Bảng ablation tổng hợp."""
    print("\n=== BƯỚC 5 (G6): ABLATION MATRIX ===")
    rows = []
    for m in all_metrics:
        if m is None:
            continue
        row = {"config": m["name"]}
        for k in sorted(m["test"].keys(), key=int):
            row[f"HR@{k}"] = round(m["test"][k]["hr"], 4)
            row[f"NDCG@{k}"] = round(m["test"][k]["ndcg"], 4)
        strat = m.get("stratified_test", {})
        for bucket in ("cold(0)", "low(1-2)", "mid(3-10)", "head(>10)"):
            entry = strat.get(bucket, {})
            row[f"{bucket}_HR@10"] = (
                round(entry["hr@10"], 4) if entry.get("hr@10") is not None else ""
            )
            row[f"{bucket}_NDCG@10"] = (
                round(entry["ndcg@10"], 4) if entry.get("ndcg@10") is not None else ""
            )
        row["best_epoch"] = m.get("best_epoch")
        row["time_s"] = m.get("wall_time_s")
        rows.append(row)

    df = pd.DataFrame(rows)
    ABLATION_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ABLATION_CSV_PATH, index=False, encoding="utf-8-sig")

    if rows:
        cols = ["config", "HR@5", "HR@10", "HR@20", "NDCG@5", "NDCG@10", "NDCG@20"]
        widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
        print(" | ".join(c.ljust(widths[c]) for c in cols))
        print("-" * (sum(widths.values()) + 3 * (len(cols) - 1)))
        for r in rows:
            print(" | ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))
    print(f"\nĐã lưu bảng đầy đủ: {ABLATION_CSV_PATH}")


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Pipeline MMGCF cải tiến — chạy một lệnh.")
    parser.add_argument("--smoke", action="store_true", help="chạy nhanh để kiểm tra pipeline")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--limit-reviews", type=int, default=0)
    parser.add_argument("--configs", type=str, default=None,
                        help="chỉ chạy các config cách nhau bởi dấu phẩy")
    parser.add_argument("--force", action="store_true", help="tính lại kể cả đã có kết quả")
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false",
                        help="không bỏ qua các bước có sẵn kết quả")
    parser.add_argument("--no-external", action="store_true", help="bỏ qua audio/video ngoài")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    statuses = {}
    t_start = time.perf_counter()

    versions, device = preflight()
    statuses["preflight"] = "ok"

    base_cfg = Config()
    base_cfg.device = str(device)
    if args.smoke:
        base_cfg.epochs = 2

    # Data prep (chỉ cẩn thiết lần đầu từ dữ liệu thô; tự bỏ qua bước đã có)
    step_data_prep(args, statuses)
    ensure_pipeline_inputs(smoke=args.smoke)

    # G1
    step_reviews(args, statuses)

    # G2
    step_external(args, statuses)

    # G5 (chuẩn bị cạnh bù trước khi train)
    step_cold_aug(base_cfg, statuses, force=args.force)

    # G0 + G3 + G4 + G6 — huấn luyện chuỗi ablation
    data = load_graph_data()
    selected = ABLATION_DEFS
    if args.configs:
        wanted = [c.strip() for c in args.configs.split(",")]
        selected = [d for d in ABLATION_DEFS if d["name"] in wanted]
        unknown = set(wanted) - {d["name"] for d in selected}
        if unknown:
            raise SystemExit(f"Config không tồn tại: {unknown}")

    audio_ok = AUDIO_FEATURES_PATH.exists()
    video_ok = VIDEO_FEATURES_PATH.exists()
    all_metrics = []
    for definition in selected:
        needs = definition.get("needs", ())
        if "audio" in needs and not audio_ok:
            print(f"\n=== {definition['name']}: BỎ QUA (thiếu audio_features.npy) ===")
            statuses.setdefault("runs", {})[definition["name"]] = "skipped (no audio)"
            all_metrics.append(None)
            continue
        if "video" in needs and not video_ok:
            print(f"\n=== {definition['name']}: BỎ QUA (thiếu video_features.npy) ===")
            statuses.setdefault("runs", {})[definition["name"]] = "skipped (no video)"
            all_metrics.append(None)
            continue
        try:
            metrics = run_one_config(
                definition["name"], definition["desc"], definition["overrides"],
                base_cfg, data, args, statuses,
            )
            all_metrics.append(metrics)
        except Exception:
            statuses.setdefault("runs", {})[definition["name"]] = "failed"
            print(f"!!! Config {definition['name']} BỊ LỖI:")
            traceback.print_exc()
            all_metrics.append(None)

    # G6 — tổng hợp từ TẤT CẢ kết quả có trên đĩa (kể cả các lần chạy trước),
    # theo đúng thứ tự ABLATION_DEFS -> chạy riêng 1 config không làm mất bảng cũ.
    final_metrics = []
    for definition in ABLATION_DEFS:
        path = RESULTS_DIR / "runs" / definition["name"] / "metrics.json"
        if path.exists():
            final_metrics.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            final_metrics.append(None)
    summarize(final_metrics)

    statuses["total_time_s"] = round(time.perf_counter() - t_start, 1)
    save_json(MANIFEST_PATH, {
        "versions": versions,
        "statuses": statuses,
        "smoke": args.smoke,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    print(f"\n=== HOÀN TẤT trong {statuses['total_time_s']}s — manifest: {MANIFEST_PATH} ===")


if __name__ == "__main__":
    main()
