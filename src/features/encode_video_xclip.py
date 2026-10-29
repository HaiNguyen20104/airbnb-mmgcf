# -*- coding: utf-8 -*-
"""
encode_video_xclip.py
=====================
G2: Encode video walking-tour khu vực -> embedding X-CLIP (512-d) -> về listing.

Mô hình: microsoft/xclip-base-patch32 (có sẵn trong transformers, 8 khung × 224²).
Trích 8 khung hình đều theo thời lượng video bằng ffmpeg (system hoặc imageio-ffmpeg).

Đầu ra:
  - features/video_district_features.npy : (K, 512) L2-norm
  - features/video_features.npy          : (n_items, 512) L2-norm
  - features/video_status.json
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pathlib import Path
import json
import subprocess

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import XCLIPModel, XCLIPProcessor

from configs.config import (
    EXTERNAL_DIR, FEATURES_DIR, DEFAULT_CONFIG, save_json,
)
from features.district_utils import build_district_table, district_to_item_features
from features.encoder_utils import extract_feature_tensor

VIDEO_DIR = EXTERNAL_DIR / "video"
STATUS_PATH = FEATURES_DIR / "video_status.json"
DISTRICT_OUT = FEATURES_DIR / "video_district_features.npy"
ITEM_OUT = FEATURES_DIR / "video_features.npy"

MODEL_NAME = "microsoft/xclip-base-patch32"
NUM_FRAMES = 8
FRAME_DIR = VIDEO_DIR / "_frames"


def get_ffmpeg():
    import shutil
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def get_duration(video: Path, ffmpeg):
    """Đọc Duration từ stderr của `ffmpeg -i` (imageio-ffmpeg không có ffprobe).

    LƯU Ý: KHÔNG được truyền -loglevel error — dòng "Duration:" thuộc mức
    INFO và sẽ bị che, khiến hàm luôn trả None (bug làm video encode bỏ qua
    toàn bộ 44 file).
    """
    result = subprocess.run(
        [ffmpeg, "-i", str(video), "-hide_banner"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=60,
    )
    text = result.stderr
    for line in text.splitlines():
        if "Duration:" in line:
            # "  Duration: 00:23:20.08, ..."
            time_part = line.split("Duration:")[1].split(",")[0].strip()
            hh, mm, ss = time_part.split(":")
            return int(hh) * 3600 + int(mm) * 60 + float(ss)
    return None


def extract_frames(video: Path, ffmpeg, duration):
    """Trích 8 khung tại các mốc 15%..85% thời lượng."""
    stamp_dir = FRAME_DIR / video.stem
    stamp_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for j in range(NUM_FRAMES):
        t = duration * (0.15 + 0.7 * j / (NUM_FRAMES - 1))
        out_path = stamp_dir / f"frame_{j:02d}.jpg"
        if not out_path.exists():
            cmd = [
                ffmpeg, "-y", "-ss", f"{t:.2f}", "-i", str(video),
                "-frames:v", "1", "-vf", "scale=224:224",
                "-loglevel", "error", str(out_path),
            ]
            subprocess.run(cmd, check=True, capture_output=True, timeout=60)
        frames.append(Image.open(out_path).convert("RGB"))
    return frames


def main():
    print("=== G2: ENCODE VIDEO (X-CLIP) ===")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Device: {device}")

    ffmpeg = get_ffmpeg()
    if ffmpeg is None:
        print("[VIDEO] Không có ffmpeg -> tạo video_features = None (missing modality).")
        save_json(STATUS_PATH, {"available": False, "reason": "no ffmpeg"})
        return

    videos = sorted(VIDEO_DIR.glob("*.mp4"))
    if not videos:
        print("[VIDEO] Không có mp4 nào -> video_features = None (missing modality).")
        save_json(STATUS_PATH, {"available": False, "reason": "no video files"})
        return

    table = build_district_table()
    name_to_id = {n: k for k, n in enumerate(table["district_names"])}
    K = len(table["district_names"])

    processor = XCLIPProcessor.from_pretrained(MODEL_NAME)
    model = XCLIPModel.from_pretrained(MODEL_NAME).to(device).eval()
    use_fp16 = device.type == "cuda"

    district_emb = np.full((K, 512), np.nan, dtype=np.float32)

    with torch.inference_mode():
        for video in tqdm(videos, desc="X-CLIP encode"):
            district = video.stem
            if district not in name_to_id:
                continue
            try:
                duration = get_duration(video, ffmpeg)
                if not duration or duration < 60:
                    print(f"  Bỏ qua {video.name}: duration={duration}")
                    continue
                frames = extract_frames(video, ffmpeg, duration)
                inputs = processor(videos=[frames], return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                if use_fp16:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        out = model.get_video_features(**inputs)
                else:
                    out = model.get_video_features(**inputs)
                emb_t = extract_feature_tensor(out)
                emb = emb_t.float().cpu().numpy()[0]
                norm = np.linalg.norm(emb)
                district_emb[name_to_id[district]] = emb / norm if norm > 0 else emb
            except Exception as exc:
                print(f"  Lỗi encode {video.name}: {exc}")

    n_ok = int(np.isfinite(district_emb).all(axis=1).sum())
    print(f"[VIDEO] Khu có embedding: {n_ok}/{K}")
    if n_ok == 0:
        save_json(STATUS_PATH, {"available": False, "reason": "no valid district embedding"})
        return

    k_nearest = 1 if not np.isfinite(table["centroids"]).all() else DEFAULT_CONFIG.audio_k_nearest
    item_emb = district_to_item_features(district_emb, table, k_nearest=k_nearest)
    covered = int((np.abs(item_emb).sum(axis=1) > 0).sum())
    print(f"[VIDEO] Listing được phủ: {covered:,}/{item_emb.shape[0]:,} (k_nearest={k_nearest})")

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    np.save(DISTRICT_OUT, district_emb)
    np.save(ITEM_OUT, item_emb)
    save_json(STATUS_PATH, {"available": True, "districts_ok": n_ok, "items_covered": covered})
    print(f"SUCCESS: video_features -> {ITEM_OUT}")


if __name__ == "__main__":
    main()
