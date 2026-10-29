# -*- coding: utf-8 -*-
"""
encode_audio_clap.py
====================
G2: Encode audio ambience khu vực -> embedding CLAP (512-d) -> truyền về listing.

Mô hình: laion/clap-htsat-unfused (có sẵn trong transformers, 512-d).
Audio: wav 48kHz mono (từ fetch_audio_freesound.py), cắt/gia hạn đúng 10s.

Đầu ra:
  - features/audio_district_features.npy : (K, 512) L2-norm
  - features/audio_features.npy          : (n_items, 512) L2-norm (khu hoặc trộn k=3 khu gần)
  - features/audio_status.json           : số khu có embedding
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import ClapModel, ClapProcessor

from configs.config import (
    EXTERNAL_DIR, FEATURES_DIR, DEFAULT_CONFIG, save_json,
)
from features.district_utils import build_district_table, district_to_item_features
from features.encoder_utils import extract_feature_tensor

AUDIO_DIR = EXTERNAL_DIR / "audio"
STATUS_PATH = FEATURES_DIR / "audio_status.json"
DISTRICT_OUT = FEATURES_DIR / "audio_district_features.npy"
ITEM_OUT = FEATURES_DIR / "audio_features.npy"

MODEL_NAME = "laion/clap-htsat-unfused"
SAMPLE_RATE = 48_000
CLIP_SECONDS = 10
CLIP_SAMPLES = SAMPLE_RATE * CLIP_SECONDS


def read_wav(path: Path):
    """WAV 16-bit -> float32 mono [-1, 1]; cắt/gia hạn đúng 10s."""
    from scipy.io import wavfile
    rate, data = wavfile.read(path)
    if rate != SAMPLE_RATE:
        raise ValueError(f"{path}: sample rate {rate} != {SAMPLE_RATE}.")
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32) / 32768.0
    if len(data) >= CLIP_SAMPLES:
        start = (len(data) - CLIP_SAMPLES) // 2
        data = data[start:start + CLIP_SAMPLES]
    else:
        pad = np.zeros(CLIP_SAMPLES - len(data), dtype=np.float32)
        data = np.concatenate([data, pad])
    return data


def main():
    print("=== G2: ENCODE AUDIO (CLAP) ===")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Device: {device}")

    if not AUDIO_DIR.exists() or not any(AUDIO_DIR.glob("*.wav")):
        print("[AUDIO] Không có wav nào -> tạo audio_features = None (missing modality).")
        save_json(STATUS_PATH, {"available": False, "reason": "no wav files"})
        return

    table = build_district_table()
    names = table["district_names"]
    name_to_id = {n: k for k, n in enumerate(names)}
    K = len(names)

    processor = ClapProcessor.from_pretrained(MODEL_NAME)
    model = ClapModel.from_pretrained(MODEL_NAME).to(device).eval()
    use_fp16 = device.type == "cuda"

    district_emb = np.full((K, 512), np.nan, dtype=np.float32)
    wavs = sorted(AUDIO_DIR.glob("*.wav"))
    print(f"Số file wav: {len(wavs)}")

    with torch.inference_mode():
        for wav in tqdm(wavs, desc="CLAP encode"):
            district = wav.stem
            if district not in name_to_id:
                continue
            try:
                audio = read_wav(wav)
            except Exception as exc:
                print(f"  Lỗi đọc {wav.name}: {exc}")
                continue
            inputs = processor(
                audio=[audio], sampling_rate=SAMPLE_RATE, return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            if use_fp16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = model.get_audio_features(**inputs)
            else:
                out = model.get_audio_features(**inputs)
            emb_t = extract_feature_tensor(out)
            emb = emb_t.float().cpu().numpy()[0]
            norm = np.linalg.norm(emb)
            district_emb[name_to_id[district]] = emb / norm if norm > 0 else emb

    n_ok = int(np.isfinite(district_emb).all(axis=1).sum())
    print(f"[AUDIO] Khu có embedding: {n_ok}/{K}")

    if n_ok == 0:
        save_json(STATUS_PATH, {"available": False, "reason": "no valid district embedding"})
        return

    k_nearest = 1 if not np.isfinite(table["centroids"]).all() else DEFAULT_CONFIG.audio_k_nearest
    item_emb = district_to_item_features(district_emb, table, k_nearest=k_nearest)
    covered = int((np.abs(item_emb).sum(axis=1) > 0).sum())
    print(f"[AUDIO] Listing được phủ: {covered:,}/{item_emb.shape[0]:,} (k_nearest={k_nearest})")

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    np.save(DISTRICT_OUT, district_emb)
    np.save(ITEM_OUT, item_emb)
    save_json(STATUS_PATH, {"available": True, "districts_ok": n_ok, "items_covered": covered})
    print(f"SUCCESS: audio_features -> {ITEM_OUT}")


if __name__ == "__main__":
    main()
