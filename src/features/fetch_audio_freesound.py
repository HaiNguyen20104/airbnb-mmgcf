# -*- coding: utf-8 -*-
"""
fetch_audio_freesound.py
========================
G2: Tải audio ambience cấp KHU VỰC từ Freesound (API v2), lọc CC0.

Chiến lược:
  - Mỗi khu trong 50 khu vực Bangkok được tìm bằng alias tên thông dụng
    (VD: Vadhana -> "Sukhumvit", Huai Khwang -> "Ratchada").
  - Filter API: license "Creative Commons 0" (CC0 - dùng tự do) + duration 10-180s.
  - Ưu tiên kết quả rating cao; nếu khu không có kết quả -> dùng clip "generic
    Bangkok" (vẫn CC0) và đánh dấu status=generic; nếu vẫn không có -> missing.
  - Preview MP3 (công khai) -> ffmpeg chuyển wav 48kHz mono pcm_s16le để CLAP đọc.

Yêu cầu: token Freesound miễn phí qua biến môi trường FREESOUND_API_TOKEN
(đăng ký tại https://freesound.org/apiv2/apply) hoặc file configs/freesound_token.txt.

Đầu ra: data/external/bangkok/audio/{district}.wav + audio_fetch_status.json
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import json
import os
import subprocess
import time
from pathlib import Path

import requests

from configs.config import EXTERNAL_DIR, save_json
from features.district_utils import build_district_table

API_SEARCH = "https://freesound.org/apiv2/search/text/"
AUDIO_DIR = EXTERNAL_DIR / "audio"
STATUS_PATH = EXTERNAL_DIR / "audio_fetch_status.json"

TOKEN_FILE = Path(__file__).resolve().parents[1] / "configs" / "freesound_token.txt"

# Alias tên thông dụng (khu ít nổi tiếng dùng tên chính thức)
DISTRICT_ALIASES = {
    "Bang Bon": ["Bang Bon"],
    "Bang Kapi": ["Ramkhamhaeng Bangkok", "Bang Kapi"],
    "Bang Khae": ["Bang Khae"],
    "Bang Khen": ["Bang Khen"],
    "Bang Kho laen": ["Sathorn river Bangkok", "Bang Kho Laem"],
    "Bang Khun thain": ["Bang Khun Thian"],
    "Bang Na": ["Bangna Bangkok", "Bang Na"],
    "Bang Phlat": ["Bang Phlat"],
    "Bang Rak": ["Silom Bangkok", "Bang Rak"],
    "Bang Sue": ["Bang Sue"],
    "Bangkok Noi": ["Bangkok Noi"],
    "Bangkok Yai": ["Bangkok Yai"],
    "Bueng Kum": ["Ramkhamhaeng night market", "Bueng Kum"],
    "Chatu Chak": ["Chatuchak market Bangkok", "Jatujak"],
    "Chom Thong": ["Chom Thong Bangkok"],
    "Din Daeng": ["Ratchadaphisek Bangkok", "Din Daeng"],
    "Don Mueang": ["Don Mueang airport"],
    "Dusit": ["Dusit Bangkok", "Dusit palace"],
    "Huai Khwang": ["Ratchada night market Bangkok", "Huai Khwang"],
    "Khan Na Yao": ["Khan Na Yao"],
    "Khlong Sam Wa": ["Khlong Sam Wa"],
    "Khlong San": ["ICONSIAM Bangkok", "Khlong San"],
    "Khlong Toei": ["Khlong Toei market Bangkok", "Rama IV Bangkok"],
    "Lak Si": ["Lak Si"],
    "Lat Krabang": ["Lat Krabang"],
    "Lat Phrao": ["Lat Phrao"],
    "Min Buri": ["Min Buri"],
    "Nong Chok": ["Nong Chok"],
    "Nong Khaem": ["Nong Khaem"],
    "Parthum Wan": ["Siam Square Bangkok", "Pathum Wan"],
    "Phasi Charoen": ["Phasi Charoen"],
    "Phaya Thai": ["Pratunam Bangkok", "Phaya Thai"],
    "Phra Khanong": ["On Nut Bangkok", "Phra Khanong"],
    "Phra Nakhon": ["Khaosan Road Bangkok", "Grand Palace Bangkok"],
    "Pom Prap Sattru Phai": ["Yaowarat Bangkok", "Chinatown Bangkok"],
    "Pra Wet": ["Prawet Bangkok", "Seacon Square"],
    "Rat Burana": ["Rat Burana"],
    "Ratchathewi": ["Pratunam Bangkok", "Ratchathewi"],
    "Sai Mai": ["Sai Mai"],
    "Samphanthawong": ["Yaowarat night Bangkok", "Samphanthawong"],
    "Saphan Sung": ["Saphan Sung"],
    "Sathon": ["Sathorn Bangkok", "Silom night Bangkok"],
    "Suanluang": ["Suan Luang"],
    "Taling Chan": ["Taling Chan floating market", "Taling Chan"],
    "Thawi Watthana": ["Thawi Watthana"],
    "Thon buri": ["Thonburi Bangkok"],
    "Thung khru": ["Thung Khru"],
    "Vadhana": ["Sukhumvit Bangkok", "Thonglor Bangkok", "Asok Bangkok"],
    "Wang Thong Lang": ["Wang Thong Lang"],
    "Yan na wa": ["Yan Nawa"],
}

GENERIC_QUERIES = ["Bangkok street ambience", "Bangkok Thailand traffic", "Bangkok city"]
CC0_FILTER = 'license:"Creative Commons 0"'
DURATION_FILTER = "duration:[10 TO 180]"


def get_token():
    token = os.environ.get("FREESOUND_API_TOKEN", "").strip()
    if not token and TOKEN_FILE.exists():
        token = TOKEN_FILE.read_text(encoding="utf-8").strip()
    return token


def search_sound(token, query, page_size=10):
    params = {
        "query": query,
        "token": token,  # Freesound yêu cầu API key trên MỌI request search
        "filter": f"{CC0_FILTER} {DURATION_FILTER}",
        "sort": "rating_desc",
        "fields": "id,name,previews,duration,license,download",
        "page_size": page_size,
    }
    response = requests.get(API_SEARCH, params=params, timeout=30)
    response.raise_for_status()
    return response.json().get("results", [])


def ffmpeg_to_wav(src: Path, dst: Path, ffmpeg_exe):
    """MP3 preview -> WAV 48kHz mono 16-bit (chuẩn đầu vào của CLAP)."""
    cmd = [
        ffmpeg_exe, "-y", "-i", str(src),
        "-ar", "48000", "-ac", "1", "-acodec", "pcm_s16le",
        "-loglevel", "error", str(dst),
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=120)


def get_ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        path = "ffmpeg" if os.name != "nt" else "ffmpeg.exe"
        try:
            subprocess.run([path, "-version"], capture_output=True, timeout=30)
            return path
        except Exception:
            return None


def fetch_all(limit_districts=None, resume=True):
    token = get_token()
    if not token:
        print("[AUDIO-FETCH] KHÔNG có token Freesound (env FREESOUND_API_TOKEN "
              "hoặc configs/freesound_token.txt). -> Bỏ qua audio (modality mask).")
        return None

    ffmpeg = get_ffmpeg()
    if ffmpeg is None:
        print("[AUDIO-FETCH] Không tìm thấy ffmpeg (cài: pip install imageio-ffmpeg). "
              "-> Bỏ qua audio.")
        return None

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    table = build_district_table()
    districts = list(table["district_names"])
    if limit_districts:
        districts = districts[:limit_districts]

    # Clip generic dùng làm fallback (tìm 1 lần)
    generic = None
    for q in GENERIC_QUERIES:
        try:
            results = search_sound(token, q, page_size=5)
            if results:
                generic = results[0]
                break
        except Exception as exc:
            print(f"[AUDIO-FETCH] Lỗi tìm generic '{q}': {exc}")
        time.sleep(0.5)

    status = {}
    for district in districts:
        out_wav = AUDIO_DIR / f"{district}.wav"
        if resume and out_wav.exists():
            status[district] = {"status": "cached", "wav": str(out_wav.name)}
            continue

        chosen, chosen_status = None, "missing"
        for alias in DISTRICT_ALIASES.get(district, [district]):
            try:
                results = search_sound(token, f"{alias} Bangkok", page_size=5)
            except Exception as exc:
                print(f"[AUDIO-FETCH] Lỗi tìm '{district}' ({alias}): {exc}")
                time.sleep(1.0)
                continue
            if results:
                chosen = results[0]
                chosen_status = "specific"
                break
            time.sleep(0.3)
        if chosen is None and generic is not None:
            chosen, chosen_status = generic, "generic"

        if chosen is None:
            status[district] = {"status": "missing"}
            continue

        preview_url = (chosen.get("previews") or {}).get("preview-hq-mp3")
        if not preview_url:
            status[district] = {"status": "missing"}
            continue
        try:
            response = requests.get(preview_url, timeout=60)
            response.raise_for_status()
            tmp = out_wav.with_suffix(".tmp.mp3")
            tmp.write_bytes(response.content)
            ffmpeg_to_wav(tmp, out_wav, ffmpeg)
            tmp.unlink(missing_ok=True)
            status[district] = {
                "status": chosen_status,
                "sound_id": chosen.get("id"),
                "name": chosen.get("name"),
                "duration": chosen.get("duration"),
                "wav": out_wav.name,
            }
            print(f"[AUDIO-FETCH] {district:<22} {chosen_status:<8} "
                  f"({chosen.get('name', '')[:50]})")
        except Exception as exc:
            print(f"[AUDIO-FETCH] Lỗi tải {district}: {exc}")
            status[district] = {"status": "missing", "error": str(exc)}
        time.sleep(0.4)

    save_json(STATUS_PATH, status)
    n_ok = sum(1 for v in status.values() if v.get("status") in {"specific", "generic", "cached"})
    print(f"[AUDIO-FETCH] Xong: {n_ok}/{len(districts)} khu có audio "
          f"(specific/generic). Trạng thái: {STATUS_PATH}")
    return status


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-districts", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    fetch_all(limit_districts=args.limit_districts, resume=not args.no_resume)
