# -*- coding: utf-8 -*-
"""
fetch_video_districts.py
========================
G2: Tải video walking-tour cấp KHU VỰC từ YouTube (qua yt-dlp).

Chiến lược:
  - Mỗi khu tìm 1 video "Bangkok {alias} walking tour", lọc dài 3-25 phút.
  - Tải bản chất lượng thấp (height<=360) — chỉ cần sample 8 khung hình
    cho X-CLIP, không lưu trữ/phát lại (mục đích nghiên cứu).

Yêu cầu: pip install yt-dlp. ffmpeg chỉ cần khi trích khung hình (encode_video_xclip.py).

Đầu ra: data/external/bangkok/video/{district}.mp4 + video_fetch_status.json
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import json
import os
import shutil
import subprocess
from pathlib import Path

from configs.config import EXTERNAL_DIR, save_json
from features.district_utils import build_district_table
from features.fetch_audio_freesound import DISTRICT_ALIASES

VIDEO_DIR = EXTERNAL_DIR / "video"
STATUS_PATH = EXTERNAL_DIR / "video_fetch_status.json"

MIN_DURATION = 180      # 3 phút
MAX_DURATION = 1500     # 25 phút


def ffmpeg_available():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def pick_video(query):
    """Tìm video hợp lệ (duration 3-25 phút) — trả về video_id đầu tiên đạt."""
    import yt_dlp

    opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": False,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"ytsearch3:{query}", download=False)

    entries = info.get("entries") or []
    for entry in entries:
        duration = entry.get("duration")
        if duration and MIN_DURATION <= duration <= MAX_DURATION:
            return {"id": entry["id"], "title": entry.get("title", ""), "duration": duration}
    return None


def download_video(video_id, out_path: Path):
    import yt_dlp

    # YouTube hiện đại thường KHÔNG còn stream muxed (best/worst empty) ->
    # lấy stream video-only (X-CLIP chỉ cần khung hình, không cần audio),
    # ưu tiên mp4 <=360p; nội dung webm ghi đè tên .mp4 cũng đọc được bởi ffmpeg.
    fmt = "bv*[height<=360][ext=mp4]/bv*[height<=360]/bv*/wv"
    opts = {
        "format": fmt,
        "outtmpl": str(out_path),
        "quiet": True,
        "no_warnings": True,
        "retries": 2,
        "concurrent_fragment_downloads": 2,
    }
    url = f"https://www.youtube.com/watch?v={video_id}"
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    return out_path.exists() and out_path.stat().st_size > 0


def fetch_all(limit_districts=None, resume=True):
    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        print("[VIDEO-FETCH] Chưa cài yt-dlp (pip install yt-dlp). -> Bỏ qua video "
              "(modality mask).")
        return None

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    table = build_district_table()
    districts = list(table["district_names"])
    if limit_districts:
        districts = districts[:limit_districts]

    status = {}
    for district in districts:
        out_path = VIDEO_DIR / f"{district}.mp4"
        if resume and out_path.exists():
            status[district] = {"status": "cached", "file": out_path.name}
            continue

        aliases = DISTRICT_ALIASES.get(district, [district])
        picked = None
        for alias in aliases:
            try:
                picked = pick_video(f"Bangkok {alias} walking tour")
            except Exception as exc:
                print(f"[VIDEO-FETCH] Lỗi tìm '{district}' ({alias}): {exc}")
                continue
            if picked:
                break
        if picked is None:
            status[district] = {"status": "missing"}
            print(f"[VIDEO-FETCH] {district:<22} missing")
            continue
        try:
            ok = download_video(picked["id"], out_path)
            status[district] = {
                "status": "downloaded" if ok else "failed",
                "video_id": picked["id"],
                "title": picked.get("title", ""),
                "duration": picked.get("duration"),
                "file": out_path.name,
            }
            size_mb = out_path.stat().st_size / 1e6 if ok else 0
            print(f"[VIDEO-FETCH] {district:<22} {status[district]['status']:<10} "
                  f"{size_mb:.1f}MB | {picked.get('title', '')[:50]}")
        except Exception as exc:
            print(f"[VIDEO-FETCH] Lỗi tải {district}: {exc}")
            status[district] = {"status": "failed", "error": str(exc)}

    save_json(STATUS_PATH, status)
    n_ok = sum(1 for v in status.values() if v.get("status") in {"downloaded", "cached"})
    print(f"[VIDEO-FETCH] Xong: {n_ok}/{len(districts)} khu có video. Trạng thái: {STATUS_PATH}")
    return status


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-districts", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    fetch_all(limit_districts=args.limit_districts, resume=not args.no_resume)
