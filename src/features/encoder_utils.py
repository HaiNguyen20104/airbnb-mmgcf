# -*- coding: utf-8 -*-
"""
encoder_utils.py
================
Tiện ích chung cho các script encode feature (CLIP/CLAP/X-CLIP).

transformers 4.x:  model.get_text_features(**inputs)      -> Tensor
transformers 5.x:  ... -> BaseModelOutputWithPooling, trong đó embedding
                   ĐÃ CHIẾU (projection) được ghi lại vào `.pooler_output`
                   (đã kiểm chứng trong source transformers 5.21 cho cả
                   CLIPModel.get_text_features, ClapModel.get_audio_features,
                   XCLIPModel.get_video_features).

extract_feature_tensor() trả về Tensor embedding đúng để mọi script dùng chung.
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch


def extract_feature_tensor(output):
    """
    Rút Tensor embedding từ kết quả của get_{text,audio,video}_features
    bất kể phiên bản transformers.

    Thứ tự ưu tiên:
      1) output đã là Tensor (transformers 4.x),
      2) .text_embeds / .audio_embeds / .video_embeds (một số phiên bản),
      3) .pooler_output (transformers 5.x — embedding đã chiếu).
    """
    if isinstance(output, torch.Tensor):
        return output

    for attr in ("text_embeds", "audio_embeds", "video_embeds", "pooler_output"):
        value = getattr(output, attr, None)
        if value is not None and isinstance(value, torch.Tensor):
            return value

    raise TypeError(
        f"Không rút được embedding từ đối tượng {type(output).__name__} "
        f"(các trường có: {list(getattr(output, '__dict__', {}).keys()) or dir(output)})"
    )
