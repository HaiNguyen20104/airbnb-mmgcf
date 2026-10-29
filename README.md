# airbnb-mmgcf (cải tiến)

Triển khai **MMGCF** (Spillo et al., UMAP 2026) trên dữ liệu **Airbnb Bangkok**
(Inside Airbnb) với 5 hướng cải tiến:

1. **4 phương thức** — Text (CLIP) + Image (CLIP) + **Audio (CLAP)** + **Video (X-CLIP)**,
   audio/video cấp khu vực (50 khu `neighbourhood_cleansed`), có missing-modality mask.
2. **Fusion chú ý thích ứng theo từng thực thể** (thay cộng trung bình cào bằng) +
   gate ID↔MM theo chiều.
3. **Khử mâu thuẫn supply–demand** — profile review khách (time-decay, CLIP text
   cùng không gian) làm bộ lọc khử nhiễu nội dung "thổi phồng" của chủ nhà.
4. **Cross-view Contrastive (InfoNCE hai chiều)** gióng hàng góc nhìn đồ thị ↔
   góc nhìn đa phương thức.
5. **Cold-start** — cạnh item-item nội dung cho item lạnh (đã khử nhiễu review)
   + đánh giá phân tầng theo bậc item.

## Chuẩn bị môi trường (lần đầu — tự tạo venv riêng)

```bash
# 1) Tạo venv trong thư mục dự án
python -m venv .venv
.venv\Scripts\activate          # Windows (bash: source .venv/bin/activate)

# 2) Cài PyTorch CUDA theo GPU của bạn (H override khi khác)
pip install torch --index-url https://download.pytorch.org/whl/cu126

# 3) Cài phần còn lại
pip install -r requirements.txt

# 4) (tuỳ chọn) bật audio: token Freesound miễn phí https://freesound.org/apiv2/apply
setx FREESOUND_API_TOKEN "xxx"  # hoặc ghi token vào src/configs/freesound_token.txt
```

Môi trường này cần Python ≥ 3.10, CUDA-capable GPU khuyến nghị (không có GPU vẫn chạy,
chậm hơn nhiều: ~3 lần).

## Lệnh chạy duy nhất (đầu → cuối trong 1 lần chạy)

```bash
python src/run_all.py
```

Một lệnh duy nhất chạy **kể cả dữ liệu chưa chuẩn bị**:
raw Airbnb → lọc listings → k-core → tải ảnh (thư mục `data/processed/.../images/`) →
split train/val/test + đồ thị → CLIP encode text/image → encode 693k reviews (G1) →
Freesound audio + YouTube video (G2) → cold-aug (G5) → **8 cấu hình ablation** (G6) → bảng tổng hợp.

Các tuỳ chọn hữu ích:

```bash
python src/run_all.py --smoke                    # chạy nhanh kiểm tra (giảm quy mô)
python src/run_all.py --force                    # tính lại mọi bước kể cả đã có kết quả
python src/run_all.py --no-external              # bỏ qua tải audio/video
python src/run_all.py --epochs 200               # tăng số epoch
python src/run_all.py --configs lightgcn,mmgcf_4mod_full
```

Lưu ý:
- `--smoke` **bỏ qua data-prep** — chỉ dùng được sau một lần chạy đầy đủ.
- Video cần **yt-dlp + ffmpeg** (cả hai đã có trong `requirements.txt` qua `imageio-ffmpeg`).
  Bỏ audio/video cũng không sao: 2 config cần video/audio sẽ tự bị bỏ qua.

**Thời gian ước tính lần đầu** (GPU RTX 3-class, tốc độ mạng vừa):

| Giai đoạn | Thời gian |
|---|---|
| Data prep: lọc + k-core + tải ảnh ~8.3k listings | **30–90 phút** (bước nặng nhất, có resume — Ctrl+C rồi chạy lại là tiếp tục) |
| CLIP encode text + image | ~5 phút |
| Encode 693k reviews (G1) | ~9 phút (GPU) |
| Audio + video theo khu (G2) | ~20–30 phút (bỏ được bằng `--no-external`) |
| 8 cấu hình ablation × 100 epochs | ~15–25 phút |
| **Tổng** | **≈ 1.5–2.5 giờ** |

## Kết quả (thư mục `results/`)

| File | Nội dung |
|---|---|
| `ablation_matrix.csv` | Bảng so sánh 7 cấu hình: HR@{5,10,20}, NDCG@{5,10,20} + phân tầng cold |
| `review_gap_report.csv` | Bằng chứng định lượng hiện tượng "thổi phồng" (cos review vs host) |
| `cold_edges_report.csv` | Thống kê cạnh nội dung bù cho item lạnh |
| `runs/<config>/metrics.json` | Chi tiết 1 lần chạy (kèm trọng số attention để diễn giải) |
| `runs/<config>/history.csv` | Loss/val theo epoch |
| `run_manifest.json` | Trạng thái từng bước của lần chạy gần nhất |

## Cấu trúc code

```
src/
  run_all.py                  # LỆNH CHẠY DUY NHẤT
  configs/config.py           # cấu hình trung tâm
  model/                      # G0/G3/G4: lightgcn, fusion, contrastive, mmgcf
  train/train.py              # BPR + reg + InfoNCE
  evaluation/evaluator.py     # full-ranking HR/NDCG + stratified cold
  features/
    build_review_profile.py   # G1: encode CLIP + time-decay + gap report
    district_utils.py         # G2: mapping khu vực ↔ listing (haversine)
    fetch_audio_freesound.py  # G2: CC0 ambience 50 khu
    encode_audio_clap.py      # G2: CLAP 512-d
    fetch_video_districts.py  # G2: yt-dlp walking tour 50 khu
    encode_video_xclip.py     # G2: X-CLIP 512-d
  graph/augment_cold_edges.py # G5: cạnh item-item cho item lạnh
  data/, graph/               # pipeline dữ liệu sẵn có (trước cải tiến)
```

## Bảng ablation mặc định

| Config | Nội dung |
|---|---|
| `lightgcn` | LightGCN thuần (mốc CF) |
| `mmgcf_orig_2mod` | MMGCF gốc: text+image, fusion mean, weighting alpha, freeze |
| `mmgcf_demand` | + review modality & demand gate |
| `mmgcf_attn` | + fusion chú ý theo thực thể |
| `mmgcf_attn_infonce` | + InfoNCE |
| `mmgcf_coldaug` | + cạnh nội dung cho item lạnh |
| `mmgcf_coldaug_video` | + video khu vực (X-CLIP) — chỉ chạy khi có video features |
| `mmgcf_4mod_full` | 4 phương thức đầy đủ (cần cả audio + video features) |

## Kết quả lần chạy đầu (693,558 reviews, 100 epochs)

| Config | HR@5 | HR@10 | HR@20 | NDCG@10 | cold0 HR@10 | low1-2 HR@10 |
|---|---|---|---|---|---|---|
| lightgcn | 0.0234 | 0.0333 | 0.0503 | 0.0189 | 0.0000 | 0.0153 |
| mmgcf_orig_2mod | 0.0224 | 0.0338 | 0.0512 | 0.0187 | 0.0000 | 0.0196 |
| mmgcf_demand | 0.0131 | 0.0202 | 0.0317 | 0.0106 | 0.0091 | 0.0109 |
| mmgcf_attn | 0.0147 | 0.0236 | 0.0356 | 0.0125 | 0.0026 | 0.0109 |
| mmgcf_attn_infonce | 0.0340 | 0.0530 | 0.0771 | 0.0287 | 0.0117 | 0.0407 |
| **mmgcf_coldaug** | **0.0401** | **0.0583** | **0.0812** | **0.0317** | **0.0365** | **0.0501** |

Đọc kết quả:
- **InfoNCE (G4)** là nhảy vọt lớn nhất: HR@10 +60% so với mmgcf_attn — gióng hàng
  góc nhìn đồ thị ↔ đa phương thức giúp cả tổng thể lẫn cold.
- **Cold-start (G5)**: item lạnh bậc 0 — LightGCN/MMGCF gốc = 0.0, coldaug = **0.0365**
  (từ đồ thị item-item nội dung đã khử nhiễu review).
- **Demand gate (G3)** riêng lẻ làm giảm HR tổng (gây nhiễu cho item "hot") nhưng
  duy nhất nó cứu được cold0 = 0.0091; khi kết hợp attention + InfoNCE thì mọi bucket đều vượt baseline.
- Gap report: cos(review ↔ text host) median 0.70, review ↔ image 0.25 —
  review và nội dung chủ nhà nằm ở hai vùng không gian khác nhau (bằng chứng supply–demand gap).

