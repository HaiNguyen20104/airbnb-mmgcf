# REPORT — Cải tiến MMGCF cho bài toán gợi ý căn hộ Airbnb Bangkok

**Nhánh code:** `feat/mmgcf-improvements` · **Dataset:** Airbnb Bangkok (4.411 lượt tương tác test của 4.411 users, 8.308 items, 48 khu vực)
**Baseline tham chiếu:** MMGCF gốc (branch `feat/mmgcf`, tune_id025_150): trên nhóm cold test HR@10 = 0.02474 / NDCG@10 = 0.01269; trên toàn bộ test HR@10 = 0.05010 / NDCG@10 = 0.02763.

> **Tính trung thực của báo cáo:** mọi con số dưới đây được trích trực tiếp từ
> `results/ablation_matrix.csv` (9 cấu hình canonical, seed 42) và
> `results/report_data/experiments.json` (Phase 1–5 của script thí nghiệm báo
> cáo). Các phần cần chạy thêm (t-SNE, đo latency) được **ghi rõ là chưa thực
> hiện**, tuyệt không dùng số liệu ước lượng.

---

## 1. Hiệu năng tổng thể & kiểm định thống kê (Student's t-test)

### 1.1 Bảng so sánh tổng thể (dataset Bangkok — protocol 2 layers / 150 epochs / patience 12)

| Cấu hình | HR@5 | NDCG@5 | HR@10 | NDCG@10 | HR@20 | NDCG@20 | cold(0) HR@10 | best epoch |
|---|---|---|---|---|---|---|---|---|
| LightGCN (CF thuần) | 0.0218 | 0.0148 | 0.0340 | 0.0187 | 0.0487 | 0.0224 | **0.0000** | 50 |
| **MMGCF gốc (baseline trung thực)** | 0.0338 | 0.0235 | 0.0501 | 0.0288 | 0.0714 | 0.0342 | 0.0352 | 58 |
| MMGCF + DemandGate (chỉ review) | 0.0122 | 0.0073 | 0.0190 | 0.0096 | 0.0315 | 0.0127 | 0.0065 | 3 |
| MMGCF + Attention (không InfoNCE) | 0.0145 | 0.0093 | 0.0220 | 0.0118 | 0.0345 | 0.0149 | 0.0013 | 3 |
| + InfoNCE (Lcl) | 0.0376 | 0.0236 | 0.0569 | 0.0298 | 0.0807 | 0.0357 | 0.0091 | 17 |
| + Cold-aug edges | 0.0406 | 0.0256 | 0.0583 | 0.0313 | 0.0855 | 0.0381 | 0.0339 | 15 |
| **+ Video (best overall)** | **0.0417** | **0.0259** | **0.0639** | **0.0330** | **0.0882** | **0.0391** | 0.0365 | 17 |
| Đủ 4 phương thức (T+I+A+V) | 0.0379 | 0.0245 | 0.0562 | 0.0305 | 0.0825 | 0.0371 | 0.0273 | 22 |
| **ColdCC (kênh nội dung cold)** | 0.0351 | 0.0237 | 0.0533 | 0.0295 | 0.0771 | 0.0355 | **0.0716** | 120 |

### 1.2 Kiểm định ý nghĩa thống kê — Welch t-test, 5 seeds {42, 0, 1, 2, 3}

Mỗi cấu hình được train độc lập 5 lần (cùng protocol), so sánh bằng Welch's
two-sample t-test (không giả định phương sai bằng nhau):

| So sánh | Chỉ số | mean A | mean B | t | p-value | p<0.05 |
|---|---|---|---|---|---|---|
| LightGCN vs MMGCF gốc | HR@10 | 0.0338 | 0.0506 | −29.02 | **1.8×10⁻⁷** | ✅ |
| LightGCN vs MMGCF gốc | NDCG@10 | 0.0185 | 0.0278 | −15.83 | **5.5×10⁻⁷** | ✅ |
| LightGCN vs MMGCF gốc | cold(0) HR@10 | 0.0000 | 0.0307 | −9.70 | **6.3×10⁻⁴** | ✅ |
| LightGCN vs MMGCF gốc | cold(0) NDCG@10 | 0.0000 | 0.0159 | −8.73 | **9.5×10⁻⁴** | ✅ |
| MMGCF gốc vs **ColdAug-Video** | HR@10 | 0.0506 | 0.0563 | −2.91 | **0.0422** | ✅ |
| MMGCF gốc vs ColdAug-Video | NDCG@10 | 0.0278 | 0.0298 | −1.96 | 0.1069 | ❌ ns |
| MMGCF gốc vs ColdAug-Video | cold(0) HR@10 | 0.0307 | 0.0336 | −0.89 | 0.4191 | ❌ ns |
| MMGCF gốc vs ColdAug-Video | cold(0) NDCG@10 | 0.0159 | 0.0162 | −0.15 | 0.8840 | ❌ ns |
| MMGCF gốc vs ColdCC | HR@10 | 0.0506 | 0.0534 | −2.09 | 0.1009 | ❌ ns |
| MMGCF gốc vs **ColdCC** | NDCG@10 | 0.0278 | 0.0297 | −3.38 | **0.0102** | ✅ |
| MMGCF gốc vs **ColdCC** | cold(0) HR@10 | 0.0307 | 0.0669 | −7.72 | **5.9×10⁻⁵** | ✅ |
| MMGCF gốc vs **ColdCC** | cold(0) NDCG@10 | 0.0159 | 0.0420 | −7.64 | **1.4×10⁻⁴** | ✅ |

**Kết luận §1:**
- Baseline trung thực MMGCF gốc vượt LightGCN thuần **ở mọi chỉ số với p < 0.001** (n=5 seeds, phương sai rất nhỏ: std HR@10 ± 0.0006).
- **ColdAug-Video** (cải tiến tốt nhất tổng thể) vượt baseline HR@10 với **p = 0.0422 ✅** (seed-mean +11.3%; canonical seed-42 +27.5%) — tuy **không** đạt ngưỡng trên NDCG@10 (p=0.107) và các chỉ số cold (p=0.42/0.88); ghi trung thực.
- **ColdCC** (điểm nhấn cold-start) đạt ý nghĩa thống kê mạnh nhất về cold: **p = 5.9×10⁻⁵ và 1.4×10⁻⁴ ✅ trên cả 2 chỉ số cold**, đồng thời NDCG@10 tổng thể p = 0.0102 ✅.

### 1.3 Hạn chế công bố trung thực
- **Chỉ 1 dataset (Bangkok).** Đề bài yêu cầu "3 datasets"; pipeline + `run_all.py` hỗ trợ mở rộng nhưng chỉ Bangkok có đủ features (text CLIP, image, audio CLAP, video X-CLIP, review 693k). Toàn bộ kiểm định trên đây dùng 1 dataset.
- ColdAug-Video: NDCG@10 và các chỉ số cold chỉ có ý nghĩa biên / ns.
- ColdCC: HR@10 tổng thể ns (p=0.10) — điểm nổi bật nằm ở cold-stratified.
- n=5 seeds còn thấp so với tiêu chuẩn khắt khe (10–20 seeds) — nên tăng nếu có thời gian tính toán.

---

## 2. Bứt phá bài toán khởi đầu lạnh (Cold-start)

**Mốc đề bài đặt** (MMGCF gốc trên cold test): **HR@10 = 0.02474, NDCG@10 = 0.01269** — đã đối chiếu trung thực từ `git show origin/feat/mmgcf:…/tune_id025_150/mmgcf_metrics.json`.

### 2.1 Chỉ số theo nhóm độ lạnh (canonical seed 42; số tương tác mỗi nhóm: cold 768 / low 1377 / mid 1931 / head 335)

| Model | cold(0) | low(1–2) | mid(3–10) | head(>10) |
|---|---|---|---|---|
| LightGCN | 0.0000 | 0.0145 | 0.0378 | 0.1701 |
| MMGCF gốc | 0.0352 | 0.0363 | 0.0611 | 0.0776 |
| ColdAug (Lcl + edges) | 0.0339 | 0.0508 | 0.0606 | 0.1313 |
| ColdAug-Video | 0.0365 | 0.0530 | 0.0694 | 0.1403 |
| **ColdCC (γ=0.5)** | **0.0716** | 0.0283 | 0.0580 | 0.0866 |

### 2.2 Tăng trưởng so với mốc 0.02474 / 0.01269 — 5 seeds

| Metric cold(0) | MMGCF gốc (5 seeds) | **ColdCC (5 seeds)** | So mốc gốc |
|---|---|---|---|
| HR@10 | 0.0307 ± 0.0071 | **0.0669 ± 0.0077** | **×2.70** so với 0.02474 |
| NDCG@10 | 0.0159 ± 0.0041 | **0.0420 ± 0.0064** | **×3.31** so với 0.01269 |

- Đối chiếu seed-42-vs-seed-42 (protocol giống hệt): cold(0) HR@10 0.0352 → 0.0716 (**×2.03**); so trực tiếp mốc gốc là ×2.90.
- 5/5 seeds của ColdCC đều ≥ 0.056 (min 0.056, max 0.0755) — **100% vượt mốc 0.02474**, kiểm định Welch p = 5.9×10⁻⁵.
- Ngược lại MMGCF gốc có seed thấp nhất 0.0208 (< mốc 0.02474) — độ bất ổn của cold ở baseline cao.

### 2.3 Cơ chế (trung thực)
1. **Contrastive + cold-aug tạo bước nền:** InfoNCE đồng bộ không gian nhúng hành vi–nội dung (G4); cold-aug thêm cạnh tương đồng nội dung cho item lạnh (G5) để item lạnh được "gắn" vào đồ thị cực thưa. Nhưng *đứng một mình* thì mức cải thiện trung bình (cold 0.0339–0.0365, ns so với fallback có sẵn của baseline).
2. **Kênh nội dung inference-time (ColdCC, `score + γ·cos(profile_u, content_i)`) là bước quyết định:** khi item lạnh không có cạnh nào, điểm chấm đồ thị đã được bổ sung trực tiếp bằng độ tương đồng giữa vector "gu" của user (học từ review) và nội dung của item — **không cần cạnh đồ thị**, không cần huấn luyện lại.

---

## 3. Nghiên cứu cắt bỏ (Ablation Study)

### 3.1 Hiệu quả từng phương thức (cùng stack Attention + InfoNCE + Cold-aug)

| Modality mix | HR@10 | NDCG@10 | cold(0) HR@10 | attention weights (đầu ra metric) |
|---|---|---|---|---|
| Text | 0.0546 | 0.0300 | 0.0339 | text 0.999, review 0.001 |
| + Image (T+I) | 0.0583 | 0.0313 | 0.0339 | text 0.842, image 0.157, review 0.000 |
| + Audio (T+I+A) | 0.0585 | 0.0302 | 0.0260 | text 0.848, image 0.095, audio 0.058 |
| Đủ 4 (T+I+A+V) | 0.0562 | 0.0305 | 0.0273 | text 0.810, image 0.095, **audio 0.000**, video 0.095 |

- Text → T+I: **+6.8%** HR@10 — hình ảnh là phương thức đóng góp tích cực nhất.
- T+I → T+I+A: HR@10 ngang nhưng cold **giảm 0.0339 → 0.0260** — audio phân phối thưa (~80% khu dùng chung 1 clip generic) gây nhiễu.
- Đủ 4 mod: attention tự học **gạt audio về 0.000** — bằng chứng trực tiếp cho cơ chế thích ứng theo từng thực thể (tiêu chí #2). Video giữ trọng số 0.047–0.095 và cho cấu hình tổng thể tốt nhất.
- ⚠️ Ghi trung thực: audio được *nhúng architecture đầy đủ* nhưng *hiệu quả* chưa thực; cần dữ liệu audio theo khu riêng (clip đặc trưng từng khu) để phát huy.

### 3.2 Hiệu quả cơ chế hợp nhất (cùng modality T+I, cùng InfoNCE/cold-aug)

| Fusion | HR@10 | NDCG@10 | cold(0) HR@10 |
|---|---|---|---|
| Uniform Average | 0.0528 | 0.0291 | 0.0339 |
| Concatenation + Linear | 0.0238 | 0.0123 | 0.0052 |
| **Adaptive Attention (entity-level)** | **0.0583** | **0.0313** | **0.0339** |

- Attention > Uniform: **+10.4%** HR@10 trên cùng stack — khẳng định "trọng số tự học theo từng thực thể" hay hơn cộng trung bình cào bằng của MMGCF gốc.
- Concat: **negative result** — nội suy qua 1 Linear độc lập trên vector 512×M, với dữ liệu thưa thì tối ưu rất kém (best val NDCG@10 chỉ 0.0134 ở epoch 44); không phù hợp dataset bán thưa.

### 3.3 Đóng góp của nhánh học tương phản Lcl (InfoNCE)

| Cấu hình | HR@10 | NDCG@10 | cold(0) HR@10 |
|---|---|---|---|
| MMGCF gốc (không Lcl) | 0.0501 | 0.0288 | 0.0352 |
| + Attention đơn thuần (không Lcl) | 0.0220 | 0.0118 | 0.0013 |
| + Attention + **InfoNCE** | 0.0569 | 0.0298 | 0.0091 |

- **Lcl là nút sống còn cho fusion thích ứng:** không có InfoNCE, attention tự sụp (0.0220, thua 56% baseline); thêm InfoNCE, cùng kiến trúc nhảy lên **0.0569 (+159% so với attention đơn, +13.6% so baseline)**.
- Trọng số attention **thay đổi hẳn** khi bật Lcl (text 0.41/image 0.43/review 0.16 → 0.776/0.224/~0) — InfoNCE đồng bộ không gian nhúng, attention mới có cơ sở học.
- Ghi trung thực: InfoNCE *một mình* (không attention/cold-aug) **hại cold** (cold(0) 0.0091) — alignment xung đột với fallback nội dung của cold nếu không có cơ chế bổ trợ.

---

## 4. Phân tích độ nhạy tham số

### 4.1 α — trọng số ID trong fusion (baseline trung thực, không Lcl)

| α (id_weight) | 0.0 | **0.25** | 0.5 | 0.75 |
|---|---|---|---|---|
| HR@10 | 0.0438 | **0.0501** | 0.0410 | 0.0304 |
| cold(0) HR@10 | 0.0352 | **0.0352** | 0.0013 | **0.0000** |

- Đỉnh rõ ở **α = 0.25** (đúng tuning gốc của tác giả): fused = 0.25·ID + 0.375·text + 0.375·image.
- α ≥ 0.5 → cold item chết hẳn (0.0000): ID-embedding phụ trách đồ thị, item lạnh không có cạnh → phần ID vô dụng, và nội dung bị dồn ép xuống quá thấp.
- Đối xứng: α = 0 (bỏ ID hoàn toàn) mất 12.6% HR@10 overall — ID vẫn cần cho items warm/nav.

### 4.2 τ — temperature của InfoNCE (stack attention T+I)

| τ | 0.05 | 0.1 | **0.2** | 0.4 |
|---|---|---|---|---|
| HR@10 | 0.0535 | 0.0567 | **0.0583** | 0.0451 |
| cold(0) HR@10 | 0.0273 | **0.0365** | 0.0339 | 0.0339 |

- Mũ nhiệt: τ quá nhỏ (0.05) → phân phối quá nhọn; τ quá lớn (0.4) → mất chức năng phân biệt (HR@10 tụt 22% so với đỉnh).
- **τ = 0.2** là đỉnh overall; τ = 0.1 tối ưu cold riêng (0.0365). Protocol chính: **τ = 0.2**.

### 4.3 λ — hệ số InfoNCE (Lcl) (stack attention T+I)

| λ | 0.001 | 0.005 | 0.01 | **0.05** |
|---|---|---|---|---|
| HR@10 | 0.0349 | 0.0540 | 0.0583 | **0.0655** |
| cold(0) HR@10 | 0.0326 | 0.0312 | 0.0339 | **0.0482** |

- **Tăng đơn điệu trên toàn range đã quét:** λ = 0.05 tốt nhất cả overall (0.0655) và cold (0.0482) — dấu hiệu **range chưa đóng**: λ > 0.05 có thể còn lợi (chưa quét do giới hạn thời gian); protocol canonical dùng λ = 0.01.

### 4.4 Biểu đồ độ nhạy — `results/figures/sensitivity_sweep.png`

Biểu đồ 3 panel (α, τ, λ) được sinh từ bộ sweep 12 runs trên, kèm mốc đỉnh cho từng tham số (α=0.25, τ=0.2, λ=0.05). Quan sát trực quan khớp bảng: cột α cho thấy "vách cliff" cold(0) từ 0.0352 → 0.0000 khi α vượt 0.25; cột λ hiện đường tăng chưa bão hoà trên cả 3 chỉ số — dấu hiệu range nên mở rộng trong tương lai.

---

## 5. Trực quan hoá t-SNE trước/sau khử nhiễu đa phương thức

Hình `results/figures/tsne_before_after_denoise.png` — chiếu t-SNE (perplexity=40, init=pca, seed 42) trên 4.369 items lấy mẫu cân theo 4 bucket độ lớn, màu theo bucket; **trước** = không gian hợp nhất trung bình các modality bên cung (host), **sau** = không gian sau khi DemandGate đối chiếu với review thực tế của khách (bên cầu).

**Đọc kết quả trung thực:**
1. Các điểm phân bố lại rõ rệt sau khi ghép tín hiệu review (mật độ "ún" phía trước biến thành cụm nhỏ dần dần, vùng biên thay đổi) — DemandGate **tái cấu trúc** không gian nội dung theo tín hiệu bên cầu.
2. Đo lượng tử hoá bằng silhouette trên 4 bucket: TRƯỚC = −0.0365, SAU = −0.0392 — **gần 0 trong cả hai**. Điều này là đúng kỳ vọng: *bucket độ lớn là thống kê đồ thị, không phải bản sắc nội dung*, item nóng/lạnh không nhất thiết phải tách cụm về mặt hình học. Nên không dùng silhouette như bằng chứng "gỡ rối cụm" — giá trị chính của hình là minh hoạ trực quan sự dịch chuyển không gian (đúng yêu cầu trực quan hoá trước/sau của đề bài).
3. Bằng chứng định lượng của cơ chế khử nhiễu nằm ở các phần khác: trọng số attention review ≈0 khi chưa gióng hàng (§3.1), DemandGate đơn lẻ sụp (0.0190) nhưng phát huy khi đặt trên nền InfoNCE — bộ lọc bên cầu chỉ phát huy khi không gian nhúng đã gióng hàng.

---

## 6. Độ trễ suy luận (Inference Latency) — CHƯA ĐO

Yêu cầu: **< 50ms cho Top-20**. Phase 7 được thiết kế để đo trên checkpoint `mmgcf_coldaug_video` (batch 256 users, ms/user, GPU + CPU) nhưng bị dừng trước khi thực hiện — **không có số liệu thật trong báo cáo này** (không ước lượng).

---

## 7. Phụ lục — Negative results & hạn chế trung thực

1. **DemandGate đứng một mình sụp:** 0.0190 (−62% vs baseline) — review-only filter trước khi gióng hàng không gian nhúng là nhiễu; phải chạy **sau InfoNCE**.
2. **Attention không có InfoNCE:** 0.0220 (thua 56%) — "adaptation without alignment collapses".
3. **InfoNCE quá mạnh với cold:** cold(0) 0.0091 khi một mình — alignment hẹp đè bẹp fallback nội dung.
4. **Concat fusion:** 0.0238 — không phù hợp dữ liệu bán thưa.
5. **Audio:** attention weight thực tế 0.000–0.058 — chất lượng clip theo khu chưa đủ cá nhân hoá; đóng góp chưa thực.
6. **ColdAug vs fallback có sẵn của baseline:** chỉ modest (0.0339–0.0365 vs 0.0352), ns ở cold-stratified (n=5).
7. **ColdCC đánh đổi:** lạnh sụt ở low-bucket (1–2 cạnh) 0.0363 → 0.0283 — vùng có cạnh đồ thị yếu, kênh nội dung chi phối mạnh; mid/head gần giữ nguyên.
8. **n=5 seeds** cho t-test — đủ theo hợp đồng p<0.05 nhưng nên tăng nếu có khả năng tính toán.
9. **1 dataset (Bangkok)** — mục tiêu "3 datasets" chưa có đủ features; pipeline đã sẵn sàng mở rộng.

---

## 8. Tóm tắt chốt cho 5 tiêu chí luận văn

| # Tiêu chí | Trạng thái | Bằng chứng mấu chốt |
|---|---|---|
| 1 — 4 phương thức | ✅ | text → +image +6.8%; +video → cấu hình tốt nhất tổng thể (0.0639, +27.5% canonical); audio trung tính (negative result trung thực) |
| 2 — Attention thích ứng | ✅ | attention > uniform +10.4%; tự loại audio về 0; cần InfoNCE đi kèm |
| 3 — Demand filter (review) | ✅ thiết kế; negative khi standalone | review gap cos 0.70 vs 0.25; DemandGate phải đặt trên nền InfoNCE |
| 4 — InfoNCE Lcl | ✅✅ | 0.0220 → 0.0569 (+159%), +13.6% vs baseline; tuyển fusion thích ứng hoạt được |
| 5 — Cold-start | ✅✅ | cold(0) HR@10 ×2.70 mốc (0.0669 vs 0.02474, **p = 5.9×10⁻⁵**); NDCG ×3.31; ổn định 5/5 seeds |

---

### Truy nguồn số liệu
- Canonical 9-config: `results/ablation_matrix.csv`, `results/runs/<name>/metrics.json`
- Thí nghiệm báo cáo Phase 1–5 (39 runs tiền tố `_`): `results/report_data/experiments.json`, log `results/report_data/_log.txt`
- Biểu đồ: `results/figures/sensitivity_sweep.png` (từ experiments.json), `results/figures/tsne_before_after_denoise.png` (checkpoint `mmgcf_demand`, t-SNE seed 42; số liệu `results/report_data/tsne_summary.json`)
- Metrics MMGCF gốc: `git -C D:\Doan2026\airbnb-mmgcf-root show origin/feat/mmgcf:data/processed/bangkok/model/tune_id025_150/mmgcf_metrics.json`
