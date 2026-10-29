# Huấn luyện và đánh giá MMGCF trên Airbnb Bangkok

`train_eval.py` mặc định dùng biến thể late fusion trong `late_fusion_mmgcf.py`: LightGCN truyền embedding ID trên đồ thị train, sau đó kết hợp với embedding ảnh và văn bản của item. Thiết kế được tham khảo từ [MMGCF của giuspillo](https://github.com/giuspillo/MMGCF/blob/main/mmgcf/src/mmgcf.py) và được triển khai lại để nạp trực tiếp `ModelInputs` của dự án, không cần PyTorch Geometric.

Chạy từ thư mục gốc:

```powershell
python src/model/train_eval.py --model late_fusion --id-weight 0.25 --epochs 150 --patience 12
```

Mặc định hiện tương ứng với lệnh trên. Dùng `--output-dir` để lưu một thí nghiệm riêng. `--validation-only` chỉ huấn luyện và chọn checkpoint bằng validation; sau khi chọn cấu hình, đánh giá checkpoint bằng:

```powershell
python src/model/train_eval.py --evaluate-checkpoint data/processed/bangkok/model/tune_id025_150/mmgcf_best.pt
```

Mã che tương tác train khi đánh giá validation, che train và validation khi đánh giá test, đồng thời báo cáo HR@K/NDCG@K cho toàn bộ item, item cold start và item warm. Mẫu âm chỉ lấy từ các item có ít nhất một tương tác train. Checkpoint và báo cáo JSON được lưu trong thư mục kết quả. Có thể chọn lại mô hình attention trước đây bằng `--model attention`.

Cấu hình `id_weight=0.25` được chọn theo validation NDCG@10 trên tập Bangkok (so sánh với 0.1 và 0.5). Lần chạy với seed 42 chọn epoch 93 và cho test HR@10 = 0.05010, NDCG@10 = 0.02763; riêng 768 tương tác test với item cold start có HR@10 = 0.02474. Đây là kết quả của tập chia hiện tại, chưa phải bảo đảm cho dữ liệu khác.

## Bước 6: suy luận và gợi ý phòng

Sau khi có checkpoint, chọn một `reviewer_id` trong `user_mapping.parquet` hoặc dùng `user_idx`:

```powershell
python src/model/recommend.py --user-idx 0 --top-k 10
python src/model/recommend.py --user-id 100155876 --top-k 10 --output recommendations.csv
```

Lệnh mặc định dùng checkpoint `tune_id025_150/mmgcf_best.pt`; có thể chọn file khác qua `--checkpoint`. Kết quả có điểm xếp hạng, ID phòng, nhãn cold start, tên, giá, khu vực và URL. Các phòng đã có trong lịch sử train/validation/test của người dùng bị loại khỏi danh sách. Chỉ hỗ trợ người dùng đã có trong mapping; người dùng hoàn toàn mới cần một chiến lược cold-user riêng.
