# his.md — Nhật ký tiến độ dự án NMS training

> Ghi lại MÔI TRƯỜNG, mọi lệnh đã chạy, số liệu THẬT theo epoch, các điều chỉnh phương pháp,
> và **lệnh resume**. Mọi số lấy từ output thật (`outputs/logs/*`). Không bịa số.

## 1. Môi trường (đã kiểm chứng)
- GPU: **NVIDIA GeForce RTX 5080**, 16 GB, compute capability **(12, 0)** (sm_120/Blackwell), driver 595.71.05.
- `uv` 0.11.16; Python **3.12.13**.
- **torch 2.11.0+cu128**, **torchvision 0.26.0+cu128** (CUDA 12.8 — bắt buộc cho sm_120).
- `uv run python check_env.py` → `cuda available: True`, `cap: (12, 0)`, `cuda matmul OK`.

## 2. Các lệnh đã chạy (theo thứ tự)
```bash
cd computer_vision/nms_training
uv init --python 3.12 --no-workspace .          # rồi sửa pyproject: index pytorch-cu128
uv add torch torchvision                        # -> 2.11.0+cu128 / 0.26.0+cu128
uv add matplotlib numpy pillow
uv run python check_env.py                       # GATE GPU
uv run python download_data.py                   # Penn-Fudan: images 170 / masks 170
uv run python train.py --epochs 8                # lần chạy KIỂM THỬ pipeline (ok, AP@0.5≈0.99)
uv run python train.py --resume outputs/checkpoints/last.pth --epochs 9   # kiểm tra resume (ok)
# --- Điều chỉnh phương pháp (xem mục 5) rồi chạy CHÍNH THỨC: ---
uv run python train.py --epochs 20               # RUN CHÍNH THỨC (sau khi xoá outputs cũ)
uv run python study_nms.py                       # sinh 5 figures + nms_sweep.csv
```

## 3. Dataset
- Penn-Fudan Pedestrian: zip **53,723,336 bytes**, sha256 `9095a9613c95586f…`.
- **170 ảnh / 170 mask**. Split seed=1 → **train 120 / val 50**. 1 lớp `person`.

## 3b. Finetune từ đâu (điểm khởi đầu trọng số)
- Kiến trúc: Faster R-CNN ResNet-50 + FPN.
- Khởi đầu: `fasterrcnn_resnet50_fpn(weights="DEFAULT")` = `FasterRCNN_ResNet50_FPN_Weights.DEFAULT` (COCO_V1), pretrain trên **COCO train2017**; torchvision tải `fasterrcnn_resnet50_fpn_coco-258fb6c6.pth` từ download.pytorch.org/models (thấy trong log epoch 0 của run đầu).
- Giữ nguyên backbone/FPN/RPN/RoI (trọng số COCO); chỉ thay `box_predictor` → 2 lớp (nền + person) rồi train tiếp trên Penn-Fudan. Mã: `model.py`.

## 4. Tiến độ train — RUN CHÍNH THỨC 20 epoch (số THẬT, outputs/logs/training_log.jsonl)
| epoch | loss | AP@0.5 | mAP@[.5:.95] | epoch | loss | AP@0.5 | mAP@[.5:.95] |
|---|---|---|---|---|---|---|---|
| 0 | 0.4026 | 0.9892 | 0.6997 | 10 | 0.0496 | 0.9926 | 0.8190 |
| 1 | 0.1414 | 0.9885 | 0.7475 | 11 | 0.0474 | 0.9921 | 0.8102 |
| 2 | 0.1047 | 0.9895 | 0.7926 | 12 | 0.0458 | 0.9920 | 0.8266 |
| 3 | 0.0900 | 0.9891 | 0.7864 | 13 | 0.0456 | 0.9921 | 0.8241 |
| 4 | 0.0844 | 0.9882 | 0.7954 | 14 | 0.0452 | 0.9922 | 0.8268 |
| 5 | 0.0750 | 0.9926 | 0.7425 | 15 | 0.0443 | 0.9918 | 0.8279 |
| 6 | 0.0621 | 0.9929 | 0.8171 | 16 | 0.0461 | 0.9917 | 0.8273 |
| 7 | 0.0517 | 0.9928 | 0.8183 | 17 | 0.0455 | 0.9918 | 0.8287 |
| 8 | 0.0527 | 0.9925 | **0.8308** | 18 | 0.0444 | 0.9918 | 0.8282 |
| 9 | 0.0513 | 0.9925 | 0.8190 | 19 | 0.0448 | 0.9918 | 0.8272 |

- **best.pth = epoch 8** (mAP@[.5:.95] = **0.8308**, AP@0.5 = 0.9925). Tổng ~2 phút trên 5080 (~5.5 s/epoch).

## 5. Hai điều chỉnh phương pháp (lý do — quan trọng cho báo cáo)
1. **Lịch LR scale theo số epoch:** `StepLR(step_size = epochs // 3)` thay cho cố định 3 → khi train 20 epoch, LR giảm trải đều (0.005 → 5e-4 ở epoch ~7 → 5e-5 ở ~14), tránh "đóng băng" sớm.
2. **Chọn best theo mAP@[.5:.95], KHÔNG theo AP@0.5:** vì AP@0.5 bão hoà ~0.99 (gần như phẳng) nên không phân biệt được epoch tốt; mAP@[.5:.95] mới phản ánh chất lượng ĐỊNH VỊ. (Ở run cũ chọn theo AP@0.5 từng chọn nhầm epoch 3 có mAP thấp.)

## 6. Kết quả NMS (study_nms.py — số THẬT)
- `NMS self-check OK` (greedy_nms tự viết trùng `torchvision.ops.nms` ở thr 0.3/0.5/0.7).
- Ảnh val đông nhất: idx=92 (7 người). fig02: **raw 140 box → sau NMS@0.5 còn 8 box**.
- Sweep (outputs/logs/nms_sweep.csv): thr 0.3/0.4/0.5/0.6/0.7/0.9 → avg_boxes 2.86/2.92/3.28/3.92/5.18/**21.68**; AP@0.5 0.9936→0.9469; recall=1.0 mọi thr.

## 7. LỆNH RESUME (train thêm — KHÔNG làm lại từ đầu)
```bash
cd computer_vision/nms_training
uv run python train.py --resume outputs/checkpoints/last.pth --epochs 24
```

## 8. Checklist — tất cả DONE
- [x] uv + cu128, GPU gate, dataset 170/170
- [x] code: dataset/model/metrics/nms/train/study
- [x] kiểm thử pipeline (8 epoch) + kiểm tra resume
- [x] điều chỉnh phương pháp (LR scale, best theo mAP)
- [x] RUN CHÍNH THỨC 20 epoch (best mAP 0.8308 @ epoch 8)
- [x] 5 figures + nms_sweep.csv
- [x] report_NMS_thuc_hanh.md (chuẩn học thuật + dễ hiểu + thuyết trình)
- [x] README + .gitignore + link computer_vision/README.md

## 9. Lỗi gặp & cách đã chủ động tránh
- Không có lỗi chặn. Đã tránh: `torch.amp.autocast('cuda')`/`GradScaler('cuda')` (API mới, không dùng `torch.cuda.amp.*` deprecated); ép `ImageReadMode.RGB`; lọc box suy biến (w/h ≤ 0); guard NMS tensor rỗng; chạy NMS per-class.
