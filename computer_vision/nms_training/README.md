# nms_training — Thực hành Standard NMS (train thật + nghiên cứu)

Dự án thực hành đi kèm tài liệu lý thuyết [`../non_maximum_suppression/`](../non_maximum_suppression/):
train một detector **thật** (Faster R-CNN trên Penn-Fudan) rồi nghiên cứu **greedy NMS** —
kỹ thuật gom các đối tượng nhận dạng chồng lấp. Quản lý môi trường bằng `uv`; chạy trên GPU (cu128).

## Cấu trúc
```
nms_training/
  pyproject.toml / uv.lock      # môi trường uv (torch+torchvision cu128)
  check_env.py                  # GATE: xác nhận GPU/CUDA chạy thật
  download_data.py              # tải & giải nén Penn-Fudan (stdlib)
  dataset.py                    # PennFudanDataset (box suy từ mask) + transforms
  model.py                      # build_model: finetune fasterrcnn_resnet50_fpn
  metrics.py                    # AP@0.5 / mAP@[.5:.95] tự viết (không pycocotools)
  nms.py                        # greedy_nms từ đầu (+ guard, assert format)
  train.py                      # train thật + checkpoint/resume + log jsonl
  study_nms.py                  # sinh figures + bảng sweep (chạy sau train)
  report_NMS_thuc_hanh.md       # BÁO CÁO chi tiết (ảnh train + kết quả + lỗi cần tránh)
  his.md                        # nhật ký tiến độ + lệnh resume
  data/                         # dataset (gitignore)
  outputs/
    checkpoints/{last,best}.pth # gitignore (lớn, tái tạo được)
    figures/*.png               # hình cho báo cáo (commit)
    logs/{training_log.jsonl,history.json,nms_sweep.csv}  # số liệu thật (commit)
```

## Chạy (tuần tự)
```bash
cd computer_vision/nms_training
uv run python check_env.py                 # phải in cap: (12, 0) + cuda matmul OK
uv run python download_data.py             # images: 170 masks: 170
uv run python train.py --epochs 8          # ~1 phút trên RTX 5080
uv run python study_nms.py                 # figures + outputs/logs/nms_sweep.csv
```

## Train thêm epoch (resume — không làm lại từ đầu)
```bash
uv run python train.py --resume outputs/checkpoints/last.pth --epochs 13
```
Tham số chính của `train.py`: `--epochs --batch-size --lr --val-size --seed --resume`.

## Kết quả (lần chạy thật — chi tiết ở report)
- AP@0.5 ≈ **0.99**, mAP@[.5:.95] ≈ **0.82** sau 8 epoch (torch 2.11.0+cu128, RTX 5080).
- NMS gom **158 → 10 hộp** trên ảnh đông; sweep ngưỡng IoU 0.3→0.9: avg_boxes 3.14→34.86, AP@0.5 0.9936→0.9531.
