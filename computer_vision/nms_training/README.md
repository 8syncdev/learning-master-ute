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
  sliding_window.py             # cửa sổ trượt + kim tự tháp ảnh (thuần NumPy) — tiền đề NMS
  train.py                      # train thật + checkpoint/resume + log jsonl
  study_nms.py                  # sinh figures + bảng sweep (chạy sau train)
  nms_deep_dive.py              # ảnh test thật: trace NMS từng bước, gallery, score_threshold
  report_NMS_thuc_hanh.md       # BÁO CÁO chi tiết (links trích dẫn ở đầu, ảnh train+test, lỗi cần tránh)
  build_sliding_window_nb.py    # sinh notebook chuyên đề sliding window
  Sliding_Window_to_NMS.ipynb   # NOTEBOOK sliding window (đã chạy, ảnh nhúng sẵn) — đọc TRƯỚC report
  his.md                        # nhật ký tiến độ + lệnh resume + finetune từ đâu
  data/                         # dataset (gitignore)
  outputs/
    checkpoints/{last,best}.pth # gitignore (lớn, tái tạo được)
    figures/01..08_*.png        # 8 hình báo cáo NMS + sw_01..07_*.png (notebook sliding window)
    logs/{training_log.jsonl,history.json,nms_sweep.csv,nms_trace.txt}  # số liệu thật (commit)
```

## Chạy (tuần tự)
```bash
cd computer_vision/nms_training
uv run python check_env.py                 # phải in cap: (12, 0) + cuda matmul OK
uv run python download_data.py             # images: 170 masks: 170
uv run python train.py --epochs 20         # ~2 phút trên RTX 5080
uv run python study_nms.py                 # hình 01–05 + nms_sweep.csv
uv run python nms_deep_dive.py             # hình 06–08 + nms_trace.txt
```

## Chuyên đề Sliding Window (notebook — đọc TRƯỚC NMS)
`Sliding_Window_to_NMS.ipynb` giải thích vì sao detector sinh ra hộp chồng lấp: cửa sổ trượt + kim tự tháp ảnh + bộ phân loại HOG/SVM (Dalal–Triggs) trên ảnh Penn-Fudan, rồi đưa thẳng các hộp thô vào `greedy_nms` của `nms.py`. Notebook **đã chạy sẵn, ảnh kết quả nhúng trong tệp** nên mở ra là xem được ngay (không cần chạy lại).
```bash
# Chỉ cần xem: mở Sliding_Window_to_NMS.ipynb (ảnh đã có sẵn).
# Chạy lại / chỉnh sửa (cần OpenCV + Jupyter, không đụng tới môi trường gốc):
uv run --with opencv-python-headless --with jupyterlab jupyter lab
# hoặc sinh lại notebook rồi thực thi để nhúng ảnh:
uv run --with opencv-python-headless --with nbconvert --with ipykernel \
  bash -c "python build_sliding_window_nb.py && \
           jupyter nbconvert --to notebook --execute --inplace Sliding_Window_to_NMS.ipynb"
```

## Train thêm epoch (resume — không làm lại từ đầu)
```bash
uv run python train.py --resume outputs/checkpoints/last.pth --epochs 24
```
Tham số chính của `train.py`: `--epochs --batch-size --lr --val-size --seed --resume`.

## Kết quả (lần chạy thật — chi tiết ở report)
- AP@0.5 ≈ **0.99**, mAP@[.5:.95] ≈ **0.83** (best @ epoch 8, run 20 epoch; torch 2.11.0+cu128, RTX 5080).
- NMS gom **140 → 8 hộp** trên ảnh đông; sweep ngưỡng IoU 0.3→0.9: avg_boxes 2.86→21.68, AP@0.5 0.9936→0.9469.
