# Changelog

Mọi thay đổi đáng kể ghi vào đây — format [Keep a Changelog](https://keepachangelog.com), versioning [SemVer](https://semver.org).
**8sync rule:** mỗi PR cập nhật mục `Unreleased` bên dưới.

## [Unreleased]

### Added (2026-07-21 — Final project ml_ad: FRF-MLP + báo cáo JTE)
- `ml_ad/final/`: đồ án cuối kỳ Học máy nâng cao — **FRF-MLP** (MLP kết hợp fuzzy logic Mamdani,
  2 mức fusion đặc trưng + quyết định) phân loại ngôn từ công kích trên ViHSD (3 lớp, split gốc).
  `fuzzy.py` (lexicon log-odds Dirichlet + 3 biến ngôn ngữ S/D/T + 7 luật Mamdani), `train.py`
  (full train: softmax/fuzzy/MLP baselines + 2 ablation + đề xuất, seed 42, CPU ~5 phút),
  `figures.py` (6 hình). Kết quả test (run canonical): FRF-MLP 84,33% acc / 63,00% macro-F1
  (+2,05 vs MLP thuần; vượt Text-CNN 61,11 / GRU 60,47 / m-BERT cased 62,69 đã công bố về macro-F1).
- `ml_ad/final/report/paper.tex`: báo cáo 10 trang tiếng Việt đúng chuẩn JTE-Template-Vie-01.2026
  (A4 3/2,5/3/2cm, Termes 11pt, ARTICLE INFO/ABSTRACT EN+VN, IMRAD, 12 refs IEEE đánh số tay,
  tiểu sử EN) — build bằng tectonic; PDF nộp tại `ml_ad/final/2611328 - .../` + `~/Downloads/`.
- Thực nghiệm "train dài hơn": nâng cap 40→100 epoch, patience 5→10 làm KÉM đi
  (mlp_feat 62,87→60,92 macro-F1) — checkpoint muộn chọn theo dev nhiễu không generalize;
  giữ protocol 40/5 (loss đã hội tụ ~0,02), có bằng chứng bão hòa cho phần bảo vệ.

### Added (2026-07-31 — Demo web FRF-MLP)
- `ml_ad/final/demo/`: app web minh hoạ model đã train — **Vite + React + TS** (frontend)
  + **FastAPI** (backend) load artifact (`mlp_feat.pt` + `artifacts.pkl`) để dự đoán real-time.
  UI dark theme chuẩn "kiểm duyệt MXH": composer dạng bình luận, verdict badge 3 màu
  (AN TOÀN/CÔNG KÍCH/THÙ GHÉT), prob bars, fusion strip `(1−λ)p_MLP + λ·p_mờ`, 3 biến mờ
  với hàm thành viên LOW/MED/HIGH, 7 luật Mamdani (luật kích hoạt nổi bật), token công
  kích tô sáng theo z-score. Có `/sample` lấy mẫu thật từ ViHSD. Verify qua browser cả 2
  nhánh HATE/CLEAN. `save_artifacts.py` dựng lại model (≈1 phút), `run.sh` khởi động cả 2.


### Changed (2026-07-31 — Demo: công cụ bắt-giải sâu + cấu hình máy)
- **Giải trình từng bước v2** (`Walkthrough.tsx`): bỏ accordion → mở hết 8 bước, TOC dính +
  scrollspy + smooth scroll. 2 chế độ chạy: **▶ Tự chạy** (auto-play qua 8 bước) và **Chạy từng
  bước** (bấm mới mở, có khoá). Mỗi bước thêm mục **"Bung số"** — công thức thay biến bằng giá trị
  thật của câu đang phân tích (μ trapezoid, min t-norm, giải mờ, softmax, fusion λ). Giáo sư bắt
  giải từng con số trên chính bình luận đó.
- **So sánh trực tiếp 3 mô hình**: endpoint `/compare` chạy cùng câu qua softmax (LogisticRegression
  C=4) · fuzzy-only · FRF-MLP, trả xác suất + nhãn + phân tích khác biệt. Bước phụ trong walkthrough
  + 4-lý-do chọn FRF-MLP (diễn giải/nhẹ CPU/nâng lớp thiểu số/tri thức tường minh) trong tab Ý tưởng.
  `save_softmax.py` lưu riêng softmax artifact (~12 giây, không retrain MLP).
- **Cấu hình máy chính xác** (paper §4.2): đo nvidia-smi/lscpu — Ryzen 9 9950X3D + RTX 5080 16GB
  + 60GB RAM + NVMe 1.8TB; làm rõ train chạy CPU (torch CPU-only build) — củng cố thesis nhẹ/không
  cần GPU. Ghi verified vào KNOWLEDGE.md.

### Added (2026-08-01 — Notebook FRF-MLP-ViHSD.ipynb)
- `ml_ad/final/FRF-MLP-ViHSD.ipynb`: notebook Jupyter **48 cell tiếng Việt, đã thực thi** (output thật +
  5 figure nhúng) đi kèm paper. Cấu trúc bám paper: Giới thiệu → Công trình liên quan → Phương pháp 4 bước
  (EDA ViHSD, tiền xử lý, TF-IDF 2 mức, lexicon log-odds, biến S/D/T, hàm thành viên + 7 luật Mamdani,
  kiến trúc MLP) → Thực nghiệm (softmax/MLP/FRF-MLP train trực tiếp, quét λ, bảng so sánh + ablation,
  ma trận nhầm lẫn, đường học) → suy diễn 1 câu (truy vết) → kết luận/hạn chế/hướng phát triển → refs.
  Kết quả đọc từ `outputs/metrics.json` (lần chạy chuẩn CPU seed 42: frf_mlp 84,33/63,00, +2,05, vượt
  m-BERT). Cell huấn luyện chạy trực tiếp pipeline (minh hoạ); note trung thực về biến thiên run-to-run.
- `ml_ad/final/build_notebook.py`: script dựng notebook (nbformat) — chạy lại khi sửa nội dung cell.

### Changed (2026-08-01 — Gom toàn bộ đồ án vào thư mục nộp bài của nhóm)
- `ml_ad/final/` → **`ml_ad/Nhom_8-Tu_Nhan_Anh/`**: thư mục nộp bài giờ **tự chứa toàn bộ công trình**
  (mã nguồn, `data/`, `outputs/` kết quả + biểu đồ, `report/` LaTeX + PDF, `demo/` web, notebook đã
  thực thi) — một nguồn duy nhất, không nhân bản.
- `build_notebook.py`: cell setup tự dò thư mục mã nguồn; kernelspec đổi `frf-mlp` → **`python3`**
  (mở được trên máy bất kỳ, không cần đăng ký kernel riêng).
- `requirements.txt`: bổ sung `jupyter`, `nbformat`, `nbconvert`, `ipykernel` — chạy lại notebook
  không cần cài thêm gì ngoài `run.sh`.
- `README.md` (thư mục nhóm): 3 tác giả, cấu trúc thư mục, bảng kết quả đầy đủ 9 mô hình, hướng dẫn
  chạy lại (demo/train/notebook/paper), ghi chú tái lập.
- Kiểm chứng sau khi chuyển: `run.sh` dựng lại venv từ đầu ở vị trí mới (torch 2.13.0+cpu, 3 artifact),
  demo trả `HATE` đúng, web 200; notebook execute lại tại chỗ → 48 cell, 5 figure, 0 lỗi, frf_mlp 84,33/63,00.
