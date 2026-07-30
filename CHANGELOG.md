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
