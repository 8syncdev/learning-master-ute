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

