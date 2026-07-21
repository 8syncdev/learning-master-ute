# STATE (8sync managed — live plan; rewrite ở MỖI phase-boundary, đọc đầu phiên)

## Goal
Đồ án cuối kỳ Học máy nâng cao (ml_ad): MLP + fuzzy logic phân loại ngôn từ công kích ViHSD, full train + báo cáo PDF chuẩn JTE. — **HOÀN THÀNH 2026-07-21.**

## Definition of Done
- [x] Full train trên ViHSD split gốc: baselines (softmax, fuzzy, MLP) + 2 ablation + FRF-MLP đề xuất
- [x] Kết quả: FRF-MLP 84,46% acc / 62,95% macro-F1 (+1,92 vs MLP; > baseline công bố về macro-F1)
- [x] Báo cáo 10 trang tiếng Việt đúng JTE-Template-Vie-01.2026, 12 refs IEEE verified, build tectonic
- [x] PDF verify trực quan từng trang; giao `ml_ad/final/2611328 - .../` + `~/Downloads/`

## Checklist
- [x] fuzzy.py · train.py · figures.py · report/paper.tex · outputs/{metrics,curves,confusions,figures}

## Current step
Done — chờ feedback (tên nhóm/đồng tác giả COMVIS_2026 nếu cần thêm vào báo cáo).

## Next
Nếu nộp nhóm: bổ sung tên 2 thành viên (Quang Anh, Phạm Hiền Nhân) vào phần tác giả + tiểu sử rồi rebuild (`tectonic -X compile report/paper.tex --outdir report/build`).

## Assumptions (auto-decided — user can correct)
- Đề tài chốt theo chat nhóm: phân loại văn bản công kích MXH tiếng Việt (ViHSD), MLP + fuzzy logic.
- Báo cáo đứng tên 1 tác giả (Nguyễn Phương Anh Tú, MSHV 2611328); email 2611328@student.hcmute.edu.vn; ORCID placeholder.
- FRF-MLP = feature fusion + λ-fusion quyết định (gate học được bị loại vì kém hơn — xem KNOWLEDGE failure).

## Open questions / blockers
_none_

## Handoff (compaction)
Kết quả + lệnh tái lập: `cd ml_ad/final && .venv/bin/python train.py && .venv/bin/python figures.py`; build PDF: `~/.local/bin/tectonic -X compile report/paper.tex --outdir report/build`.
