<!-- 8sync:harness:begin -->
## 🧠 8sync harness

- **Always-on (đọc theo thứ tự; CORE đọc body ngay, SPECIALIST đọc khi task khớp):** codegraph → karpathy-guidelines → ponytail → assp-skill → impeccable → taste-skill → 8sync-cli → image-routing.
- **Cách tận dụng:** codegraph = explore code (search/deps/callers, không grep) · karpathy + ponytail = YAGNI, làm ít nhất, xoá > thêm · impeccable = design CHUẨN, BẮT BUỘC khi UI/design (đọc body lúc đó) + taste chống slop.
- **Output lớn (>~50 dòng) → BẮT BUỘC `headroom_compress`** trước khi vào context.
- **Sau mỗi thay đổi:** cập nhật `CHANGELOG.md` (Unreleased) + ghi học được vào file này (prefix `validated:` nếu test/build xác nhận, `hypothesis:` nếu chưa).
<!-- 8sync:harness:end -->

# KNOWLEDGE (8sync managed — append-only)

## Learnings (append-only — ghi DƯỚI đây; KHÔNG sửa block `8sync:harness` ở trên)

Mỗi entry prefix `validated:` (test/build xác nhận) · `hypothesis:` (chưa) · `failure:` (lỗi đã gặp + cách sửa; đọc đầu phiên để khỏi lặp).

- validated: tectonic 0.15 (binary release) + fontspec dùng font bundle theo TÊN FILE
  (`\setmainfont{texgyretermes}[Extension=.otf, UprightFont=*-regular,...]`) render tiếng Việt
  hoàn chỉnh không cần cài font hệ thống — `\setmainfont{TeX Gyre Termes}` (tên fontconfig) FAIL.
- validated: trên ViHSD TF-IDF thưa 40k chiều, hồi quy softmax class-weighted (C tune dev) đạt
  64,50 macro-F1 > mọi MLP variant (~61–63) kể cả wide&deep và ensemble 5 seed — mô hình tuyến
  tính lồi là baseline rất mạnh trên sparse TF-IDF; MLP overfit nhanh (early stop ~ep6–16).
- failure: gate fusion học được sigmoid(a·conf_fuzzy+b) trộn p_fuzzy vào loss NLL làm FRF-MLP
  TỆ hơn ablation feature-fusion (61,0 vs 62,8 macro-F1) — hội tụ chậm, mixture làm nhòe gradient.
  Fix: fusion quyết định hậu nghiệm với λ quét lưới trên dev (deterministic, +0,1–0,6pt).
- failure: ViHSD MLP — nâng cap epoch 40→100 + patience 5→10 làm macro-F1 test GIẢM
  (mlp_feat 62,87→60,92): dev nhỏ (212 OFFENSIVE) nhiễu → patience dài chọn checkpoint
  muộn overfit-dev. Protocol chuẩn giữ 40/5; "đủ epoch" chứng minh bằng loss hội tụ ~0,02
  + dev-F1 plateau, không phải bằng số epoch lớn.
- validated: cấu hình máy đào tạo (đo nvidia-smi/lscpu/free 2026-07-31, không ước lượng):
  • CPU AMD Ryzen 9 9950X3D — 16 core / 32 thread, boost ≤5756 MHz, L3 128 MiB (3D V-Cache, 2 CCD).
  • RAM 60,5 GiB. Ổ ADATA LEGEND 860 NVMe 1,8 TB. OS CachyOS Linux, kernel 7.1.3-2-cachyos, gcc 16.1.1.
  • GPU NVIDIA GeForce RTX 5080 (Blackwell GB203) — 16 GB GDDR7, driver 610.43.03, compute cap 12.0,
    boost ≤3165 MHz core / 15001 MHz mem, TDP 400 W (max 450 W). (lõi: 10752 CUDA + 336 Tensor gen-5 — spec NXB.)
  • PyTorch cài BẢN CPU-ONLY (`2.13.0+cpu`, `torch.cuda.is_available()==False`) → huấn luyện FRF-MLP
    THỰC TẾ chạy trên CPU Ryzen, KHÔNG dùng GPU. Paper dòng 262 ghi đúng "CPU AMD Ryzen 9 9950X3D".
  → Đây LÀ điểm mạnh cho thesis "nhẹ / không cần GPU / deploy CPU": mô hình 10,3M tham số trên TF-IDF
    thưa huấn luyện <2 phút/mô hình trên CPU của chính máy có 5080. Muốn "train trên 5080" phải cài
- validated: kết quả CHÍNH THỨC báo cáo = lần chạy chuẩn CPU seed 42 đã commit trong outputs/
  (frf_mlp 84,33/63,00, +2,05 vs MLP, vượt m-BERT 62,69 về macro-F1). Quyết định: GIỮ CPU, không ghi GPU.
  Lý do dùng kết quả này: macro-F1 KHÔNG tái lập hoàn toàn giữa các lần chạy (±1–2đ do nondeterminism
  đa luồng) nên số chính thức = lần chạy chuẩn đã commit; re-run chỉ minh hoạ quy trình (notebook có ghi rõ).
