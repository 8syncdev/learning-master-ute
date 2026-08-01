# Nhóm 8 — Đồ án cuối kỳ Học máy nâng cao

**FRF-MLP: Mạng nơ-ron đa lớp hợp nhất luật mờ cho phát hiện ngôn từ công kích tiếng Việt (ViHSD)**

## Nhóm tác giả — ĐH Công nghệ Kỹ thuật TP.HCM

| Vai trò | Họ tên | MSSV |
|---|---|---|
| **Trưởng nhóm** | Nguyễn Phương Anh Tú | 2611328 |
| Thành viên | Đinh Hữu Quang Anh | 2611301 |
| Thành viên | Phạm Hiền Nhân | 2611316 |

## Nội dung

- **`FRF-MLP-ViHSD.ipynb`** — notebook 48 cell tiếng Việt, **đã thực thi sẵn** (giữ nguyên toàn bộ
  output huấn luyện + 5 biểu đồ nhúng). Giải thích chi tiết từng bước theo đúng bố cục bài báo:

  1. Giới thiệu — bài toán, động lực, đóng góp
  2. Công trình liên quan — 3 hướng tiếp cận, khoảng trống nghiên cứu
  3. Phương pháp — EDA ViHSD, tiền xử lý, TF-IDF 2 mức (từ + ký tự, 40.000 chiều),
     từ điển log-odds Dirichlet, 3 biến rõ S/D/T, hàm thành viên hình thang,
     7 luật mờ Mamdani, kiến trúc MLP (~10,3M tham số)
  4. Thực nghiệm — huấn luyện trực tiếp Softmax / MLP / FRF-MLP, quét siêu tham số λ,
     bảng so sánh với Text-CNN / GRU / m-BERT, nghiên cứu loại bỏ (ablation),
     ma trận nhầm lẫn, đường học
  5. Suy diễn — truy vết đầy đủ một câu qua toàn bộ pipeline
  6. Kết luận — ưu/nhược điểm, hạn chế, hướng phát triển

## Kết quả chính (ViHSD, tập test)

| Mô hình | Accuracy | macro-F1 |
|---|---|---|
| Softmax (TF-IDF) | 84,79 | 64,50 |
| MLP thuần | 83,82 | 60,95 |
| **FRF-MLP (đề xuất)** | **84,33** | **63,00** |
| m-BERT cased (tham chiếu) | 86,88 | 62,69 |

FRF-MLP cải thiện **+2,05 điểm macro-F1** so với MLP thuần và **vượt m-BERT về macro-F1**
trong khi chỉ dùng ~10,3M tham số, huấn luyện và suy diễn hoàn toàn **trên CPU**.

## Cách chạy lại

Notebook tự tìm thư mục mã nguồn (`../final`) nên chạy được ngay tại thư mục này:

```bash
cd ml_ad/final && bash demo/run.sh      # (tuỳ chọn) dựng môi trường + web demo
cd ../Nhom_8-Tu_Nhan_Anh
../final/.venv/bin/jupyter notebook FRF-MLP-ViHSD.ipynb
```

Chạy lại toàn bộ notebook không cần mở giao diện:

```bash
../final/.venv/bin/jupyter nbconvert --to notebook --execute --inplace FRF-MLP-ViHSD.ipynb
```

## Mã nguồn đầy đủ

Toàn bộ mã nguồn, bài báo LaTeX và ứng dụng web demo nằm ở [`../final/`](../final/):

- `fuzzy.py` — trích xuất biến rõ, hàm thành viên, hệ luật Mamdani
- `train.py` — pipeline đặc trưng + huấn luyện 6 mô hình
- `figures.py`, `save_artifacts.py` — sinh biểu đồ và tạo tác vật cho demo
- `report/paper.tex` — bài báo định dạng JTE
- `demo/` — web demo (FastAPI + React) giải trình từng bước quyết định của mô hình
