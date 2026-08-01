# Nhóm 8 — Đồ án cuối kỳ Học máy nâng cao

**FRF-MLP: Mạng nơ-ron đa lớp hợp nhất luật mờ cho phát hiện ngôn từ công kích tiếng Việt (ViHSD)**

## Nhóm tác giả — ĐH Công nghệ Kỹ thuật TP.HCM

| Vai trò | Họ tên | MSSV |
|---|---|---|
| **Trưởng nhóm** | Nguyễn Phương Anh Tú | 2611328 |
| Thành viên | Đinh Hữu Quang Anh | 2611301 |
| Thành viên | Phạm Hiền Nhân | 2611316 |

Thư mục này **tự chứa toàn bộ công trình**: mã nguồn, dữ liệu, kết quả huấn luyện,
biểu đồ, bài báo (LaTeX + PDF), notebook đã thực thi và ứng dụng web demo.

## Cấu trúc thư mục

```
Nhom_8-Tu_Nhan_Anh/
├── FRF-MLP-ViHSD.ipynb      ★ Notebook 48 cell tiếng Việt, ĐÃ THỰC THI (output + 5 biểu đồ)
├── fuzzy.py                   Biến rõ S/D/T, hàm thành viên hình thang, 7 luật Mamdani
├── train.py                   Pipeline đặc trưng + huấn luyện 6 mô hình + đánh giá
├── figures.py                 Sinh 6 biểu đồ cho bài báo
├── save_artifacts.py          Đóng gói mô hình FRF-MLP cho web demo
├── save_softmax.py            Đóng gói baseline Softmax cho chức năng so sánh
├── requirements.txt           Toàn bộ phụ thuộc (torch CPU + jupyter)
├── data/                      Bộ dữ liệu ViHSD (33.398 mẫu, train/dev/test)
├── outputs/                   ★ KẾT QUẢ: metrics.json, curves.json, confusions.npz + 6 figures
├── report/                    ★ BÀI BÁO: paper.tex (định dạng JTE) + paper.pdf
├── demo/                      ★ WEB DEMO: FastAPI (api/) + React-TypeScript (web/) + run.sh
└── 2611328-...-JTE.pdf        Bản PDF bài báo nộp
```

## Kết quả thực nghiệm (ViHSD, tập test)

| Mô hình | Accuracy | macro-F1 |
|---|---|---|
| Fuzzy thuần (chỉ luật) | 62,71 | 41,56 |
| Softmax (TF-IDF) | 84,79 | 64,50 |
| MLP thuần | 83,82 | 60,95 |
| MLP + mờ mức quyết định | 83,79 | 60,97 |
| MLP + mờ mức đặc trưng | 84,16 | 62,87 |
| **FRF-MLP (đề xuất, λ\*=0,25)** | **84,33** | **63,00** |
| Text-CNN [1] | 86,69 | 61,11 |
| GRU [1] | 85,20 | 60,47 |
| m-BERT cased [1] | 86,88 | 62,69 |

**FRF-MLP cải thiện +2,05 điểm macro-F1** so với MLP thuần và **vượt m-BERT về macro-F1**
(63,00 vs 62,69) trong khi chỉ dùng ~10,3M tham số, **huấn luyện và suy diễn hoàn toàn trên CPU**
(~2 phút/mô hình) thay vì ~178M tham số của m-BERT.

Nghiên cứu loại bỏ (ablation) cho thấy mức đặc trưng đóng góp chính (+1,92) và mức quyết định
bổ trợ thêm, tổng +2,05 — hai mức mang thông tin **bổ trợ**, không thay thế nhau.

## Notebook — bằng chứng quy trình nghiên cứu

`FRF-MLP-ViHSD.ipynb` (48 cell, đã lưu sẵn output) trình bày đúng bố cục bài báo:

1. **Giới thiệu** — bài toán, động lực, đóng góp
2. **Công trình liên quan** — 3 hướng tiếp cận, khoảng trống nghiên cứu
3. **Phương pháp** — EDA ViHSD, tiền xử lý, TF-IDF hai mức (từ + ký tự, 40.000 chiều),
   từ điển log-odds Dirichlet, ba biến rõ S/D/T, hàm thành viên hình thang, 7 luật Mamdani,
   kiến trúc MLP
4. **Thực nghiệm** — huấn luyện trực tiếp Softmax / MLP / FRF-MLP (có log từng epoch),
   quét siêu tham số λ, bảng so sánh, ablation, ma trận nhầm lẫn, đường học
5. **Suy diễn** — truy vết đầy đủ một câu qua toàn bộ pipeline
6. **Kết luận** — ưu/nhược điểm, hạn chế, hướng phát triển

## Chạy lại toàn bộ

Chỉ cần **một lệnh** — tự dựng môi trường ảo, cài phụ thuộc, sinh tác vật rồi mở web demo:

```bash
cd ml_ad/Nhom_8-Tu_Nhan_Anh
bash demo/run.sh          # → http://localhost:3000 (web) + http://localhost:8000/docs (API)
```

Huấn luyện lại từ đầu và sinh biểu đồ:

```bash
.venv/bin/python train.py       # → outputs/metrics.json, curves.json, confusions.npz
.venv/bin/python figures.py     # → outputs/*.png
```

Mở hoặc chạy lại notebook:

```bash
.venv/bin/jupyter notebook FRF-MLP-ViHSD.ipynb
.venv/bin/jupyter nbconvert --to notebook --execute --inplace FRF-MLP-ViHSD.ipynb
```

Biên dịch bài báo:

```bash
tectonic -X compile report/paper.tex --outdir report/build
```

## Web demo — công cụ giải trình quyết định

Ứng dụng cho phép nhập một bình luận bất kỳ và xem **toàn bộ chuỗi suy luận**:
token được tô sáng theo trọng số từ điển, ba biến rõ S/D/T, độ thuộc của từng tập mờ
(kèm điểm gãy hình thang), mức kích hoạt của 7 luật, phân phối xác suất của MLP và của
hệ mờ, và bước hợp nhất cuối `(1−λ)·p_MLP + λ·p_mờ`. Có chế độ tự chạy từng bước và
bảng so sánh trực tiếp Softmax / Fuzzy / FRF-MLP trên cùng một câu.

## Ghi chú về tính tái lập

Số liệu chính thức trong bài báo và notebook được đọc từ `outputs/metrics.json` — lần chạy
chuẩn với `seed = 42` trên CPU đã lưu trong kho mã. Do tính không tất định của phép tính
đa luồng trên CPU, huấn luyện lại có thể lệch khoảng ±1–2 điểm macro-F1; các cell huấn luyện
trực tiếp trong notebook mang tính minh hoạ quy trình, điều này được ghi rõ trong notebook.

## Tài liệu tham khảo

[1] Luu, S. T., Nguyen, K. V., Nguyen, N. L.-T. *A Large-scale Dataset for Hate Speech
Detection on Vietnamese Social Media Texts.* IEA/AIE 2021.
