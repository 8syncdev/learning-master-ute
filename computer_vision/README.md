# Computer Vision

Thư mục môn **Computer Vision** (chương trình thạc sĩ — UTE).
Học viên: Nguyễn Phương Anh Tú — MSHV 2611328.

## Các chủ đề

### 1. Non-Maximum Suppression (NMS) — gom đối tượng nhận dạng chồng lấp
Thư mục: [`non_maximum_suppression/`](non_maximum_suppression/)

| File | Nội dung |
|---|---|
| `nghien_cuu_NMS.md` | Tài liệu nghiên cứu sâu (lý thuyết, thuật toán, biến thể, ứng dụng, NMS-free, mã minh hoạ) |
| `slide_outline.md` | Dàn ý slide (Markdown) — nguồn để sinh `.pptx` |
| `references.md` | Danh mục tài liệu tham khảo (đã kiểm chứng nguồn, kèm link arXiv/hội nghị) |
| `build_slides.py` | Bộ sinh `.pptx` từ Markdown — **chỉ dùng thư viện chuẩn Python** |
| `NMS_slides.pptx` | Bộ slide đã sinh (18 trang, mở được bằng PowerPoint/LibreOffice/Google Slides) |

## Sinh slide PowerPoint từ Markdown

> **Trả lời câu hỏi "có skill nào để AI tạo slide pptx từ md research không?":**
> Trong môi trường này **không có skill chuyên dụng** md→pptx (hai skill hiện có là `ans` để ghi Q&A và `last30days` để khảo sát mạng xã hội), và máy **không có** `pandoc`/`marp`/`python-pptx`/`pip`.
> Vì vậy đã viết sẵn `build_slides.py` — một bộ chuyển Markdown → `.pptx` **không phụ thuộc thư viện ngoài** (tự dựng gói OOXML bằng `zipfile` + XML của thư viện chuẩn). Đây chính là cách AI Agent tạo `.pptx` trực tiếp từ file research `.md`.

Cách dùng:
```bash
cd non_maximum_suppression
python build_slides.py                       # slide_outline.md -> NMS_slides.pptx
python build_slides.py slide_outline.md out.pptx   # tuỳ chỉnh input/output
```

Định dạng Markdown đầu vào: mỗi slide ngăn bởi dòng `---`; dòng `# ...` là tiêu đề; dòng `- ` là bullet (thụt lề 2 dấu cách = tăng cấp). Sửa nội dung trong `slide_outline.md` rồi chạy lại là có slide mới — không cần đụng vào XML.

Kiểm chứng tính hợp lệ của file (tuỳ chọn, cần LibreOffice có sẵn trên máy):
```bash
soffice --headless --convert-to pdf NMS_slides.pptx
```

### Lưu ý về công cụ thay thế (nếu cần slide đẹp hơn / có hình ảnh)
- **python-pptx** (`pip install python-pptx`): API tạo slide phong phú (ảnh, bảng, biểu đồ) — cần cài bằng pip.
- **Marp** (`npx @marp-team/marp-cli slides.md --pptx`): chủ đề đẹp, nhưng mỗi slide là ảnh render (khó sửa chữ trong PowerPoint) và cần Node + tải gói.
- `build_slides.py` ở đây ưu tiên: chạy ngay, không phụ thuộc, và **chữ trong slide vẫn sửa được** trong PowerPoint.
