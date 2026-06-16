---
name: ans
description: Ghi lại câu hỏi–trả lời theo từng buổi học trong repo này (thư mục ml_ad/b{N}/note.md). Use when the user asks to note/save a Q&A for a session — triggers like "note câu trả lời", "note câu hỏi buổi bN", "lưu Q&A buổi này", "ghi lại đáp án buổi ...". Defines the per-session note convention, the Q&A entry template, and the commit step.
---

# ans — Note Q&A theo buổi

## Mục đích
Mỗi buổi học có một file note tích luỹ các cặp **câu hỏi → trả lời**. Skill này
chuẩn hoá: file đặt ở đâu, định dạng mỗi entry, và bước commit.

## Quy ước file
- Một buổi = một thư mục `ml_ad/b{N}/` (N = số buổi, ví dụ `b1`, `b3`).
- Note của buổi nằm tại **`ml_ad/b{N}/note.md`** — một file, **nhiều** Q&A nối tiếp.
- Nếu file/thư mục chưa có → tạo mới với tiêu đề `# Buổi b{N} — Note`.
- Nếu đã có → **append** entry mới xuống cuối, **không** ghi đè entry cũ.

## Quy trình
1. **Xác định buổi.** Lấy số `N` từ yêu cầu của user (vd "buổi b3"). Nếu user
   không nói rõ → hỏi lại đúng 1 câu, hoặc dùng buổi đang làm việc gần nhất.
2. **Định vị file** `ml_ad/b{N}/note.md`. Tạo nếu thiếu, đọc nếu đã có.
3. **Soạn câu trả lời cho đúng — không bịa.** Trả lời phải chính xác về kỹ thuật:
   - nêu trực tiếp kết luận trước, rồi **LUÔN giải thích lý do (vì sao)** — bắt buộc;
   - nếu liên quan code trong repo, đọc code đó và bám sát (đúng tên hàm/biến);
   - kèm ví dụ code chạy được khi câu hỏi mang tính thực hành.
4. **Append theo template** bên dưới.
5. **Auto commit & push — BẮT BUỘC.** Sau mỗi lần tạo/append note, **luôn tự động** commit và push, **không hỏi lại**:
   ```bash
   git add ml_ad/b{N}/note.md && \
   git commit -m "b{N}: note Q{n} — <tóm tắt câu hỏi>" && \
   git push -q origin main
   ```
   Message gắn số `Q{n}` của entry vừa thêm. Đây là bước kết thúc bắt buộc — note chưa push coi như chưa xong.

## Template mỗi entry
Append nguyên khối này (đánh số `Q` tăng dần trong buổi):

````markdown
## Q{n}. <câu hỏi tóm tắt 1 dòng>

**Hỏi:** <câu hỏi đầy đủ của user>

**Trả lời ngắn:** <1–3 câu chốt đáp án>

**Vì sao:** <giải thích LÝ DO — cơ chế/nguyên nhân; nếu là lựa chọn (chọn X thay vì Y) phải nêu rõ lý do chọn. BẮT BUỘC, không bao giờ bỏ.>

**Chi tiết:**
- <ý 1: lý do/cơ chế>
- <ý 2: ...>

**Ví dụ:** *(bỏ qua nếu không cần code)*
```python
# code minh hoạ, chạy được
```
````

## Nguyên tắc
- **LUÔN GIẢI THÍCH LÝ DO TẠI SAO.** Mọi đáp án phải nêu *vì sao* (cơ chế / nguyên nhân), không chỉ "làm thế nào" hay kết luận suông. Có lựa chọn giữa nhiều phương án → bắt buộc nói rõ **lý do chọn cái này thay vì cái kia**.
- **Tiếng Việt**, giữ nguyên thuật ngữ tiếng Anh (`fit_transform`, `data leakage`, ...).
- Một entry = một câu hỏi. Nhiều câu hỏi → nhiều entry trong cùng `note.md`.
- Ngắn gọn, đủ ý. Ưu tiên gạch đầu dòng hơn đoạn văn dài.
- Câu hỏi phức tạp: trong **Chi tiết** được phép mở rộng — sub-heading `###`, các bước đánh số, hoặc blockquote cho "quy tắc vàng".
- Không sửa entry cũ trừ khi user yêu cầu sửa lại đáp án.

## Ví dụ vị trí
`ml_ad/b3/note.md` — buổi 3: "vì sao train dùng `fit_transform`, test dùng `transform`".
