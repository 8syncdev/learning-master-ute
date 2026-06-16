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
   - nêu trực tiếp kết luận trước, rồi lý do;
   - nếu liên quan code trong repo, đọc code đó và bám sát (đúng tên hàm/biến);
   - kèm ví dụ code chạy được khi câu hỏi mang tính thực hành.
4. **Append theo template** bên dưới.
5. **Commit + push**: `git add ml_ad/b{N}/note.md && git commit -m "b{N}: note — <tóm tắt câu hỏi>" && git push`.

## Template mỗi entry
Append nguyên khối này (đánh số `Q` tăng dần trong buổi):

````markdown
## Q{n}. <câu hỏi tóm tắt 1 dòng>

**Hỏi:** <câu hỏi đầy đủ của user>

**Trả lời ngắn:** <1–3 câu chốt đáp án>

**Chi tiết:**
- <ý 1: lý do/cơ chế>
- <ý 2: ...>

**Ví dụ:** *(bỏ qua nếu không cần code)*
```python
# code minh hoạ, chạy được
```
````

## Nguyên tắc
- **Tiếng Việt**, giữ nguyên thuật ngữ tiếng Anh (`fit_transform`, `data leakage`, ...).
- Một entry = một câu hỏi. Nhiều câu hỏi → nhiều entry trong cùng `note.md`.
- Ngắn gọn, đủ ý. Ưu tiên gạch đầu dòng hơn đoạn văn dài.
- Câu hỏi phức tạp: trong **Chi tiết** được phép mở rộng — sub-heading `###`, các bước đánh số, hoặc blockquote cho "quy tắc vàng".
- Không sửa entry cũ trừ khi user yêu cầu sửa lại đáp án.

## Ví dụ vị trí
`ml_ad/b3/note.md` — buổi 3: "vì sao train dùng `fit_transform`, test dùng `transform`".
