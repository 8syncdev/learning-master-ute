# Thuyết trình: Chia để trị - Bài toán cặp điểm gần nhất

Chủ đề được chọn: **Closest Pair of Points** - tìm hai điểm gần nhau nhất trên mặt phẳng.

Lý do chọn chủ đề này:

- Thuộc nhóm **Chia để trị** rõ ràng: chia tập điểm thành hai nửa, giải từng nửa, rồi gộp kết quả ở vùng biên.
- Khó hơn Binary Search và Merge Sort, nhưng vẫn có thể giải thích trực quan bằng hình học.
- Có phân tích độ phức tạp đẹp: từ vét cạn `O(n^2)` tối ưu xuống `O(n log n)`.
- Dễ demo bằng code C++ và ví dụ tọa độ 2D.

## Cấu trúc theo đúng yêu cầu trên lớp

1. Chia để trị là gì?
2. Mục tiêu của nhóm thuật toán chia để trị.
3. Các bước thực hiện: Divide - Conquer - Combine.
4. Trình bày ít nhất 3 thuật toán tiêu biểu trong nhóm:
   - Quick Sort
   - Karatsuba Multiplication
   - Closest Pair of Points
5. Với từng thuật toán:
   - Ý tưởng
   - Thuật toán
   - Công thức `T(n)`
   - Suy ra Big O
   - Demo
   - Đánh giá và khuyến nghị sử dụng
6. Ứng dụng thực tế của chia để trị.

## File trong thư mục

- `presentation.marp.md`: slide chính, có thể dùng Marp để xuất PDF/PPTX.
- `speaker_notes.md`: lời thoại thuyết trình dễ nói trên lớp.
- `demo_closest_pair.cpp`: demo C++ cho thuật toán Closest Pair of Points.

## Cách chạy demo

```bash
g++ demo_closest_pair.cpp -std=c++17 -O2 -Wall -o closest_pair
./closest_pair
```

## Gợi ý xuất slide

Nếu có Marp CLI:

```bash
marp presentation.marp.md --pdf
marp presentation.marp.md --pptx
```
