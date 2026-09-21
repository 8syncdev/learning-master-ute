---
marp: true
theme: default
paginate: true
size: 16:9
---

# Chia để trị
## Closest Pair of Points

**Tìm cặp điểm gần nhất trên mặt phẳng**  
Chủ đề: thuật toán chia để trị, khó hơn Merge Sort / Binary Search nhưng vẫn dễ demo

---

# 1. Chia để trị là gì?

**Chia để trị** là chiến lược giải bài toán lớn bằng cách:

1. **Divide**: chia bài toán thành các bài toán con nhỏ hơn.
2. **Conquer**: giải từng bài toán con.
3. **Combine**: gộp kết quả con thành kết quả cuối cùng.

Ví dụ đời thường: muốn tìm lỗi trong một project lớn, ta chia theo module: frontend, backend, database, config; xử lý từng phần rồi ghép kết luận.

---

# 2. Mục tiêu của nhóm thuật toán này

Mục tiêu chính:

- Giảm bài toán lớn thành nhiều bài toán nhỏ dễ xử lý hơn.
- Tận dụng cấu trúc lặp lại của bài toán.
- Giảm độ phức tạp so với vét cạn.
- Dễ phân tích bằng công thức truy hồi `T(n)`.

Ví dụ: Closest Pair giảm từ `O(n^2)` xuống `O(n log n)`.

---

# 3. Các bước thực hiện

Mẫu chung:

```text
solve(problem):
    if problem đủ nhỏ:
        giải trực tiếp
    chia problem thành problem_left và problem_right
    ans_left = solve(problem_left)
    ans_right = solve(problem_right)
    return combine(ans_left, ans_right)
```

Điểm khó nhất thường nằm ở bước **Combine**.

---

# 4. Thuật toán tiêu biểu số 1: Quick Sort

## Ý tưởng

Chọn một phần tử làm **pivot**, chia mảng thành:

- Nhóm nhỏ hơn pivot
- Pivot
- Nhóm lớn hơn pivot

Sau đó sắp xếp đệ quy hai nhóm còn lại.

---

# Quick Sort - thuật toán

```text
quickSort(a, l, r):
    if l >= r: return
    pivot = choosePivot(a, l, r)
    p = partition(a, l, r, pivot)
    quickSort(a, l, p - 1)
    quickSort(a, p + 1, r)
```

## T(n)

Trường hợp trung bình:

```text
T(n) = 2T(n/2) + O(n)
```

Suy ra: **O(n log n)**.

Trường hợp xấu: **O(n²)** nếu chia quá lệch.

---

# Quick Sort - demo ngắn

Mảng: `[7, 2, 9, 1, 5]`

Chọn pivot = `5`:

```text
[2, 1] | 5 | [7, 9]
```

Tiếp tục xử lý hai nửa:

```text
[1, 2] | 5 | [7, 9]
=> [1, 2, 5, 7, 9]
```

## Khuyến nghị

Dùng khi cần sắp xếp nhanh trong bộ nhớ, nhưng nên chọn pivot tốt hoặc random pivot.

---

# 5. Thuật toán tiêu biểu số 2: Karatsuba

## Ý tưởng

Nhân hai số lớn bằng cách chia mỗi số thành hai nửa.

Thay vì nhân 4 lần như cách thường, Karatsuba chỉ cần **3 phép nhân lớn**.

Với:

```text
x = a * 10^m + b
y = c * 10^m + d
```

Cần tính `ac`, `bd`, và `(a+b)(c+d)`.

---

# Karatsuba - thuật toán

```text
karatsuba(x, y):
    if x hoặc y nhỏ: return x * y
    tách x thành a, b
    tách y thành c, d
    z2 = karatsuba(a, c)
    z0 = karatsuba(b, d)
    z1 = karatsuba(a + b, c + d) - z2 - z0
    return z2 * 10^(2m) + z1 * 10^m + z0
```

## T(n)

```text
T(n) = 3T(n/2) + O(n)
```

Suy ra: **O(n^log₂3) ≈ O(n^1.585)**.

---

# Karatsuba - demo ngắn

Nhân `1234 × 5678`:

```text
x = 12 | 34
y = 56 | 78
```

Tính:

```text
z2 = 12 × 56
z0 = 34 × 78
z1 = (12 + 34)(56 + 78) - z2 - z0
```

## Khuyến nghị

Dùng khi nhân số nguyên rất lớn, ví dụ thư viện Big Integer, mật mã học, tính toán số học hiệu năng cao.

---

# 6. Thuật toán chính: Closest Pair of Points

## Bài toán

Cho `n` điểm trên mặt phẳng 2D.  
Tìm hai điểm có khoảng cách Euclid nhỏ nhất.

Ví dụ:

```text
(2,3), (12,30), (40,50), (5,1), (12,10), (3,4)
```

Cặp gần nhất là `(2,3)` và `(3,4)`.

---

# Vì sao không vét cạn?

Cách đơn giản:

```text
so sánh mọi cặp điểm
```

Số cặp cần xét:

```text
n(n - 1) / 2
```

Độ phức tạp:

```text
O(n²)
```

Nếu `n = 1,000,000`, số cặp là cực lớn.  
Ta cần cách tốt hơn.

---

# Ý tưởng chia để trị

1. Sắp xếp điểm theo trục `x`.
2. Chia tập điểm thành nửa trái và nửa phải.
3. Tìm cặp gần nhất ở nửa trái.
4. Tìm cặp gần nhất ở nửa phải.
5. Lấy `d = min(d_left, d_right)`.
6. Kiểm tra thêm các điểm nằm gần đường chia vì cặp gần nhất có thể nằm ở hai bên.

---

# Điểm khó: bước Combine

Sau khi có `d`, chỉ cần xét các điểm cách đường chia ít hơn `d`.

Vùng này gọi là **strip**.

```text
Left side     |     Right side
              |
      o       |   o
   o          |
--------------|-------------- đường chia
      o       |       o
              |
```

Trong strip, sắp xếp theo `y`, mỗi điểm chỉ cần so sánh với vài điểm kế tiếp.

---

# Closest Pair - thuật toán

```text
closest(points sorted by x):
    if n <= 3:
        return bruteForce(points)

    mid = n / 2
    left = points[0..mid-1]
    right = points[mid..n-1]

    d_left = closest(left)
    d_right = closest(right)
    d = min(d_left, d_right)

    strip = các điểm có |x - midX| < d
    sort strip theo y

    kiểm tra các cặp gần nhau trong strip
    return min(d, best_strip)
```

---

# T(n) của Closest Pair

Nếu cài đặt tốt:

```text
T(n) = 2T(n/2) + O(n)
```

Theo Master Theorem:

```text
T(n) = O(n log n)
```

Nếu mỗi lần đệ quy lại sort strip từ đầu, có thể thành:

```text
O(n log² n)
```

Vì vậy cần tối ưu bước sắp xếp / gộp theo `y`.

---

# Demo bằng ví dụ nhỏ

Tập điểm:

```text
A(2,3), B(12,30), C(40,50),
D(5,1), E(12,10), F(3,4)
```

Sau khi sort theo `x`:

```text
A(2,3), F(3,4), D(5,1), E(12,10), B(12,30), C(40,50)
```

Chia đôi:

```text
Left:  A, F, D
Right: E, B, C
```

Left tìm được `A-F`, khoảng cách `√2`.

---

# Đánh giá Closest Pair

## Ưu điểm

- Nhanh hơn vét cạn rất nhiều.
- Thể hiện rõ tư duy chia để trị.
- Có ứng dụng thực tế trong không gian 2D.

## Nhược điểm

- Cài đặt khó hơn Quick Sort.
- Cần hiểu hình học và quản lý dữ liệu đã sắp xếp.
- Dễ sai ở bước strip/combine.

## Khuyến nghị

Dùng khi `n` lớn và cần tìm khoảng cách gần nhất giữa các điểm.

---

# Ứng dụng thực tế

Chia để trị thường được ứng dụng trong:

- Sắp xếp dữ liệu lớn.
- Tìm kiếm và tối ưu hóa.
- Xử lý hình học tính toán.
- Nhân số lớn trong mật mã học.
- Xử lý ảnh, chia vùng ảnh để phân tích.
- Game / bản đồ: tìm đối tượng gần nhau.
- Hệ thống định vị: tìm trạm, xe, robot, cảm biến gần nhất.

---

# So sánh nhanh 3 thuật toán

| Thuật toán | Bài toán | T(n) | Big O | Khi nên dùng |
|---|---|---:|---:|---|
| Quick Sort | Sắp xếp | `2T(n/2)+O(n)` | TB `O(n log n)` | Sort nhanh trong RAM |
| Karatsuba | Nhân số lớn | `3T(n/2)+O(n)` | `O(n^1.585)` | Big Integer |
| Closest Pair | Cặp điểm gần nhất | `2T(n/2)+O(n)` | `O(n log n)` | Hình học 2D, dữ liệu lớn |

---

# Kết luận

- Chia để trị giúp biến bài toán lớn thành các bài toán nhỏ hơn.
- Công thức quen thuộc: **Divide - Conquer - Combine**.
- Điểm quan trọng nhất là bước **Combine**.
- Closest Pair of Points là ví dụ hay vì:
  - Khó hơn Merge Sort / Binary Search.
  - Có trực quan hình học.
  - Giảm từ `O(n²)` xuống `O(n log n)`.

---

# Câu hỏi gợi mở

Nếu dữ liệu là điểm trong không gian 3D thay vì 2D thì thuật toán Closest Pair cần thay đổi gì?

Gợi ý:

- Khoảng cách dùng thêm trục `z`.
- Vùng strip trở thành một vùng không gian 3D.
- Bước combine phức tạp hơn.
