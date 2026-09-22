# Bài tập Thuật toán - QuickSort và Tìm kiếm nhị phân

**Sinh viên:** Nguyễn Phương Anh Tú  
**MSSV:** 2611328

## Dữ liệu

Mảng đề bài:

```text
[10, 33, 25, 45, 50, 100, 72, 95, 90, 85, 61]
```

> Đề ghi "mảng đã được sắp xếp", nhưng dãy số trên thực tế chưa tăng dần. Vì câu 1 yêu cầu minh họa sắp xếp, bài làm xem đây là mảng ban đầu, dùng QuickSort trước rồi mới Binary Search.

Sau QuickSort:

```text
[10, 25, 33, 45, 50, 61, 72, 85, 90, 95, 100]
```

## 1. QuickSort tăng dần

Dùng **Lomuto partition**, chọn phần tử cuối đoạn làm pivot.

| Partition | Đoạn | Pivot | Mảng sau partition |
|---:|---:|---:|---|
| 1 | 0..10 | 61 | 10, 33, 25, 45, 50, 61, 72, 95, 90, 85, 100 |
| 2 | 0..4 | 50 | 10, 33, 25, 45, 50, 61, 72, 95, 90, 85, 100 |
| 3 | 0..3 | 45 | 10, 33, 25, 45, 50, 61, 72, 95, 90, 85, 100 |
| 4 | 0..2 | 25 | 10, 25, 33, 45, 50, 61, 72, 95, 90, 85, 100 |
| 5 | 6..10 | 100 | 10, 25, 33, 45, 50, 61, 72, 95, 90, 85, 100 |
| 6 | 6..9 | 85 | 10, 25, 33, 45, 50, 61, 72, 85, 90, 95, 100 |
| 7 | 8..9 | 95 | 10, 25, 33, 45, 50, 61, 72, 85, 90, 95, 100 |

QuickSort: trung bình **O(n log n)**, xấu nhất **O(n²)**.

## 2. Binary Search tìm 72

Mảng đã tăng dần:

```text
Index:  0   1   2   3   4   5   6   7   8   9   10
Value: 10  25  33  45  50  61  72  85  90  95  100
```

- Lần 1: `low=0, high=10, mid=5, A[5]=61`. Vì `72>61`, tìm `[6..10]`.
- Lần 2: `low=6, high=10, mid=8, A[8]=90`. Vì `72<90`, tìm `[6..7]`.
- Lần 3: `low=6, high=7, mid=6, A[6]=72`. **Tìm thấy**.

Kết quả: **index 0-based = 6**, tức **vị trí thứ 7** nếu đếm từ 1.

## 3. Binary Search đệ quy, T(n), Big O

```text
binarySearch(A, target, low, high):
    if low > high:
        return -1

    mid = low + (high - low) / 2

    if A[mid] == target:
        return mid

    if target < A[mid]:
        return binarySearch(A, target, low, mid - 1)

    return binarySearch(A, target, mid + 1, high)
```

Mỗi lần chỉ còn khoảng nửa dữ liệu:

```text
T(n) = T(n/2) + O(1)
```

Sau k lần chia: `n / 2^k = 1` nên `k = log2(n)`.

**Kết luận:** `T(n) = O(log n)`.

- Best case: **O(1)**
- Average/Worst case: **O(log n)**
- Với `n=11`, tối đa khoảng `floor(log2(11))+1 = 4` lần kiểm tra.
- Riêng tìm `72` trong bài này: **3 bước**.

Runtime nanosecond đo bằng chương trình chỉ mang tính tham khảo vì phụ thuộc CPU/compiler. Big O mới là kết luận tổng quát.

## 4. Demo

File: `2611328_NguyenPhuongAnhTu.cpp`

```bash
g++ -std=c++17 -O2 -Wall -Wextra 2611328_NguyenPhuongAnhTu.cpp -o demo
./demo
```

`demo.html` là bản minh họa offline từng bước, chỉ cần mở bằng trình duyệt.
