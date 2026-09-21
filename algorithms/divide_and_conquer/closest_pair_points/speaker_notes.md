# Speaker Notes - Chia để trị: Closest Pair of Points

## Slide 1 - Mở đầu

Chào thầy/cô và các bạn. Hôm nay nhóm em trình bày về nhóm thuật toán **chia để trị**. Thay vì chọn ví dụ quen thuộc như Merge Sort hay Binary Search, nhóm em chọn một bài toán khó hơn nhưng vẫn rất trực quan: **Closest Pair of Points**, tức là tìm hai điểm gần nhau nhất trên mặt phẳng.

## Slide 2 - Chia để trị là gì?

Chia để trị là cách giải bài toán lớn bằng cách chia nó thành nhiều bài toán nhỏ hơn. Mình có thể hiểu giống như khi debug một project lớn: không ai đọc toàn bộ project một lúc, mà sẽ chia thành frontend, backend, database, config. Sau khi xử lý từng phần, ta ghép kết quả lại để tìm lỗi chính.

## Slide 3 - Mục tiêu

Mục tiêu của chia để trị không chỉ là chia nhỏ cho dễ nhìn, mà quan trọng hơn là giảm độ phức tạp. Nhiều bài toán nếu làm vét cạn sẽ rất chậm, nhưng nếu biết chia đúng cách thì có thể giảm từ bậc hai xuống `n log n`.

## Slide 4 - Mẫu chung

Một thuật toán chia để trị thường có ba bước: Divide, Conquer, Combine. Trong đó, bước Combine thường là bước thể hiện độ tinh tế của thuật toán. Nếu combine không tốt, thuật toán có thể vẫn chậm.

## Slide 5 đến 7 - Quick Sort

Quick Sort chọn một pivot, chia mảng thành nhóm nhỏ hơn pivot và lớn hơn pivot. Với trường hợp chia cân bằng, ta có công thức `T(n) = 2T(n/2) + O(n)`, nên độ phức tạp trung bình là `O(n log n)`. Tuy nhiên nếu pivot quá xấu, ví dụ luôn là phần tử nhỏ nhất hoặc lớn nhất, thuật toán có thể rơi vào `O(n^2)`.

## Slide 8 đến 10 - Karatsuba

Karatsuba là thuật toán nhân số lớn. Bình thường khi nhân hai số được tách đôi, ta cần bốn phép nhân lớn. Karatsuba giảm còn ba phép nhân lớn bằng cách tính thêm `(a+b)(c+d)`. Vì vậy công thức truy hồi là `T(n)=3T(n/2)+O(n)`, suy ra `O(n^1.585)`, nhanh hơn cách nhân thường `O(n^2)` khi số rất lớn.

## Slide 11 - Bài toán chính

Bài toán chính của nhóm em là Closest Pair of Points. Đề bài: cho `n` điểm trên mặt phẳng, hãy tìm hai điểm có khoảng cách Euclid nhỏ nhất. Nếu làm vét cạn, ta phải so sánh mọi cặp điểm, tức là khoảng `n(n-1)/2` cặp, độ phức tạp là `O(n^2)`.

## Slide 12 đến 14 - Ý tưởng thuật toán

Thuật toán chia để trị làm như sau. Đầu tiên sắp xếp các điểm theo trục `x`. Sau đó chia tập điểm thành hai nửa trái và phải. Ta tìm cặp gần nhất ở bên trái, tìm cặp gần nhất ở bên phải. Gọi khoảng cách tốt nhất hiện tại là `d`.

Nhưng có một trường hợp đặc biệt: hai điểm gần nhất có thể nằm ở hai bên đường chia. Vì vậy ta cần kiểm tra thêm một vùng gần đường chia, gọi là strip. Trong strip, ta sắp xếp theo trục `y` và chỉ cần so sánh mỗi điểm với một số điểm kế tiếp gần nó.

## Slide 15 - T(n)

Nếu cài đặt tốt, mỗi lần chia tạo ra hai bài toán con kích thước `n/2`, và bước gộp chạy tuyến tính `O(n)`. Do đó:

```text
T(n) = 2T(n/2) + O(n)
```

Suy ra theo Master Theorem là `O(n log n)`. Đây là cải tiến lớn so với vét cạn `O(n^2)`.

## Slide 16 - Demo

Với ví dụ gồm sáu điểm, ta sort theo `x`, chia thành ba điểm trái và ba điểm phải. Ở nửa trái có hai điểm `(2,3)` và `(3,4)` rất gần nhau, khoảng cách là `sqrt(2)`. Sau đó ta kiểm tra vùng strip để đảm bảo không bỏ sót cặp nằm qua hai bên đường chia. Kết quả cuối cùng vẫn là cặp `(2,3)` và `(3,4)`.

## Slide 17 - Đánh giá

Ưu điểm của thuật toán là rất nhanh khi số điểm lớn, thể hiện rõ tư duy chia để trị, và có nhiều ứng dụng trong hình học tính toán. Nhược điểm là cài đặt khó hơn các thuật toán chia để trị cơ bản, đặc biệt dễ sai ở bước strip.

## Slide 18 - Ứng dụng

Trong thực tế, tư duy chia để trị xuất hiện rất nhiều: sắp xếp dữ liệu lớn, nhân số lớn, xử lý ảnh, tìm kiếm trong bản đồ, định vị robot, game hoặc hệ thống cần tìm các đối tượng gần nhau.

## Slide 19 - Kết luận

Tóm lại, chia để trị là chiến lược mạnh vì nó biến bài toán lớn thành bài toán nhỏ và tận dụng kết quả con. Với Closest Pair, ta thấy rõ sức mạnh này: từ cách vét cạn `O(n^2)`, thuật toán tối ưu có thể đạt `O(n log n)`. Đây là lý do nhóm em chọn bài toán này làm ví dụ chính.
