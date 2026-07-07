# Tìm hiểu Mô hình Perceptron — báo cáo lý thuyết và mô phỏng huấn luyện bằng tay

**Môn học:** Học máy nâng cao (ml_ad) · **Học viên:** Nguyễn Phương Anh Tú · **MSHV:** 2611328
**Đề bài minh hoạ:** Bảng 2 — bộ dữ liệu mô phỏng mối quan hệ giữa nhiệt độ và hiệu suất hoạt động (4 mẫu: (20,35), (30,50), (40,82), (50,91)).

> **Notebook đi kèm:** `2611328 - Nguyễn Phương Anh Tú - Tìm hiểu Mô hình Perceptron/Perceptron.ipynb` — trả lời đầy đủ 6 câu hỏi của đề, mô phỏng huấn luyện bằng tay trên Bảng 2, rồi triển khai lại theo đúng 4 giai đoạn của môn (giải tay → thuần NumPy → thư viện → framework), có thêm minh hoạ Perceptron cổ điển (phân loại nhị phân) để làm rõ câu hỏi 1.

---

## Đề bài (nguyên văn, 6 ý)

1. Thường được dùng để giải quyết bài toán gì?
2. Vì sao gọi mô hình này là nền tảng của các mô hình học sâu?
3. Trình bày các bước huấn luyện mô hình?
4. Những bước nào tương ứng với giai đoạn Propagation?
5. Những bước nào tương ứng với giai đoạn Backpropagation?
6. Mô phỏng quá trình huấn luyện bằng cách tính toán bằng tay ví dụ ở Bảng 2?

---

## 1. Perceptron thường được dùng để giải quyết bài toán gì?

Perceptron (Rosenblatt, 1958) nguyên bản là một **bộ phân loại tuyến tính nhị phân**: tìm siêu phẳng $w^\top x+b=0$ chia hai lớp có thể phân tách tuyến tính, gán nhãn theo phía của điểm dữ liệu. Ví dụ kinh điển: cổng logic AND, OR (minh hoạ ở §4 dưới).

Trong ngôn ngữ hiện đại, "Perceptron" cũng chỉ **một đơn vị tính toán (neuron)** đơn lẻ: $\hat{y}=f(w^\top x+b)$. Tuỳ hàm kích hoạt $f$: đồng nhất → **hồi quy** (giá trị liên tục), bước nhảy → **phân loại nhị phân** (đúng nghĩa gốc), sigmoid → **phân loại nhị phân có xác suất** (hồi quy logistic một neuron). Vì vậy Perceptron/neuron giải quyết được **cả hồi quy tuyến tính và phân loại tuyến tính**, tuỳ hàm kích hoạt đầu ra.

## 2. Vì sao gọi mô hình này là nền tảng của các mô hình học sâu?

1. **Đơn vị tính toán nhỏ nhất.** Perceptron/neuron = tổng có trọng số + hàm kích hoạt. Mọi mạng sâu (MLP, CNN, RNN, Transformer) là nhiều neuron xếp lớp, ghép nối bằng trọng số — không có cấu trúc "lớp" nào phức tạp hơn một tập neuron song song.
2. **Nguyên lý forward/backward đã đủ ở quy mô 1 neuron.** Tính đầu ra từ đầu vào (Propagation) và cập nhật trọng số theo sai số đầu ra (Backpropagation) — hai trụ cột huấn luyện mọi mạng sâu — đã có mặt trọn vẹn trong quy tắc học Perceptron. Khi xếp nhiều lớp, backpropagation chỉ là áp dụng lặp lại quy tắc chuỗi (chain rule) qua từng lớp.
3. **Ý nghĩa lịch sử.** Perceptron là mô hình học từ dữ liệu đầu tiên được chứng minh hội tụ (Novikoff, 1962) cho dữ liệu phân tách tuyến tính — tiền thân trực tiếp của MLP (Rumelhart et al., 1986) và học sâu hiện đại.

## 3. Các bước huấn luyện mô hình

| Bước | Nội dung |
|---|---|
| 0. Khởi tạo | $w,b$ (thường $=0$ hoặc số ngẫu nhiên nhỏ) |
| 1. Forward | $z=w^\top x+b$, $\hat{y}=f(z)$ |
| 2. Tính mất mát | $\mathcal{L}(\hat{y},y)$ |
| 3. Backward | $\partial\mathcal{L}/\partial w$, $\partial\mathcal{L}/\partial b$ |
| 4. Cập nhật | $w\leftarrow w-\eta\,\partial\mathcal{L}/\partial w$, $b\leftarrow b-\eta\,\partial\mathcal{L}/\partial b$ |
| 5. Lặp | Quay lại Bước 1 đến khi hội tụ/hết epoch |

Với Perceptron cổ điển (activation bước nhảy, không khả vi): $w\leftarrow w+\eta(y-\hat{y})x$, chỉ cập nhật khi dự đoán sai — không tổng quát hoá qua gradient.

## 4. Bước nào là Propagation? Bước nào là Backpropagation?

- **Propagation = Bước 1 (Forward)**: lan truyền $x$ xuôi chiều qua $w,b,f$ để ra $\hat{y}$ — một chiều duy nhất, không có bước tính ngược.
- **Backpropagation = Bước 2, 3, 4**: tính sai số, lan truyền ngược để suy ra đạo hàm riêng của mất mát theo từng tham số, rồi cập nhật theo hướng giảm mất mát. Với **một neuron duy nhất**, backprop suy biến thành đạo hàm trực tiếp — không cần chain rule qua nhiều lớp; đây là khác biệt duy nhất so với MLP nhiều lớp ẩn.

## 5. Mô phỏng huấn luyện bằng tay trên Bảng 2

Mô hình: neuron tuyến tính $\hat{y}=w\cdot x+b$ (activation đồng nhất, vì Performance liên tục), mất mát MSE trên 4 mẫu:

$$\mathcal{L}=\frac{1}{4}\sum_{i=1}^4(\hat{y}_i-y_i)^2,\qquad \frac{\partial\mathcal{L}}{\partial w}=\frac{2}{4}\sum(\hat{y}_i-y_i)x_i,\qquad \frac{\partial\mathcal{L}}{\partial b}=\frac{2}{4}\sum(\hat{y}_i-y_i)$$

Khởi tạo $w_0=b_0=0$, $\eta=0{,}0005$ (thang giá trị Temperature gốc 20–50 chưa chuẩn hoá, cần $\eta$ nhỏ để không phân kỳ — thử $\eta=0{,}01$ sẽ nổ số ngay epoch 2, notebook có minh chứng cụ thể):

| Epoch | $w$ | $b$ | $\hat{y}$ | Mất mát | $\partial w$ | $\partial b$ | $w$ mới | $b$ mới |
|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | [0, 0, 0, 0] | 4682,5 | −5015,0 | −129,0 | 2,5075 | 0,0645 |
| 1 | 2,5075 | 0,0645 | [50,21; 75,29; 100,36; 125,44] | 598,59 | 1759,77 | 46,65 | 1,6276 | 0,0412 |
| 2 | 1,6276 | 0,0412 | [32,59; 48,87; 65,15; 81,42] | 95,72 | −617,55 | −14,98 | 1,9364 | 0,0487 |

Mất mát giảm $4682{,}5\to598{,}6\to95{,}7$ — đúng hướng, nhưng để hội tụ hẳn về nghiệm bình phương tối thiểu $w^\*=2{,}0,\ b^\*=-5{,}5$ (MSE$^\*=22{,}25$) cần rất nhiều epoch hơn nữa do thang giá trị Temperature chưa chuẩn hoá — xác nhận bằng chạy đầy đủ trong notebook (§5–7): chuẩn hoá (z-score) rồi huấn luyện 2000 epoch, cả 3 giai đoạn thuần/thư viện/framework đều khớp đúng $w=2{,}0000,\ b=-5{,}5000$. Gradient giải tích kiểm chứng khớp tuyệt đối với sai phân hữu hạn (vì mất mát là đa thức bậc 2, đạo hàm số học chính xác).

## 6. Minh hoạ bổ sung — Perceptron cổ điển cho phân loại nhị phân

Vì Bảng 2 chỉ có nhãn liên tục, notebook bổ sung ví dụ cổng **AND** ($X=\{(0,0),(0,1),(1,0),(1,1)\}$, $Y=\{0,0,0,1\}$) dùng đúng activation bước nhảy và quy tắc học Perceptron gốc. Khởi tạo $w_1=w_2=b=0$, $\eta=1$: sau **5 epoch đầy đủ** (20 lượt xem mẫu), hội tụ về $w_1=2,\ w_2=1,\ b=-3$ — không còn mẫu nào sai (định lý hội tụ Novikoff áp dụng đúng vì AND phân tách tuyến tính được). Đây là ví dụ trực tiếp trả lời câu hỏi 1 ở khía cạnh phân loại, đối lập với neuron hồi quy dùng cho Bảng 2.

## 7. Kết quả tổng hợp 4 giai đoạn (Bảng 2)

| Giai đoạn | $w$ | $b$ | MSE |
|---|---|---|---|
| Nghiệm OLS (đích) | 2,0000 | −5,5000 | 22,2500 |
| Giải tay (3 epoch, minh hoạ) | 1,9364 | 0,0487 | 33,79 |
| Bản thuần (NumPy, 2000 epoch, chuẩn hoá) | 2,0000 | −5,5000 | 22,2500 |
| Bản thư viện (`SGDRegressor`) | 1,9870 | −5,1819 | 22,2904 |
| Bản thư viện (`LinearRegression`, closed-form) | 2,0000 | −5,5000 | 22,2500 |
| Bản framework (PyTorch `nn.Linear`) | 2,0000 | −5,5000 | 22,2500 |

Cả 4 giai đoạn hội tụ về cùng nghiệm — bài toán lồi, dữ liệu sạch, không có khoảng cách đáng kể giữa thuần/thư viện/framework. Điểm mấu chốt sư phạm nằm ở giải tay: chỉ vài epoch trên thang giá trị gốc chưa đủ để hội tụ; chuẩn hoá đầu vào là bước bắt buộc để gradient descent thực tế hiệu quả — cùng bài học đã rút ra ở `ml_ad/b4` và `ml_ad/b8`.

## Tài liệu tham khảo

- Rosenblatt, F. (1958). *The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain.* Psychological Review, 65(6), 386–408.
- Novikoff, A.B.J. (1962). *On Convergence Proofs on Perceptrons.* Symposium on the Mathematical Theory of Automata.
- Minsky, M., Papert, S. (1969). *Perceptrons: An Introduction to Computational Geometry.* MIT Press.
- Rumelhart, D.E., Hinton, G.E., Williams, R.J. (1986). *Learning representations by back-propagating errors.* Nature, 323, 533–536.
- Goodfellow, I., Bengio, Y., Courville, A. (2016). *Deep Learning.* Chương 6. https://www.deeplearningbook.org/
- Bishop, C.M. (2006). *Pattern Recognition and Machine Learning.* Chương 4. Springer.
- scikit-learn — `SGDRegressor`: https://scikit-learn.org/stable/modules/sgd.html
- PyTorch — `nn.Linear`: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
