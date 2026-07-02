# Multi-Task Learning trên dữ liệu Diamonds — Linear + Logistic + Softmax Regression

**Môn học:** Học máy nâng cao (ml_ad) · **Học viên:** Nguyễn Phương Anh Tú · **MSHV:** 2611328
**Dữ liệu:** [Diamonds](https://ggplot2.tidyverse.org/reference/diamonds.html) (thư viện `ggplot2`/`seaborn`, thực tế) — 53 940 viên kim cương, 10 cột, không thiếu dữ liệu (sau khi loại 20 dòng có kích thước bằng 0).

> **Phương pháp luận của môn (bắt buộc với mọi thuật toán).** Bốn giai đoạn cho từng mô hình:
> 1. **Giải tay** — dẫn xuất công thức + thay số cụ thể trên một mẫu thật, kiểm chứng gradient bằng sai phân hữu hạn.
> 2. **Bản thuần** — NumPy thuần, không thư viện học máy.
> 3. **Bản thư viện** — `scikit-learn`.
> 4. **Bản framework** — `PyTorch`, bước cuối cùng.
>
> Notebook đi kèm (4 file, cùng thư mục `2611328 - Nguyễn Phương Anh Tú - Multi-Task Learning (Diamonds)/`):
> - `Linear_Regression_Diamonds.ipynb` — dự đoán `price` (nhãn **liên tục**)
> - `Logistic_Regression_Diamonds.ipynb` — phân loại nhị phân "cắt Ideal hay không" (nhãn **rời rạc**, nhị phân hoá từ `cut`)
> - `Softmax_Regression_Diamonds.ipynb` — phân loại đầy đủ 5 mức `cut` (nhãn **rời rạc**, đa lớp)
> - `MultiTask_Diamonds.ipynb` — **tổng hợp**: một mạng nơ-ron với thân chia sẻ + ba đầu ra, huấn luyện đồng thời cả ba nhiệm vụ trên

---

## Mục lục
1. Vì sao chọn Diamonds — đúng yêu cầu đề bài
2. Tám đặc trưng dùng chung và vấn đề đa cộng tuyến
3. Ba mô hình cơ sở: công thức và kết quả
4. Multi-task learning: kiến trúc và hàm mất mát kết hợp
5. Kết quả thực nghiệm tổng hợp
6. Đa dạng cách áp dụng mô hình đã huấn luyện
7. Hạn chế và khi nào nên/không nên multi-task
8. Tài liệu tham khảo

---

## 1. Vì sao chọn Diamonds

Đề bài yêu cầu một tập dữ liệu **nhiều đặc trưng**, có **đúng hai cột nhãn** — một **liên tục**, một **rời rạc** — để vừa hồi quy vừa phân loại trên cùng dữ liệu. Diamonds thoả cả ba điều kiện bằng dữ liệu **thực tế** (không tổng hợp):

| Cột | Vai trò | Kiểu |
|---|---|---|
| `price` (326 – 18 823 USD) | Nhãn hồi quy | **Liên tục** |
| `cut` (Fair/Good/Very Good/Premium/Ideal) | Nhãn phân loại | **Rời rạc, 5 mức** |
| `carat, depth, table, x, y, z, color, clarity` | Đặc trưng đầu vào | 8 cột |

Cột `cut` được dùng theo **hai cách khác nhau** cho hai mô hình phân loại — đây là điểm khiến "chỉ 2 cột nhãn" vẫn đủ cho cả ba mô hình:
- **Softmax Regression**: giữ nguyên 5 lớp.
- **Logistic Regression**: nhị phân hoá `y = 1` nếu `cut == "Ideal"` (≈ 40% dữ liệu), `y = 0` nếu ngược lại.

## 2. Tám đặc trưng dùng chung và vấn đề đa cộng tuyến

Bộ đặc trưng dùng **giống hệt** ở cả 4 notebook (bắt buộc để so sánh công bằng và để notebook multi-task ghép được ba đầu ra):

```
FEAT_COLS = [carat, depth, table, x, y, z, color_ord, clarity_ord]
```

`color`/`clarity` vốn là hạng thứ bậc (ví dụ màu D tốt nhất → J tệ nhất) nên được mã hoá bằng số nguyên thứ tự, không one-hot — giữ đúng ngữ nghĩa "tốt hơn/kém hơn". Sau khi chuẩn hoá (z-score trên tập train), ma trận đặc trưng có **condition number ≈ 624** — không quá nghiêm trọng nhưng đáng chú ý, vì `carat` tương quan 0,96–0,98 với `x`, `y`, `z` (kích thước vật lý của viên kim cương gần như xác định bởi khối lượng). Hệ quả quan sát được ở cả 3 notebook mô hình đơn:

- **Linear Regression**: cần đến **30 000 epoch** (thay vì vài trăm như ở `b4`/`b7`) để gradient descent thuần hội tụ khớp chính xác với nghiệm OLS của `sklearn` (chênh lệch trọng số → 0 đúng bằng 0 ở 30 000 epoch).
- **Logistic Regression**: độ chính xác/F1 hội tụ nhanh và khớp gần như tuyệt đối giữa các giai đoạn, nhưng **trọng số riêng lẻ thì không** (`max|w_thuần − w_lib| ≈ 1,48`) — vì hướng đa cộng tuyến tạo ra một "sống núi" phẳng trong không gian trọng số: nhiều tổ hợp `w` khác nhau cho cùng một `z = w·x`, nên cùng dự đoán nhưng khác hệ số. Đây là hiện tượng **không định danh được (non-identifiability)** kinh điển của đa cộng tuyến, không phải lỗi cài đặt.

Bài học thực tế: **độ đo hiệu năng hội tụ không có nghĩa là trọng số hội tụ** khi đặc trưng tương quan cao — một điểm dễ bị bỏ qua nếu chỉ nhìn accuracy/RMSE.

## 3. Ba mô hình cơ sở: công thức và kết quả

### 3.1 Linear Regression (dự đoán `price`)

$$\hat{y} = w^\top x + b, \qquad \mathcal{L} = \frac{1}{N}\sum_i(\hat{y}_i - y_i)^2, \qquad \frac{\partial \mathcal{L}}{\partial w} = \frac{2}{N}X^\top(Xw+b-y)$$

| Giai đoạn | RMSE (test) | R² (test) |
|---|---|---|
| Thuần (lr=0,1, 30 000 epoch) | 1253,37 | 0,9013 |
| Thư viện (`LinearRegression`) | 1253,37 | 0,9013 |
| Framework (PyTorch, SGD) | 1253,37 | 0,9013 |

Ba giai đoạn **trùng khớp tuyệt đối** — bằng chứng bản thuần đúng và bài toán lồi được giải tới nghiệm toàn cục. Ridge (`alpha=10`, kiểm chứng bổ sung) cho RMSE 1252,49 — gần như không đổi so với OLS vì cỡ mẫu lớn (43 136 dòng train) áp đảo hiệu ứng chính quy hoá.

### 3.2 Logistic Regression (phân loại nhị phân "Ideal hay không")

Là trường hợp đặc biệt $K=2$ của softmax: $p_1 = e^{z_1}/(e^{z_0}+e^{z_1})$, chia cả tử và mẫu cho $e^{z_0}$ và đặt $z = z_1-z_0$, ta được dạng quen thuộc:

$$p = \sigma(z) = \frac{1}{1+e^{-z}}, \qquad z = w^\top x + b, \qquad \ell = -[y\log p+(1-y)\log(1-p)], \qquad \frac{\partial \ell}{\partial z} = p - y$$

| Giai đoạn | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|
| Thuần (lr=0,5, 2000 epoch) | 0,7941 | 0,7375 | 0,7519 | 0,7237 |
| Thư viện (`LogisticRegression`) | 0,7944 | 0,7376 | 0,7528 | 0,7230 |
| Framework (PyTorch, BCEWithLogits) | 0,7940 | 0,7373 | 0,7517 | 0,7234 |

Gradient giải tích khớp sai phân hữu hạn tới $10^{-11}$. Ba giai đoạn gần như trùng khớp về mọi độ đo — mục tiêu nhị phân này **không** bị ảnh hưởng nặng bởi đa cộng tuyến như mục tiêu `price` (Hessian của log-likelihood bị chặn bởi $0{,}25 X^\top X$, làm dịu hiệu ứng).

### 3.3 Softmax Regression (phân loại 5 mức `cut`)

$$z = W^\top x + b \in \mathbb{R}^5, \qquad p=\mathrm{softmax}(z), \qquad \ell = -\sum_k y_k\log p_k, \qquad \frac{\partial \ell}{\partial z} = p-y$$

| Giai đoạn | Accuracy | F1-macro |
|---|---|---|
| Thuần (lr=0,5, 6000 epoch, ~66s) | 0,6504 | 0,5471 |
| Thư viện (`LogisticRegression`, LBFGS+L2) | 0,6517 | 0,5492 |
| Framework (PyTorch, SGD) | 0,6499 | 0,5456 |

F1 theo từng lớp (bản thuần): Fair 0,608 · Good 0,232 · Very Good 0,430 · Premium 0,675 · Ideal 0,790 — lớp **Good** khó phân biệt nhất vì nằm giữa Fair và Very Good, chồng lấn về mặt vật lý (không có ranh giới rõ giữa các bậc chất lượng cắt liền kề). Đây là bài toán khó hơn hẳn hai bài trên: 5 lớp cần 4 siêu phẳng phân tách độc lập thay vì 1.

## 4. Multi-Task Learning: kiến trúc và hàm mất mát kết hợp

Ba mô hình trên đều là hàm **tuyến tính riêng biệt**. Câu hỏi trọng tâm của `MultiTask_Diamonds.ipynb`: có thể dùng **một** mạng, chia sẻ một biểu diễn ẩn, giải cả ba nhiệm vụ **đồng thời** không?

$$h = \mathrm{ReLU}(W_1^\top x+b_1) \quad\text{(thân chia sẻ)} \qquad \hat y_{\text{price}} = w_r^\top h+b_r,\quad p_{\text{Ideal}}=\sigma(w_b^\top h+b_b),\quad p_{\text{cut}}=\mathrm{softmax}(W_m^\top h+b_m)$$

$$\mathcal{L} = \underbrace{\text{MSE}}_{\text{từ Linear}} + \underbrace{\text{BCE}}_{\text{từ Logistic}} + \underbrace{\text{CE}}_{\text{từ Softmax}}$$

Khi lan truyền ngược, gradient của **cả ba** mất mát cộng dồn vào $\partial \mathcal{L}/\partial h$ trước khi tiếp tục lan vào $W_1, b_1$ — thân chia sẻ được huấn luyện bởi ba tín hiệu giám sát cùng lúc. Bốn giai đoạn triển khai:

1. **Giải tay** — sơ đồ kiến trúc (hộp + mũi tên) vẽ đúng luồng $x \to h \to$ 3 đầu $\to$ 3 mất mát $\to$ tổng.
2. **Thuần** — MLP 1 lớp ẩn (32 nơ-ron), lan truyền ngược viết tay qua cả ba nhánh, SGD có động lượng ($\beta=0{,}9$) vì GD thuần không thích ứng hội tụ rất chậm trên mạng nhiều tham số.
3. **Thư viện** — `scikit-learn` **không có** kiến trúc thân-chia-sẻ dựng sẵn, nên đường mốc thư viện hợp lý nhất là **ba `MLPRegressor`/`MLPClassifier` độc lập** (kiến trúc gần tương đương) — đây đồng thời là đường mốc "không multi-task" dùng để đo lợi ích thực của việc chia sẻ.
4. **Framework** — PyTorch, một `nn.Module`, `loss.backward()` một lần cho tổng ba mất mát.

## 5. Kết quả thực nghiệm tổng hợp

| Cách tiếp cận | Số tham số | RMSE giá ↓ | R² giá ↑ | F1 nhị phân ↑ | F1-macro 5 lớp ↑ |
|---|---|---|---|---|---|
| 3 mô hình tuyến tính riêng (Linear+Logistic+Softmax) | ~63 | 1253,4 | 0,9013 | 0,7376 | 0,5492 |
| Multi-task — bản thuần (NumPy, 1 lớp ẩn) | 519 | 703,0 | 0,9689 | 0,8544 | 0,7119 |
| 3 MLP độc lập — thư viện (không chia sẻ) | 1367 | 743,5 | 0,9653 | 0,8615 | 0,7669 |
| Multi-task — framework (PyTorch, chia sẻ, đồng thời) | 1575 | **644,0** | **0,9739** | **0,8623** | 0,7623 |

**Đọc kết quả trung thực (không tô hồng):**

- Cả ba cách dùng **mạng nơ-ron** vượt xa ba mô hình **tuyến tính** — vì `price` phụ thuộc phi tuyến vào carat (giá tăng nhanh hơn tuyến tính khi carat lớn), điều softmax/linear tuyến tính đơn giản không nắm bắt được.
- So sánh công bằng nhất — multi-task PyTorch so với 3 MLP độc lập (cùng là mạng nơ-ron, khác ở "chia sẻ hay không"): multi-task **thắng rõ** ở hồi quy giá (RMSE 644,0 so với 743,5) nhờ tín hiệu bổ sung từ hai đầu phân loại; ở hai nhiệm vụ phân loại thì **xấp xỉ nhau** (F1 nhị phân 0,8623 so với 0,8615; F1-macro 5 lớp 0,7623 so với 0,7669 — multi-task nhỉnh hơn một bên, kém hơn bên kia trong khoảng nhiễu ngẫu nhiên).
- Kết luận thành thật: multi-task **không thắng tuyệt đối mọi độ đo**. Lợi ích rõ ràng nhất **không phải** "chính xác hơn ở mọi nhiệm vụ" mà là **một mô hình, một lượt suy luận, ba kết quả** — hiệu quả triển khai, không phải phép màu về độ chính xác.

## 6. Đa dạng cách áp dụng mô hình đã huấn luyện

`MultiTask_Diamonds.ipynb` §7 minh hoạ ba cách dùng khác nhau cho **cùng một mô hình đã huấn luyện**, không cần huấn luyện lại:

1. **Suy luận một lượt** — một lần gọi `forward()` trả về đồng thời giá dự đoán, xác suất "Ideal", và hạng cắt — so với việc phải gọi ba mô hình riêng ba lần.
2. **Tìm kiếm tương đồng qua embedding** — dùng riêng lớp ẩn $h$ (không dùng bất kỳ đầu dự đoán nào) để đo khoảng cách Euclid giữa các viên kim cương; 5 viên "gần" một viên truy vấn trong không gian embedding có carat/giá/hạng cắt gần nhau thật — bằng chứng $h$ học được một biểu diễn tổng quát, không chỉ phục vụ một đầu ra.
3. **Điều chỉnh ngưỡng quyết định** — đầu nhị phân trả về xác suất; đổi ngưỡng từ 0,5 lên 0,8 chuyển trọng tâm từ "tối đa F1" sang "tối đa precision" (F1 giảm từ 0,862 xuống 0,752 nhưng precision tăng từ 0,819 lên 0,878) — phù hợp khi báo nhầm "Ideal" tốn kém hơn bỏ sót, **không cần huấn luyện lại mô hình**.

## 7. Hạn chế và khi nào nên/không nên multi-task

- **Cần các nhiệm vụ có liên quan.** Multi-task chỉ có lợi khi các nhiệm vụ chia sẻ cấu trúc tiềm ẩn (ở đây: cùng phụ thuộc "kích thước + chất lượng" của kim cương). Ba nhiệm vụ không liên quan có thể gây **negative transfer** (một nhiệm vụ kéo biểu diễn theo hướng có hại cho nhiệm vụ khác).
- **Cân bằng trọng số mất mát.** Ở đây dùng trọng số bằng nhau (1:1:1) vì thang giá trị ba hàm mất mát tương đồng sau khi chuẩn hoá `price`; với các bài toán khác, có thể cần điều chỉnh trọng số hoặc kỹ thuật cân bằng gradient (GradNorm, uncertainty weighting) để một nhiệm vụ không lấn át hai nhiệm vụ còn lại.
- **Khó gỡ lỗi hơn mô hình đơn.** Khi một đầu ra kém, khó biết lỗi do thân chia sẻ hay do đầu đó — cần huấn luyện thêm phiên bản đơn nhiệm để đối chiếu (chính là vai trò của bản "thư viện, 3 MLP độc lập" ở §5).
- **`scikit-learn` không phục vụ tốt kiến trúc này** — bất kỳ mô hình chia sẻ tham số thực sự nào đều cần một framework học sâu (PyTorch/TensorFlow/JAX).

## 8. Tài liệu tham khảo

- Caruana, R. (1997). *Multitask Learning.* Machine Learning, 28(1), 41–75. — bài báo nền tảng đặt tên và hình thức hoá multi-task learning.
- Ruder, S. (2017). *An Overview of Multi-Task Learning in Deep Neural Networks.* arXiv:1706.05098. https://arxiv.org/abs/1706.05098
- Kendall, A., Gal, Y., Cipolla, R. (2018). *Multi-Task Learning Using Uncertainty to Weigh Losses.* CVPR. — kỹ thuật cân bằng trọng số mất mát đa nhiệm.
- Bishop, C.M. (2006). *Pattern Recognition and Machine Learning.* Chương 3 (hồi quy tuyến tính), Chương 4 (hồi quy logistic/softmax). Springer.
- Goodfellow, I., Bengio, Y., Courville, A. (2016). *Deep Learning.* Mục 6.2 (đơn vị đầu ra), Mục 7.7 (multi-task learning trong mạng nơ-ron). https://www.deeplearningbook.org/
- scikit-learn — `MLPRegressor`/`MLPClassifier`: https://scikit-learn.org/stable/modules/neural_networks_supervised.html
- PyTorch — xây mô hình nhiều đầu ra (multi-head): https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
- Diamonds dataset (nguồn `ggplot2`, phổ biến lại qua `seaborn-data`): https://github.com/mwaskom/seaborn-data/blob/master/diamonds.csv
