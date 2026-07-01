# Softmax Regression — báo cáo nghiên cứu và phương pháp triển khai

**Môn học:** Học máy nâng cao (ml_ad) · **Học viên:** Nguyễn Phương Anh Tú · **MSHV:** 2611328
**Dữ liệu minh hoạ:** Iris (3 loài hoa, 4 đặc trưng số, 150 mẫu) — tập dữ liệu kinh điển cho bài toán phân loại đa lớp.

> **Phương pháp luận của môn (bắt buộc với mọi thuật toán).** Triển khai theo đúng 4 giai đoạn, có lý do cho từng bước:
> 1. **Giải tay** — dẫn xuất công thức toán từng bước trước khi viết mã.
> 2. **Bản thuần, không thư viện** — cài đặt trực tiếp từ công thức bằng NumPy; kiểm chứng gradient bằng sai phân hữu hạn.
> 3. **Bản thư viện** — `scikit-learn` (`LogisticRegression`), dùng solver công nghiệp (LBFGS).
> 4. **Bản framework** — `PyTorch` (`nn.Linear` + `CrossEntropyLoss`), bước cuối cùng.
>
> Notebook đi kèm (3 file, cùng thư mục `2611328 - Nguyễn Phương Anh Tú - Hồi quy Softmax & Tuyến tính/`): `Softmax_Regression_Iris.ipynb` (softmax, minh hoạ chính), `Softmax_ViHSD.ipynb` (softmax trên data b1 — hate speech), `Linear_Regression_Hanoi.ipynb` (hồi quy tuyến tính trên data b1 — giá nhà Hà Nội). Cả ba đều theo trình tự: giải tay (thay số) → bản thuần NumPy → bản lib → so sánh.

---

## Mục lục
1. Định vị: softmax regression là gì và không phải là gì
2. Mô hình toán
3. Hàm mất mát: entropy chéo
4. Dẫn xuất gradient (giải tay)
5. Các quyết định triển khai then chốt
6. Kết quả thực nghiệm trên Iris (4 giai đoạn)
7. So sánh với các phương pháp phân loại đa lớp khác
8. Hạn chế và khi nào không nên dùng
9. Lịch sử và nguồn
10. Tài liệu tham khảo

---

## 1. Định vị

Softmax regression (còn gọi là **multinomial logistic regression**) là mô hình tuyến tính tổng quát hoá hồi quy logistic từ hai lớp sang **nhiều lớp**. Cho một mẫu đặc trưng $x \in \mathbb{R}^d$ và $K$ lớp, mô hình ước lượng xác suất mẫu thuộc mỗi lớp. Nó là một **bộ phân loại tuyến tính** (ranh giới quyết định là các siêu phẳng) và đồng thời là **thành phần đầu ra tiêu chuẩn** của mọi mạng nơ-ron phân loại (lớp softmax).

Điểm cần làm rõ: softmax regression **không** "học biểu diễn" như mạng nơ-ron sâu — nó chỉ là một phép biến đổi tuyến tính cộng softmax. Sức mạnh đến từ đặc trưng đầu vào; nếu đặc trưng đã tách lớp (như Iris), mô hình đơn giản này đã đủ đạt độ chính xác cao.

## 2. Mô hình toán

Với ma trận trọng số $W \in \mathbb{R}^{d \times K}$ và thiên lệch $b \in \mathbb{R}^K$, "điểm số" (logit) của mẫu $x$ ở mỗi lớp là:
$$z = W^\top x + b \in \mathbb{R}^K.$$

Để biến $K$ điểm số (có âm, không bị chặn) thành $K$ xác suất (dương, tổng bằng 1), ta áp dụng **hàm softmax**:
$$\hat{y}_k = \text{softmax}(z)_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}.$$
Mẫu mẫu $e^{z_k}$ tự nhiên cho giá trị dương; việc chia cho tổng chuẩn hoá thành phân phối xác suất. Hàm softmax **bất biến cộng** theo hằng số (cộng cùng một giá trị vào mọi $z_k$ không đổi kết quả) — đây là cơ sở của thủ tục ổn định số (Mục 5).

Dự đoán là lớp có xác suất cao nhất: $\hat{c} = \arg\max_k \hat{y}_k$.

## 3. Hàm mất mát: entropy chéo

Với nhãn thật mã hoá one-hot $y \in \{0,1\}^K$ (đúng lớp $c$ thì $y_c=1$, các vị trí khác 0), hàm mất mát của **một** mẫu là **entropy chéo phân loại**:
$$\ell = -\sum_{k=1}^{K} y_k \log \hat{y}_k = -\log \hat{y}_c,$$
dạng rút gọn cuối xảy ra vì chỉ một $y_c$ bằng 1. Trên toàn bộ tập $N$ mẫu, mất mát trung bình:
$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\log \hat{y}_{i,c_i}.$$
Ý nghĩa: mô hình bị phạt nặng khi gán xác suất thấp cho lớp đúng; khi $\hat{y}_{c} \to 1$ thì $\ell \to 0$.

## 4. Dẫn xuất gradient (giải tay)

Đây là phần cốt lõi — gradient được dẫn xuất tường minh để mã hoá trực tiếp vào bản thuần.

**Bước 1. Chuỗi theo logit.** Với $\ell = -\sum_k y_k \log \hat{y}_k$ và $\hat{y}_k = e^{z_k}/\sum_j e^{z_j}$, đạo hàm riêng theo $z_m$ là:
$$\frac{\partial \ell}{\partial z_m} = \hat{y}_m - y_m.$$
Chứng minh ngắn: $\partial \log \hat{y}_k / \partial z_m = \mathbb{1}[k=m] - \hat{y}_m$, thay vào và dùng $\sum_k y_k = 1$.

**Bước 2. Theo trọng số và thiên lệch.** Vì $z_m = W_{\cdot m}^\top x + b_m$ nên:
$$\frac{\partial \ell}{\partial W_{jm}} = x_j(\hat{y}_m - y_m), \qquad \frac{\partial \ell}{\partial b_m} = \hat{y}_m - y_m.$$

**Bước 3. Trên toàn batch.** Gộp $N$ mẫu, đặt $P \in \mathbb{R}^{N\times K}$ là các xác suất dự đoán và $Y \in \{0,1\}^{N\times K}$ là nhãn one-hot:
$$\frac{\partial \mathcal{L}}{\partial W} = \frac{1}{N} X^\top (P - Y), \qquad \frac{\partial \mathcal{L}}{\partial b} = \frac{1}{N}\mathbf{1}^\top (P - Y).$$

**Tính đẹp của kết hợp softmax + entropy chéo:** gradient của logit chỉ là $(P-Y)$ — sai số dự đoán trừ nhãn. Nhờ đó bản thuần chỉ cần hai phép ma trận; không cần đạo hàm bậc hai hay thư viện tự động vi phân. Notebook kiểm chứng gradient này bằng **sai phân hữu hạn** (độ lệch so với giải tích $\approx 10^{-10}$).

## 5. Các quyết định triển khai then chốt

| Quyết định | Lý do |
|---|---|
| **Trừ max trước exp** (log-sum-exp trick) | $e^{z}$ tràn (overflow) khi $z$ lớn; vì softmax bất biến cộng, ta tính $e^{z - \max z}$ — kết quả đúng mà không tràn. Bắt buộc khi cài bằng tay. |
| **Mã hoá one-hot nhãn** | để công thức gradient $(P-Y)/N$ gọn và ma trận hoá được. |
| **Chuẩn hoá đặc trưng** (trừ trung bình, chia độ lệch) | softmax + GD nhạy với tỉ lệ đặc trưng; Iris có "đài hoa" ≈ cm và "cánh hoa" ≈ cm nhưng nếu không chuẩn hoá, GD chậm hoặc dao động. |
| **Chính quy hoá L2** | khi dữ liệu tách lớp (như Iris), trọng số có xu hướng **phân kỳ** (logits ngày càng lớn để chắc chắn); L2 giữ trọng số nhỏ, chống overfit. `sklearn` mặc định có L2 ($C=1$); bản thuần cần tự thêm khi muốn so sánh công bằng. |
| **Chọn bộ tối ưu** | bản thuần dùng **gradient descent nguyên bản** (minibatch/full-batch) để minh hoạ nguyên lý; `sklearn`/`torch` dùng **LBFGS** (giải Newton + xấp xỉ Hessian) — hội tụ sâu hơn trên bài toán lồi này. |

## 6. Kết quả thực nghiệm trên Iris (4 giai đoạn)

Cùng một phân chia (70/30, `random_state=1`, phân tầng), cùng chuẩn hoá:

| Giai đoạn | Độ chính xác train | Độ chính xác test | Hàm mất mát cuối | Ghi chú |
|---|---|---|---|---|
| Giải tay (1 mẫu kiểm chứng) | — | — | $\ell = 2{,}464$ | gradient giải tích khớp sai phân hữu hạn tới $1{,}1\times10^{-10}$ |
| Thuần NumPy (GD, lr 0.5, 3000 epoch) | 0.981 | **1.000** | 0.059 | trọng số nhỏ, chưa đẩy loss cực thấp do lr cố định |
| Thư viện sklearn (LBFGS, $C\!=\!10^6$) | 0.981 | **1.000** | thấp hơn | solver bậc hai → trọng số lớn hơn |
| Framework PyTorch (LBFGS) | 0.981 | **1.000** | thấp hơn | trùng khớt với sklearn về độ chính xác |

**Đọc kết quả.** Bốn cách triển khai cho **cùng độ chính xác** trên Iris — minh chứng trực tiếp rằng bản thuần được cài **đúng** (nếu sai, kết quả đã lệch). Trọng số các bản LBFGS lớn hơn vì chúng đẩy mất mát thấp hơn trên bài toán gần tách lớp; dấu các hệ số **giống nhau ở mọi giai đoạn** (ví dụ hệ số "cánh hoa" âm với lớp setosa) — xác nhận cùng một nghiệm định tính. Điểm lỗi train duy nhất là một mẫu versicolor/virginica nằm chồng ranh giới — mọi phương án đều hỏi như nhau.

## 7. So sánh với các phương pháp phân loại đa lớp khác

- **One-vs-Rest (OvR):** chia bài toán $K$ lớp thành $K$ hồi quy logistic nhị phân. Đơn giản nhưng các lớp không cạnh tranh trực tiếp; xác suất không chuẩn hoá thật sự. Softmax cạnh tranh đồng thời nên thường ưu việm hơn khi các lớp liên quan.
- **One-vs-One:** huấn luyện $K(K-1)/2$ bộ phân loại nhị phân. Tốn khi $K$ lớn; hiếm dùng cho softmax.
- **Mạng nơ-ron sâu:** softmax chỉ là **lớp đầu ra**; mạng sâu thêm các lớp ẩn để học biểu diễn phi tuyến. Softmax regression = mạng nơ-ron không lớp ẩn.
- **Hồi quy logistic nhị phân + ngưỡng:** chỉ dùng được cho 2 lớp; softmax là tổng quát hoá tự nhiên.

## 8. Hạn chế và khi nào không nên dùng

- **Chỉ tuyến tính:** ranh giới quyết định là siêu phẳng; với dữ liệu phi tuyến (ví dụ "hai vành nhẫn đồng tâm") softmax regression cơ bản sẽ thất bại trừ khi thêm đặc trưng đa thức hoặc kernel.
- **Nhạy với đặc trưng:** cần chuẩn hoá và kỹ nghệ đặc trưng; không "tự học" được.
- **Trọng số phân kỳ trên dữ liệu tách lớp hoàn toàn:** cần L2 hoặc early stopping.
- **Không xử lý được thứ bậc** giữa các lớp (lớp có thứ tự nên dùng ordinal regression).

## 9. Lịch sử và nguồn

Hồi quy logistic nhị phân có từ Cox (1958). Dạng **multinomial** với hàm softmax được Bridle (1990) phổ biến trong mạng nơ-ron ("probabilistic output layer"). Tên "softmax" do John S. Bridle đặt, để phân biệt với "hardmax" (chọn argmax cứng). Hàm mất mát entropy chéo có gốc trong lý thuyết thông tin (Kullback–Leibler). Trở thành lớp đầu ra mặc định của phân loại sâu từ thập niên 2010 (AlexNet 2012, và mọi classifier hiện đại).

## 10. Tài liệu tham khảo

- Bridle, J.S. (1990). *Probabilistic Interpretation of Feedforward Classification Network Outputs, with Relationships to Statistical Pattern Recognition.* Neurocomputing. — nguồn hàm softmax trong mạng nơ-ron.
- Bishop, C.M. (2006). *Pattern Recognition and Machine Learning.* Chương 4 (Logistic Regression). Springer.
- Goodfellow, I., Bengio, Y., Courville, A. (2016). *Deep Learning.* Mục 6.2.2 (Softmax). MIT Press. https://www.deeplearningbook.org/contents/mlp.html
- Hastie, T., Tibshirani, R., Friedman, J. (2009). *The Elements of Statistical Learning.* Chương 4. Springer. https://hastie.su.domains/ElemStatLearn/
- Cox, D.R. (1958). *The Regression Analysis of Binary Sequences.* JRSS-B. — gốc hồi quy logistic.
- Böhning, D. (1992). *Multinomial logistic regression algorithm.* Annals of the Institute of Statistical Mathematics.
- scikit-learn — `LogisticRegression`: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
- PyTorch — `CrossEntropyLoss` (gộp log-softmax + NLL): https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
- Iris dataset (Anderson, 1935; dùng bởi Fisher, 1936): https://archive.ics.uci.edu/dataset/53/iris
