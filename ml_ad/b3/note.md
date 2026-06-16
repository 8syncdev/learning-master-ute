# Buổi b3 — Note

## Q1. Vì sao train dùng `fit_transform`, test dùng `transform`?

**Hỏi:** Tại sao tập **train** (sau khi chuẩn hoá + cân bằng) dùng `fit_transform`, còn tập **test** chỉ dùng `transform`?

**Trả lời ngắn:** `fit` = **học tham số** từ dữ liệu. `transform` = **áp dụng** tham số đã học. Chỉ được phép học tham số từ **train**. Test phải dùng lại đúng tham số đó để mô phỏng dữ liệu "chưa từng thấy" → tránh **data leakage** (rò rỉ dữ liệu).

**Chi tiết:**

### 1. `fit` và `transform` làm gì
Với `StandardScaler`:
- `fit` → học `mean` và `std` của từng feature từ dữ liệu đưa vào.
- `transform` → tính `(x - mean) / std` bằng `mean`/`std` **đã học**.
- `fit_transform` → gộp 2 bước: học rồi biến đổi luôn (chỉ là tối ưu, không phải phép thuật).

(MinMaxScaler tương tự nhưng học `min`/`max`.)

### 2. Vì sao train dùng `fit_transform`
Scaler cần một nguồn để học thống kê. Nguồn hợp lệ duy nhất là **train** — đây là toàn bộ "kiến thức" model được phép biết. `fit_transform` học `mean`/`std` từ train rồi chuẩn hoá train ngay.

### 3. Vì sao test chỉ dùng `transform`
- **Tránh data leakage**: nếu `fit` trên test, thống kê (mean/std) của test lọt vào pipeline. Lúc đó điểm đánh giá sẽ **lạc quan ảo** — model gián tiếp "nhìn trộm" test, không phản ánh hiệu năng thật khi gặp dữ liệu mới.
- **Cùng một thang đo**: train và test phải được biến đổi bằng **cùng bộ tham số**. Nếu mỗi tập `fit` riêng, chúng nằm trên 2 thang đo khác nhau → feature không nhất quán, model nhận sai phân phối.
- **Mô phỏng production**: khi deploy, mỗi mẫu mới đến được `transform` bằng tham số **cố định** đã học từ train. Không thể (và không nên) `fit` lại trên từng mẫu mới. `transform` ở test chính là tập dượt đúng tình huống đó.

### 4. Phần "cân bằng" (balancing)
Cân bằng lớp (SMOTE / oversampling / undersampling) cũng **chỉ áp dụng cho train**, **không** áp dụng cho test:
- Test phải giữ **phân phối gốc thực tế** thì chỉ số (accuracy, F1, recall theo lớp...) mới trung thực. Cân bằng test = bóp méo thực tế cần đo.
- Cân bằng test còn là một dạng leakage: sinh mẫu tổng hợp trên dữ liệu đáng lẽ phải "chưa thấy".

### 5. Thứ tự đúng
```
1. train_test_split            # tách trước tiên, test "đóng băng"
2. scaler.fit_transform(X_train)   # học mean/std TRÊN train + chuẩn hoá train
   scaler.transform(X_test)        # dùng LẠI mean/std của train cho test
3. balance CHỈ trên (X_train, y_train)   # SMOTE/oversampling — không đụng test
4. train model → evaluate trên test
```

> Quy tắc vàng: **mọi thứ học tham số (`fit`) đều chỉ được nhìn train.** Test chỉ đi qua `transform`, không bao giờ `fit`.

**Ví dụ:**
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)   # học + biến đổi trên train
X_test  = scaler.transform(X_test)        # CHỈ biến đổi, dùng tham số của train

# Cân bằng CHỈ trên train
X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)
```

Nếu vô tình viết `scaler.fit_transform(X_test)` → test bị `fit` lại bằng mean/std của chính nó → **data leakage**, kết quả đánh giá không còn đáng tin.

## Q2. Phân biệt underfitting và overfitting: nguyên nhân, so sánh, cách khắc phục

**Hỏi:** Phân biệt underfitting và overfitting — nguyên nhân, so sánh chi tiết, cách khắc phục.

**Trả lời ngắn:** **Underfitting** = model quá đơn giản, học chưa đủ → sai **cả train lẫn test** (high **bias**). **Overfitting** = model quá phức tạp, học luôn cả nhiễu của train → train rất tốt nhưng **test tệ** (high **variance**). Mục tiêu là điểm cân bằng **bias–variance** nơi test error thấp nhất.

**Chi tiết:**

### 1. Underfitting (dưới khớp) — high bias
Model không đủ sức nắm pattern trong dữ liệu.
- **Dấu hiệu:** train error **cao**, test error **cao**, hai cái xấp xỉ nhau (đều tệ).
- **Nguyên nhân:**
  - Model quá đơn giản so với dữ liệu (linear cho quan hệ phi tuyến, polynomial bậc thấp, cây quá nông, mạng quá ít layer/neuron).
  - Thiếu feature, hoặc feature ít thông tin.
  - **Regularization quá mạnh** (λ lớn ép hệ số về 0).
  - Train chưa đủ (quá ít epoch / dừng quá sớm).

### 2. Overfitting (quá khớp) — high variance
Model thuộc lòng cả nhiễu và chi tiết riêng của tập train.
- **Dấu hiệu:** train error **rất thấp** nhưng test/validation error **cao** → **gap** train–test lớn (generalization gap).
- **Nguyên nhân:**
  - Model quá phức tạp so với lượng dữ liệu (quá nhiều tham số, cây quá sâu, polynomial bậc cao, mạng quá lớn).
  - **Dữ liệu train ít** hoặc không đại diện; nhiều nhiễu, nhiều feature dư thừa.
  - Train **quá lâu** (quá nhiều epoch).
  - Data leakage làm model "thấy trộm" thông tin (xem Q1).

### 3. So sánh

| Tiêu chí | Underfitting | Overfitting |
|---|---|---|
| Train error | Cao | Thấp |
| Test/validation error | Cao | Cao |
| Gap train ↔ test | Nhỏ | Lớn |
| Bias | Cao | Thấp |
| Variance | Thấp | Cao |
| Độ phức tạp model | Quá thấp | Quá cao |
| Bản chất | Học **chưa đủ** | Học cả **nhiễu** |

### 4. Cách khắc phục
**Underfitting** (tăng năng lực học):
- Tăng độ phức tạp model: thêm feature, bậc cao hơn, cây sâu hơn, thêm layer/neuron.
- Feature engineering: tạo/thêm feature có thông tin.
- **Giảm** regularization (giảm λ).
- Train lâu hơn; đổi sang model mạnh hơn (phi tuyến).

**Overfitting** (tăng khả năng tổng quát):
- **Thêm dữ liệu** / data augmentation.
- **Giảm** độ phức tạp model: đơn giản hoá, pruning cây, giảm bậc, giảm layer.
- **Regularization**: L1/L2, dropout, **early stopping** theo validation.
- **Cross-validation** (k-fold) để chọn model/hyperparameter.
- Feature selection / giảm chiều (PCA); **ensemble** (bagging, Random Forest) để hạ variance.

> Quy tắc vàng: chẩn đoán bằng **khoảng cách train–test**. Gap nhỏ mà cả hai đều tệ → underfit (tăng độ phức tạp). Gap lớn, train tốt test tệ → overfit (regularize / thêm data).

**Ví dụ:** chẩn đoán bằng learning curve (so train score vs validation score)
```python
import numpy as np
from sklearn.model_selection import learning_curve

train_sizes, train_sc, val_sc = learning_curve(
    model, X, y, cv=5, scoring="accuracy",
    train_sizes=np.linspace(0.1, 1.0, 5),
)
tr, va = train_sc.mean(1), val_sc.mean(1)
gap = tr[-1] - va[-1]

if tr[-1] < 0.8 and va[-1] < 0.8:
    print("Underfitting: train & val đều thấp → tăng độ phức tạp / thêm feature")
elif gap > 0.1:
    print(f"Overfitting: gap={gap:.2f} lớn → regularize / thêm data")
else:
    print("Cân bằng tốt: gap nhỏ, val cao")
```

## Q3. Nhận biết mô hình bias cao / variance cao — tại sao bị, ví dụ dễ hiểu

**Hỏi:** Làm sao nhận biết mô hình bias cao, variance cao? Tại sao bị? Cho ví dụ dễ hiểu.

**Trả lời ngắn:** Nhìn **2 con số: train error và validation error**. Cả hai **đều cao** và sát nhau → **bias cao** (model quá cứng, học chưa tới). Train **rất thấp** nhưng validation **cao** (gap lớn) → **variance cao** (model quá nhạy, học cả nhiễu). Bias cao do model quá đơn giản; variance cao do model quá phức tạp / ít dữ liệu.

**Chi tiết:**

### 1. Nhận biết bias cao
- **Dấu hiệu:** train error **cao**, validation error **cao**, hai số **gần bằng nhau** (cùng tệ).
- Thêm dữ liệu **không cứu được** — đường train và val đều phẳng ở mức tệ.
- Dự đoán đơn điệu, bỏ qua pattern rõ ràng trong dữ liệu.

### 2. Nhận biết variance cao
- **Dấu hiệu:** train error **rất thấp** (gần như hoàn hảo) nhưng validation error **cao** → **gap lớn**.
- **Không ổn định:** đổi một phần nhỏ tập train → model thay đổi nhiều, kết quả nhảy lung tung.
- Thêm dữ liệu **giúp ích** — gap thu hẹp dần.

### 3. Tại sao bị
- **Bias cao:** giả định của model quá đơn giản/cứng so với thực tế (dùng đường thẳng cho quan hệ cong, cây quá nông, regularization quá mạnh). Model **không đủ sức** biểu diễn pattern → sai lệch hệ thống.
- **Variance cao:** model quá linh hoạt (nhiều tham số, cây quá sâu, polynomial bậc cao) + **ít dữ liệu** → nó "thuộc lòng" cả nhiễu của tập train thay vì học quy luật chung → gặp dữ liệu mới là sai.

### 4. Ví dụ dễ hiểu

**a) Bắn cung vào bia (kinh điển):**
- **Bias cao, variance thấp:** các mũi tên chụm nhau nhưng **lệch hẳn** khỏi tâm về một phía → sai *đều đặn* một kiểu.
- **Variance cao, bias thấp:** các mũi tên **trung bình quanh tâm** nhưng **tản mác** khắp nơi → lúc trúng lúc trượt, không ổn định.
- Lý tưởng: chụm **và** đúng tâm (bias thấp, variance thấp).

**b) Học sinh ôn thi:**
- **Bias cao:** học sinh chỉ học **một công thức** rồi áp cho mọi bài → bài tập ở nhà cũng sai, đi thi cũng sai (sai như nhau).
- **Variance cao:** học sinh **học thuộc lòng đáp án** từng bài trong đề cương → làm lại đề cương đúng 100%, nhưng đề thi đổi số là trượt.

**c) Vẽ đường khớp dữ liệu cong:**
- **Bias cao:** ép một **đường thẳng** qua đám điểm hình chữ U → trượt xa ở hai đầu.
- **Variance cao:** dùng **đa thức bậc rất cao** uốn éo đi qua *đúng từng điểm* (kể cả điểm nhiễu) → giữa các điểm thì sai bét, gặp điểm mới là lệch.

> Mẹo nhớ: **bias = sai vì lười học (quá đơn giản)**; **variance = sai vì học vẹt (quá phức tạp)**. Chẩn đoán luôn nhìn cặp (train error, val error) và **gap** giữa chúng.

## Q4. Ý nghĩa của `stratify` khi chia dữ liệu

**Hỏi:** `stratify` trong việc chia dữ liệu có ý nghĩa gì?

**Trả lời ngắn:** `stratify` = **chia phân tầng**: giữ cho **tỉ lệ các lớp** (class distribution) trong tập train và test **giống y như** tập gốc. Quan trọng nhất với **dữ liệu mất cân bằng** — tránh việc chia ngẫu nhiên làm lớp hiếm bị thiếu/lệch ở một phía.

**Chi tiết:**

### 1. Vấn đề khi chia ngẫu nhiên
Mặc định `train_test_split` chia **ngẫu nhiên**. Với dữ liệu lệch lớp, ngẫu nhiên có thể tạo train/test có tỉ lệ lớp khác hẳn tập gốc — thậm chí một lớp hiếm gần như **biến mất** khỏi test. Hậu quả:
- Test không đại diện → chỉ số đánh giá (accuracy, F1) **không trung thực**.
- Train thiếu mẫu lớp hiếm → model học kém lớp đó.

### 2. `stratify` giải quyết gì
Truyền `stratify=y` (y = nhãn lớp) → mỗi lớp được chia theo **đúng tỉ lệ** vào cả train lẫn test. Ví dụ lớp chiếm 10% ở tập gốc thì train và test đều ~10% lớp đó.

### 3. Khi nào dùng
- **Phân loại (classification)**, nhất là **mất cân bằng** hoặc tập nhỏ → nên luôn `stratify=y`.
- **Hồi quy (regression)** với target liên tục → `stratify` trực tiếp **không dùng được** (phải bin hoá target thành nhóm trước nếu muốn phân tầng).
- Trong cross-validation, tương đương là `StratifiedKFold`.

### 4. Liên hệ dataset trong repo
- **ViHSD** (CLEAN / OFFENSIVE / HATE) — 3 lớp **mất cân bằng** → bắt buộc `stratify=y` để mỗi lớp đều có mặt đúng tỉ lệ ở train/test.
- **Hanoi housing** (giá nhà, liên tục) — bài hồi quy → **không** stratify theo giá (trừ khi chia giá thành khoảng).

**Ví dụ:** dữ liệu 1000 mẫu, lớp 0 = 90%, lớp 1 = 10%, `test_size=0.2`.
```python
from sklearn.model_selection import train_test_split

# KHÔNG stratify: tỉ lệ lớp 1 ở test có thể trôi (5%? 15%?), không ổn định
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

# CÓ stratify: test giữ đúng ~10% lớp 1 như tập gốc
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
# y_test: lớp 0 ≈ 180 mẫu (90%), lớp 1 ≈ 20 mẫu (10%) — phản ánh đúng phân phối gốc
```

> Quy tắc: bài phân loại → mặc định `stratify=y`. Đặc biệt khi lớp hiếm, đây là cách duy nhất đảm bảo cả train lẫn test đều "nhìn thấy" lớp đó theo đúng tỉ lệ.

## Q5. Chia train / val / test chuẩn nhất — chia mấy phần, cắt ra sao (real-world)

**Hỏi:** Cách chia tập train/test/val chuẩn nhất? Chia mấy phần, cắt ra sao? (theo thực tế)

**Trả lời ngắn:** Chia **3 tập** vai trò khác nhau: **train** (học), **val/dev** (tune & chọn model), **test** (đánh giá cuối, khoá lại chỉ chạm 1 lần). Tỉ lệ **KHÔNG cố định** — tuỳ kích thước data: data nhỏ ~ 60/20/20 hoặc 70/15/15; data lớn (triệu+) ~ 98/1/1. Trong thực tế, điều quan trọng hơn con số là: **chia theo thời gian** nếu có yếu tố thời gian, **cùng phân phối với production**, và **khoá test set**.

**Chi tiết:**

### 1. Ba tập làm gì
- **train** — học tham số model.
- **validation (dev)** — tune hyperparameter, chọn model, early stopping. Chạm **nhiều lần**.
- **test** — đo hiệu năng cuối cùng trên dữ liệu "chưa từng thấy". Chạm **đúng 1 lần**, ở cuối.

### 2. Tỉ lệ — theo lượng data, không phải con số thần thánh
- **Nhỏ/vừa** (vài nghìn → vài chục nghìn mẫu): `60/20/20`, `70/15/15`, `80/10/10`.
- **Rất lớn** (≥ 1 triệu): `98/1/1` — vì 1% đã là chục nghìn mẫu, quá đủ để chỉ số ổn định; phần còn lại dồn hết cho train.
- Quy tắc thật: val/test chỉ cần **đủ lớn để metric ổn định** (vài nghìn mẫu là ổn), không cần phải đúng 20%.

### 3. Cắt thế nào (sklearn — `train_test_split` 2 lần)
`train_test_split` không tách thẳng 3 phần → tách **2 lần**: lấy test ra trước, rồi cắt val từ phần còn lại.
```python
from sklearn.model_selection import train_test_split

# B1: tách TEST (20%) ra trước rồi KHOÁ lại
X_tmp, X_test, y_tmp, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42)

# B2: cắt VAL từ phần còn lại. Muốn val = 20% tổng → 0.20/0.80 = 0.25
X_train, X_val, y_train, y_val = train_test_split(
    X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=42)

# Kết quả: train 60% / val 20% / test 20%
```
Công thức phần cắt lần 2: `val_size_2 = val_mong_muốn / (1 - test_size)`.
(Nhớ: scaler `fit` **chỉ trên train**, rồi `transform` val/test — xem Q1; cân bằng cũng **chỉ trên train**.)

### 4. Real-world — mấy điều textbook hay bỏ
- **Chia theo thời gian** (time-based) cho dữ liệu có mốc thời gian / bài toán production: train = **quá khứ**, val/test = **tương lai**, **KHÔNG shuffle**. Shuffle ngẫu nhiên = leakage thời gian → điểm đẹp ảo, deploy là sập.
- **Cùng phân phối với production**: val & test phải giống dữ liệu thật lúc chạy thật. Tune trên phân phối lệch = tối ưu nhầm mục tiêu.
- **Group split**: nếu nhiều dòng cùng một thực thể (cùng user, cùng căn nhà, cùng phiên) → gom **cùng một phía** (`GroupKFold` / split theo group), tránh model "thấy" thực thể đó ở cả train lẫn test.
- **K-fold CV khi data ít**: thay val cố định bằng **k-fold cross-validation** trên (train+val) để tune, vẫn giữ **test held-out** riêng. Tận dụng hết data, ước lượng ổn định hơn.
- **Khoá test**: chỉ mở 1 lần ở cuối. Tune đi tune lại theo test = overfit lên test set ("test set rot").

### 5. Liên hệ dataset trong repo
- **ViHSD** (comment MXH, 3 lớp lệch): `stratify=y` ở **cả 2 lần** cắt. Nếu comment có timestamp → cân nhắc chia theo thời gian.
- **Hanoi housing** (giá nhà, hồi quy): nếu tin đăng có **ngày** → chia theo thời gian để mô phỏng "dự đoán giá tương lai"; nếu không → random split + k-fold CV (regression không stratify theo giá trừ khi bin hoá).

> Tóm: con số tỉ lệ là thứ yếu. Thứ tự đúng + **không leakage** (chia trước khi xử lý, theo thời gian/nhóm khi cần) + **test khoá kín** mới là cái làm kết quả đáng tin ngoài thực tế.

## Q6. K-Fold Cross-Validation — apply ra sao, chọn loại Fold nào & vì sao

**Hỏi:** Apply K-Fold thế nào, in ra từng fold? Nên dùng loại Fold nào và vì sao? (ghi chú kĩ, dễ đọc trên mobile)

**Trả lời ngắn:** K-Fold chia dữ liệu thành **K khối**; mỗi vòng lấy **1 khối làm val**, K−1 khối còn lại làm train; lặp K lần → mỗi mẫu được val đúng 1 lần. Kết quả báo cáo **mean ± std**. **Chọn loại Fold theo bản chất dữ liệu**, không mặc định một loại.

**Vì sao:** 1 lần chia train/val có thể "ăn may" hoặc "ăn rủi" → điểm không tin cậy. K-Fold lấy trung bình nhiều lần chia → ước lượng **ổn định** hơn, và `std` cho biết model **nhạy** với cách chia tới đâu (std lớn = variance cao).

**Chi tiết:**

### Cách K-Fold chạy
- Chia train-full thành K khối đều nhau (vd K=5 → mỗi khối ~20%).
- Vòng i: khối i = val, phần còn lại = train.
- Lặp K vòng → K điểm số → lấy `mean ± std`.
- `test` **tách riêng (held-out)**, KHÔNG tham gia CV.

### Chọn loại Fold nào — và lý do
- **KFold** (chia ngẫu nhiên):
  - dùng cho **hồi quy** / dữ liệu độc lập (i.i.d.).
  - vì không có "lớp" để phải giữ tỉ lệ.
- **StratifiedKFold** (giữ tỉ lệ lớp):
  - dùng cho **phân loại**, nhất là **mất cân bằng**.
  - vì KFold thường dễ tạo fold **thiếu lớp hiếm** → điểm dao động/sai.
- **GroupKFold** (giữ nhóm 1 phía):
  - khi có **thực thể trùng** (cùng user, cùng nhà, cùng phiên).
  - vì cùng nhóm ở cả train lẫn val = **leakage theo nhóm**.
- **TimeSeriesSplit** (train quá khứ → val tương lai):
  - khi dữ liệu có **thời gian** / bài forecasting.
  - vì shuffle ngẫu nhiên cho model "thấy tương lai" → **leakage thời gian**.

### Thứ tự ĐÚNG trong mỗi fold
- Cân bằng + `fit` (vectorizer/scaler/encoder) đặt **trong** vòng lặp, **chỉ** trên train-fold.
- val-fold chỉ `transform`. Dễ nhất: gói vào **Pipeline** rồi `pipe.fit(train_fold)`.
- Lý do: làm ngoài vòng lặp (trên cả dữ liệu) = val-fold rò rỉ vào bước học → điểm ảo.

### Kết quả thực nghiệm (notebook `Pipeline_Split_Balance_CV.ipynb`)
- **ViHSD → StratifiedKFold(5):** F1-macro = **0.6231 ± 0.0083** (5 fold sát nhau → ổn định).
- **Leakage demo:** cân bằng TRƯỚC khi chia → **0.9378** (ảo), so với làm đúng **0.6231** → thổi phồng **+0.31**.
- **Hanoi housing → KFold(5):** RMSE = **42.30 ± 0.67** triệu/m², R² = **0.313**.
- **Hanoi housing → TimeSeriesSplit(5):** RMSE = **42.02 ± 1.41**; train lớn dần (13.5k → 67.6k), val luôn là quãng thời gian kế tiếp.

**Ví dụ:**
```python
# Phân loại lệch lớp -> StratifiedKFold (giữ tỉ lệ lớp)
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for i, (tr, va) in enumerate(skf.split(X, y), 1):
    # cân bằng + fit vectorizer CHỈ trên X[tr]; transform X[va]
    ...

# Hồi quy -> KFold; có thời gian -> TimeSeriesSplit
from sklearn.model_selection import KFold, TimeSeriesSplit
kf  = KFold(n_splits=5, shuffle=True, random_state=42)   # i.i.d.
tss = TimeSeriesSplit(n_splits=5)                         # train quá khứ -> val tương lai
```

> Tóm: K-Fold = lấy trung bình nhiều lần chia cho ổn định. **Loại Fold chọn theo dữ liệu**: lệch lớp → Stratified; nhóm trùng → Group; có thời gian → TimeSeries; còn lại → KFold.
