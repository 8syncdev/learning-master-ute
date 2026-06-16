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
