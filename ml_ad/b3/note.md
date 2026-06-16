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
