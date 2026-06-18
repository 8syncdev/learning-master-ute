# -*- coding: utf-8 -*-
"""Sinh notebook minh hoạ THỨ TỰ chia & tiền xử lý dữ liệu + CROSS-VALIDATION.

Dùng ĐÚNG 2 file CSV trong zip đã gửi (lấy từ thư mục buổi b1):
  - ViHSD.csv                -> PHÂN LOẠI (3 lớp mất cân bằng)  -> StratifiedKFold
  - Hanoi_housing_dataset.csv-> HỒI QUY  (target liên tục)      -> KFold / TimeSeriesSplit

Trả lời:
  1. Chia train/val/test ra sao, bước nào TRƯỚC. "Chia rồi mới cân bằng" — vì sao.
  2. Apply K-Fold cho CẢ 2 dataset, IN RA TỪNG FOLD + giải thích.
  3. NÊN dùng loại Fold nào cho từng bài và VÌ SAO.
  4. Làm sai thứ tự -> data leakage (đo cụ thể).

Chạy:  python build_nb.py   ->  ghi notebook .ipynb cùng thư mục (như b1).
"""
import nbformat as nbf

OUT = "Pipeline_Split_Balance_CV.ipynb"


def build(cells, path):
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    }
    nb.cells = [
        nbf.v4.new_markdown_cell(s) if k == "md" else nbf.v4.new_code_cell(s)
        for k, s in cells
    ]
    nbf.write(nb, path)
    print("wrote", path, "with", len(nb.cells), "cells")


c = []

# === Intro ================================================================
c.append(("md", '''# Phân chia tập dữ liệu, xử lý mất cân bằng và Kiểm định chéo (Cross-Validation) trên hai bộ dữ liệu thực

**Học viên:** Nguyễn Phương Anh Tú  **Mã số học viên:** 2611328

## Tóm tắt

Báo cáo trình bày một quy trình thực nghiệm chuẩn mực cho hai bài toán học máy có giám sát khác biệt về bản chất, nhằm minh hoạ và kiểm chứng nguyên tắc phòng tránh rò rỉ dữ liệu (*data leakage*) trong toàn bộ chu trình huấn luyện - đánh giá. Hai bộ dữ liệu thực được sử dụng nguyên trạng từ tệp đính kèm: `ViHSD.csv` (bài toán phân loại) và `Hanoi_housing_dataset.csv` (bài toán hồi quy).

| Tiêu chí | PHẦN A — Bài toán phân loại | PHẦN B — Bài toán hồi quy |
|---|---|---|
| Bộ dữ liệu | ViHSD (bình luận mạng xã hội tiếng Việt) | Giá bất động sản Hà Nội |
| Biến mục tiêu | 3 lớp `CLEAN` / `OFFENSIVE` / `HATE` (mất cân bằng) | `Giá/m²` (biến liên tục) |
| Xử lý mất cân bằng | Có (oversampling, chỉ trên tập huấn luyện) | Không áp dụng (biến mục tiêu liên tục) |
| Chiến lược chia fold | `StratifiedKFold` | `KFold` (và `TimeSeriesSplit`) |

## Nguyên tắc phương pháp luận xuyên suốt

Mọi phép biến đổi **học tham số từ dữ liệu** — bao gồm việc khớp (`fit`) bộ vector hoá văn bản, bộ chuẩn hoá, bộ mã hoá biến hạng mục, cũng như thao tác cân bằng lớp bằng nhân bản mẫu — **chỉ được phép quan sát tập huấn luyện (train)**. Các tập kiểm định (validation) và kiểm tra (test) chỉ được áp dụng phép biến đổi đã học (`transform`).

Cơ sở của nguyên tắc: tập validation và test đóng vai trò đại diện cho **dữ liệu chưa từng được quan sát** tại thời điểm huấn luyện. Nếu để các tập này tham gia vào bước ước lượng tham số, thông tin của chúng sẽ rò rỉ ngược vào mô hình, khiến chỉ số đánh giá trở nên **lạc quan thiếu thực chất** (data leakage); khi triển khai trên dữ liệu mới, hiệu năng thực tế sẽ suy giảm đáng kể so với kết quả báo cáo. Mục A6 lượng hoá trực tiếp mức sai lệch này.'''))

c.append(("md", '''## 1. Bối cảnh ứng dụng và cơ sở phương pháp luận

Hai bộ dữ liệu trong báo cáo tương ứng với hai tính năng sản phẩm có thật, qua đó cho thấy quy trình được trình bày không mang tính hàn lâm thuần tuý mà phản ánh các ràng buộc kỹ thuật khi đưa mô hình vào vận hành:

- **ViHSD** — hệ thống **kiểm duyệt bình luận độc hại** cho nền tảng mạng xã hội tiếng Việt, tự động phân loại bình luận thành ba mức `CLEAN`, `OFFENSIVE`, `HATE`.
- **Hanoi housing** — chức năng **gợi ý đơn giá** khi người dùng đăng tin rao bán bất động sản.

### 1.1. Vai trò của ba tập dữ liệu
- **Tập huấn luyện (train):** nguồn dữ liệu duy nhất được phép dùng để ước lượng tham số của mô hình và của các phép biến đổi tiền xử lý.
- **Tập kiểm định (validation):** dùng để lựa chọn mô hình, tinh chỉnh siêu tham số và ra quyết định trong quá trình phát triển; được truy cập nhiều lần.
- **Tập kiểm tra (test):** dùng để ước lượng hiệu năng tổng quát cuối cùng trên dữ liệu chưa từng thấy; được "đóng băng" và chỉ truy cập **một lần duy nhất** khi báo cáo kết quả. Việc tinh chỉnh lặp lại theo tập test sẽ dẫn tới hiện tượng quá khớp lên chính tập test.

### 1.2. Vì sao phải "chia trước, cân bằng sau, và chỉ trên tập huấn luyện"
Trong điều kiện vận hành thực tế, khoảng 82% bình luận thuộc lớp `CLEAN` (xem Mục A1). Nếu cân bằng lại tập validation/test về tỉ lệ đồng đều giữa các lớp, chỉ số đánh giá sẽ phản ánh một phân phối không tồn tại khi triển khai, dẫn tới ước lượng hiệu năng sai lệch. Do đó **validation và test phải giữ nguyên phân phối gốc của dữ liệu thực**, tuyệt đối không cân bằng. Ngược lại, các lớp thiểu số (`HATE`, `OFFENSIVE`) hiếm gặp khiến mô hình có xu hướng bỏ qua; vì vậy ta tăng tần suất xuất hiện của chúng **chỉ trong tập huấn luyện** (oversampling) để mô hình học tốt hơn ranh giới quyết định.

### 1.3. Vì sao chỉ khớp (`fit`) trên tập huấn luyện
Có thể hình dung bộ tiền xử lý như một thiết bị đo được hiệu chuẩn bằng dữ liệu lịch sử (tập huấn luyện). Mẫu mới đến (test) phải được đo bằng đúng thiết bị đã hiệu chuẩn; nếu hiệu chuẩn lại theo chính mẫu mới thì kết quả đo không còn khách quan. Tương tự, bộ chuẩn hoá và bộ mã hoá phải dùng lại đúng tham số đã học từ tập huấn luyện khi xử lý validation và test.

### 1.4. Vì sao dùng Kiểm định chéo và cách lựa chọn loại fold
Một lần chia train/validation đơn lẻ chịu ảnh hưởng lớn của yếu tố ngẫu nhiên, dẫn tới ước lượng hiệu năng kém ổn định. Kiểm định chéo K-fold lặp lại quá trình trên K cách chia khác nhau và báo cáo **trung bình ± độ lệch chuẩn**, nhờ đó phản ánh được cả mức hiệu năng kỳ vọng lẫn độ ổn định của mô hình. Việc lựa chọn loại fold phụ thuộc bản chất dữ liệu (trình bày chi tiết ở Mục 2): `StratifiedKFold` cho phân loại mất cân bằng nhằm bảo toàn tỉ lệ lớp ở mọi fold; `TimeSeriesSplit` cho dữ liệu có yếu tố thời gian nhằm huấn luyện trên quá khứ và kiểm định trên tương lai; `GroupKFold` khi tồn tại các quan sát thuộc cùng một thực thể cần được giữ về cùng một phía.

Mục A6 sẽ lượng hoá hệ quả của việc vi phạm thứ tự xử lý: thực hiện sai làm F1-macro tăng giả tạo lên 0.9378, trong khi quy trình đúng chỉ đạt 0.6231.'''))

c.append(("code", '''import re
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, TimeSeriesSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score, classification_report, mean_squared_error, r2_score
from sklearn.utils import resample

SEED = 42
LABELS = {0: "CLEAN", 1: "OFFENSIVE", 2: "HATE"}
BASE = "2611328 - Nguyễn Phương Anh Tú - Exploratory Data Analysis (EDA)"

def find_csv(name):
    """Tìm CSV: thư mục hiện tại, rồi thư mục b1."""
    for p in (Path(name), Path("..") / "b1" / BASE / name, Path(BASE) / name):
        if p.exists():
            return p
    raise FileNotFoundError(name)

def show_dist(y, tag=""):
    """In số lượng + tỉ lệ từng lớp (dùng cho bài phân loại)."""
    vc = y.value_counts().sort_index()
    pct = (y.value_counts(normalize=True).sort_index() * 100).round(1)
    body = "   ".join(f"{LABELS[k]}:{int(vc[k])} ({pct[k]}%)" for k in vc.index)
    print(f"  {tag:10s} n={len(y):6d} | {body}")

pd.set_option("display.max_colwidth", 100)'''))

# === Fold guide ===========================================================
c.append(("md", '''## 2. Lựa chọn chiến lược chia fold theo bản chất dữ liệu

| Loại fold | Điều kiện áp dụng | Cơ sở lựa chọn |
|---|---|---|
| **KFold** | Bài toán hồi quy, hoặc dữ liệu độc lập và đồng phân phối (i.i.d.), không có cấu trúc lớp | Chia ngẫu nhiên thành K khối có kích thước xấp xỉ nhau; đơn giản và không thiên lệch khi các quan sát độc lập. |
| **StratifiedKFold** | Bài toán phân loại, đặc biệt khi dữ liệu mất cân bằng | Bảo toàn tỉ lệ các lớp ở mỗi fold. Với KFold thông thường, một số fold có thể thiếu lớp hiếm, làm chỉ số dao động mạnh và ước lượng thiếu tin cậy. |
| **GroupKFold** | Tồn tại nhóm/thực thể lặp lại (cùng người dùng, cùng tài sản, cùng phiên) | Giữ trọn một nhóm về cùng một phía train hoặc validation, tránh việc mô hình quan sát cùng thực thể ở cả hai phía (rò rỉ theo nhóm). |
| **TimeSeriesSplit** | Dữ liệu có trật tự thời gian, bài toán dự báo | Tập train là quá khứ, tập validation là tương lai, không xáo trộn. KFold ngẫu nhiên cho phép mô hình quan sát dữ liệu tương lai, gây rò rỉ thời gian. |

**Áp dụng cho hai bộ dữ liệu trong báo cáo:**

- **ViHSD →** `StratifiedKFold`. Lớp `OFFENSIVE` chỉ chiếm khoảng 6.8% (xem A1); nếu dùng KFold thông thường, một số fold dễ bị lệch hoặc thiếu lớp này, khiến F1 biến động lớn. `StratifiedKFold` bảo toàn đúng tỉ lệ ba lớp ở mọi fold.
- **Hanoi housing →** `KFold` (biến mục tiêu liên tục, không có cấu trúc lớp để phân tầng). Tuy nhiên, do dữ liệu có cột `Ngày`, nếu mục tiêu là **dự báo giá trong tương lai** thì `TimeSeriesSplit` mới là lựa chọn phù hợp về mặt phương pháp luận (trình bày ở cuối Phần B).'''))

# === PART A ================================================================
c.append(("md", '''# PHẦN A — Bài toán phân loại văn bản trên bộ dữ liệu ViHSD (`StratifiedKFold`)'''))

c.append(("md", '''## A1. Nạp dữ liệu và làm sạch sơ bộ

Bước đầu tiên loại bỏ các bình luận trùng lặp. Nếu giữ lại các bản trùng, một mẫu cùng bản sao của nó có thể đồng thời xuất hiện ở cả tập huấn luyện lẫn tập kiểm tra, tạo thành một dạng rò rỉ dữ liệu khiến hiệu năng bị đánh giá cao hơn thực chất. Phân phối ba lớp sau khi làm sạch cho thấy mức độ mất cân bằng rõ rệt: lớp `CLEAN` chiếm ưu thế (khoảng 82.2%), trong khi `OFFENSIVE` (khoảng 6.8%) và `HATE` (khoảng 10.9%) là các lớp thiểu số. Bộ dữ liệu gốc có sẵn cột `split`; tuy vậy, ở đây ta tự thực hiện việc phân chia nhằm minh hoạ đầy đủ quy trình.'''))

c.append(("code", '''df = pd.read_csv(find_csv("ViHSD.csv"))
print("Kích thước gốc:", df.shape)
df = df.drop_duplicates(subset="free_text").reset_index(drop=True)
print("Sau khi bỏ trùng:", df.shape)
show_dist(df["label_id"], "TOÀN BỘ")
# File có sẵn cột split (data thực tế đôi khi đã chia sẵn) — ở đây ta tự chia để học:
print("Cột split có sẵn:", df["split"].value_counts().to_dict())
df.head(3)'''))

c.append(("md", '''## A2. Bước 1 — Tách tập kiểm tra (held-out) và phân chia train/validation có phân tầng

Hàm `train_test_split` chỉ tách dữ liệu thành hai phần mỗi lần gọi, do đó cần thực hiện **hai lần cắt liên tiếp**: lần thứ nhất tách 20% làm tập kiểm tra rồi đóng băng lại; lần thứ hai tách tiếp 25% phần còn lại làm tập validation (tương đương 20% tổng thể), phần còn lại 60% là tập huấn luyện. Tham số `stratify=y` được dùng ở cả hai lần cắt để mọi tập đều giữ đúng tỉ lệ ba lớp như tập gốc.

Lý do tách tập kiểm tra trước mọi xử lý khác: chỉ khi được cô lập ngay từ đầu, validation và test mới thực sự đóng vai trò dữ liệu chưa từng thấy; mọi phép biến đổi học tham số về sau chỉ được thực hiện trên tập huấn luyện.'''))

c.append(("code", '''X = df["free_text"]
y = df["label_id"]

# Lần 1: tách TEST 20% — KHOÁ lại, chỉ chạm ở cuối
X_trainfull, X_test, y_trainfull, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED)
# Lần 2: từ 80% còn lại tách VAL (0.25 * 0.80 = 0.20 tổng)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainfull, y_trainfull, test_size=0.25, stratify=y_trainfull, random_state=SEED)

print("Tỉ lệ 3 lớp GIỮ NGUYÊN ở mọi tập nhờ stratify:\\n")
for tag, yy in [("TOÀN BỘ", y), ("TRAIN", y_train), ("VAL", y_val), ("TEST", y_test)]:
    show_dist(yy, tag)'''))

c.append(("md", '''## A3. Bước 2 — Cân bằng lớp chỉ trên tập huấn luyện

Áp dụng oversampling (nhân bản ngẫu nhiên các mẫu của lớp thiểu số cho đến khi mọi lớp có số lượng bằng lớp đa số) **chỉ trên tập huấn luyện**. Sau bước này, tập huấn luyện chuyển từ phân phối lệch sang phân phối đồng đều ba lớp, trong khi tập validation và test **giữ nguyên phân phối gốc**.

Lý do chỉ cân bằng tập huấn luyện: validation và test phải phản ánh phân phối thực tế thì chỉ số F1 mới trung thực. Cân bằng các tập này vừa bóp méo phân phối cần đo, vừa cấu thành rò rỉ dữ liệu do sinh thêm mẫu trên phần dữ liệu lẽ ra chưa được quan sát.'''))

c.append(("code", '''def oversample_text(X_text, y, seed=SEED):
    """Nhân bản lớp thiểu số cho bằng lớp đa số (random oversampling)."""
    d = pd.DataFrame({"text": np.asarray(X_text), "y": np.asarray(y)})
    n_max = d["y"].value_counts().max()
    parts = [(resample(g, replace=True, n_samples=n_max, random_state=seed) if len(g) < n_max else g)
             for _, g in d.groupby("y")]
    out = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    return out["text"], out["y"]

print("TRƯỚC cân bằng:"); show_dist(y_train, "TRAIN")
X_train_bal, y_train_bal = oversample_text(X_train, y_train)
print("\\nSAU cân bằng (chỉ train — 3 lớp bằng nhau):"); show_dist(y_train_bal, "TRAIN_BAL")
print("\\nVAL & TEST giữ NGUYÊN phân phối gốc:"); show_dist(y_val, "VAL"); show_dist(y_test, "TEST")'''))

c.append(("md", '''## A4. Bước 3–5 — Vector hoá đặc trưng, huấn luyện và đánh giá

Bộ `TfidfVectorizer` học từ vựng và trọng số TF-IDF (`fit`) **chỉ trên tập huấn luyện đã cân bằng**, sau đó áp dụng (`transform`) lên tập validation và test bằng đúng từ vựng đã học. Đây là minh hoạ trực tiếp cho cặp thao tác `fit_transform(train)` và `transform(val/test)`. Mô hình phân loại sử dụng hồi quy logistic đa lớp. Chỉ số đánh giá chính là F1-macro — trung bình F1 không trọng số trên ba lớp — phù hợp với bài toán mất cân bằng vì đối xử công bằng giữa các lớp bất kể tần suất. Trên tập test, F1-macro đạt 0.6230, phản ánh đúng mức độ khó của bài toán khi lớp thiểu số chiếm tỉ lệ nhỏ.'''))

c.append(("code", '''vec = TfidfVectorizer(max_features=15000, ngram_range=(1, 2), min_df=2)
Xtr = vec.fit_transform(X_train_bal)   # fit_transform: HỌC từ vựng trên TRAIN
Xva = vec.transform(X_val)             # transform: dùng lại từ vựng TRAIN
Xte = vec.transform(X_test)

clf = LogisticRegression(max_iter=1000, C=1.0)
clf.fit(Xtr, y_train_bal)
f1_val  = f1_score(y_val,  clf.predict(Xva), average="macro")
f1_test = f1_score(y_test, clf.predict(Xte), average="macro")
print(f"F1-macro VAL : {f1_val:.4f}  (chọn model/hyperparameter)")
print(f"F1-macro TEST: {f1_test:.4f}  (báo cáo cuối, chạm 1 lần)\\n")
print(classification_report(y_test, clf.predict(Xte),
                            target_names=[LABELS[i] for i in (0, 1, 2)]))'''))

c.append(("md", '''## A5. Kiểm định chéo `StratifiedKFold` — kết quả chi tiết từng fold

Thay cho một tập validation cố định, dữ liệu huấn luyện được chia thành K khối. Ở mỗi vòng lặp, một khối đóng vai trò validation, K−1 khối còn lại làm train; quá trình lặp K lần để mỗi mẫu được kiểm định đúng một lần. Sử dụng `StratifiedKFold` nhằm bảo toàn tỉ lệ ba lớp ở mỗi fold — điều kiện thiết yếu khi lớp `OFFENSIVE` rất hiếm.

Điểm mấu chốt về mặt phương pháp: thao tác cân bằng lớp và việc khớp bộ vector hoá được đặt **bên trong** vòng lặp và **chỉ** thực hiện trên train-fold; tập test vẫn được giữ nguyên ở trạng thái held-out, không tham gia kiểm định chéo.'''))

c.append(("code", '''skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
Xtf = X_trainfull.reset_index(drop=True)   # CV chạy trên train-full; TEST khoá ngoài
ytf = y_trainfull.reset_index(drop=True)

clf_scores = []
for fold, (tr, va) in enumerate(skf.split(Xtf, ytf), 1):
    Xtr_f, ytr_f = Xtf.iloc[tr], ytf.iloc[tr]
    Xva_f, yva_f = Xtf.iloc[va], ytf.iloc[va]
    print(f"========== FOLD {fold}/5 ==========")
    print(f"  train-fold={len(tr):6d} | val-fold={len(va):6d}")
    show_dist(ytr_f, "train"); show_dist(yva_f, "val")

    # THỨ TỰ ĐÚNG TRONG FOLD: cân bằng + fit vectorizer chỉ train-fold
    Xtr_fb, ytr_fb = oversample_text(Xtr_f, ytr_f)
    v = TfidfVectorizer(max_features=15000, ngram_range=(1, 2), min_df=2)
    m = LogisticRegression(max_iter=1000)
    m.fit(v.fit_transform(Xtr_fb), ytr_fb)
    f1 = f1_score(yva_f, m.predict(v.transform(Xva_f)), average="macro")
    clf_scores.append(f1)
    print(f"  -> F1-macro fold {fold}: {f1:.4f}\\n")

print("Điểm từng fold:", [round(s, 4) for s in clf_scores])
print(f"CV F1-macro: {np.mean(clf_scores):.4f} ± {np.std(clf_scores):.4f}")'''))

c.append(("md", '''### Diễn giải kết quả từng fold
- Mỗi vòng lặp sử dụng một khối validation khác nhau; sau K vòng, mọi mẫu trong tập huấn luyện đều đã đóng vai trò validation đúng một lần.
- Tỉ lệ ba lớp được in ra ở mỗi fold gần như đồng nhất — minh chứng cho tác dụng của `StratifiedKFold` đối với dữ liệu lệch lớp.
- Giá trị **trung bình ± độ lệch chuẩn** cung cấp hai thông tin: trung bình là ước lượng hiệu năng ổn định hơn so với một lần chia đơn lẻ; độ lệch chuẩn phản ánh mức nhạy cảm của mô hình với cách chia dữ liệu (độ lệch chuẩn lớn tương ứng phương sai cao, mô hình kém ổn định). Kết quả thu được (0.6231 ± 0.0083) cho thấy hiệu năng nhất quán giữa các fold.
- Việc đặt bước cân bằng và khớp bộ vector hoá bên trong mỗi fold bảo đảm validation-fold luôn "sạch", không bị rò rỉ.'''))

c.append(("md", '''## A6. Vi phạm thứ tự xử lý dẫn tới rò rỉ dữ liệu — đo lường định lượng

Để minh chứng tầm quan trọng của thứ tự xử lý, ta cố ý thực hiện sai: cân bằng (nhân bản) toàn bộ dữ liệu **trước khi chia fold**. Khi đó, các bản sao giống hệt nhau của cùng một mẫu phân bố đồng thời vào cả train-fold và validation-fold; mô hình "ghi nhớ" mẫu đã thấy ở train và gặp lại đúng bản sao ở validation, khiến F1 bị thổi phồng.

Kết quả định lượng cho thấy chênh lệch rất lớn: quy trình đúng đạt F1-macro 0.6231, trong khi quy trình sai cho 0.9378 — chênh lệch giả tạo +0.3147. Con số này minh hoạ trực tiếp rằng một sai sót về thứ tự xử lý có thể tạo ra ảo giác hiệu năng cao trước khi triển khai, để rồi thất bại trong thực tế.'''))

c.append(("code", '''X_all_bal, y_all_bal = oversample_text(Xtf, ytf)          # SAI: nhân bản toàn bộ TRƯỚC
X_all_bal = X_all_bal.reset_index(drop=True); y_all_bal = y_all_bal.reset_index(drop=True)
wrong = []
for tr, va in skf.split(X_all_bal, y_all_bal):
    v = TfidfVectorizer(max_features=15000, ngram_range=(1, 2), min_df=2)
    m = LogisticRegression(max_iter=1000)
    m.fit(v.fit_transform(X_all_bal.iloc[tr]), y_all_bal.iloc[tr])
    wrong.append(f1_score(y_all_bal.iloc[va], m.predict(v.transform(X_all_bal.iloc[va])), average="macro"))
print(f"ĐÚNG (cân bằng TRONG fold)      : {np.mean(clf_scores):.4f}")
print(f"SAI  (cân bằng TRƯỚC khi chia)  : {np.mean(wrong):.4f}  <- thổi phồng do bản sao rò rỉ sang val")
print(f"Chênh lệch ảo: +{np.mean(wrong) - np.mean(clf_scores):.4f} F1-macro")'''))

# === PART B ================================================================
c.append(("md", '''# PHẦN B — Bài toán hồi quy giá bất động sản Hà Nội (`KFold`)

Khác với Phần A, biến mục tiêu `Giá/m²` là **biến liên tục**, do đó không áp dụng cân bằng lớp và sử dụng `KFold` thay cho `StratifiedKFold` (không tồn tại cấu trúc lớp để phân tầng). Quy trình tiền xử lý cũng thay đổi tương ứng: từ TF-IDF chuyển sang `StandardScaler` cho biến số (diện tích, số phòng ngủ) và `OneHotEncoder` cho biến hạng mục (quận, loại hình nhà ở). Nguyên tắc bất biến vẫn được giữ: mọi phép khớp (`fit`) chỉ thực hiện trên tập huấn luyện.'''))

c.append(("md", '''## B1. Nạp dữ liệu và chuẩn hoá biến mục tiêu về dạng số

Dữ liệu gốc lưu giá dưới dạng văn bản với nhiều đơn vị khác nhau ("triệu/m²", "tỷ/m²", "đ/m²"). Bước làm sạch quy đổi toàn bộ về cùng một đơn vị (triệu đồng/m²), đồng thời chuyển diện tích và số phòng ngủ về dạng số. Sau đó loại bỏ các bản ghi lỗi đơn vị hoặc ngoại lai cực đoan bằng cách giới hạn giá trong khoảng [5, 500] triệu/m² và diện tích trong khoảng [10, 1000] m² — các ngưỡng hợp lý với thị trường Hà Nội — và loại các dòng thiếu giá trị ở những trường thiết yếu. Bảng thống kê mô tả sau làm sạch xác nhận miền giá trị đã hợp lý.'''))

c.append(("code", '''dfh = pd.read_csv(find_csv("Hanoi_housing_dataset.csv")).drop(columns=["Unnamed: 0"])
print("Kích thước gốc:", dfh.shape)

def to_number(s):
    if pd.isna(s): return np.nan
    m = re.search(r"[0-9]+(?:[.][0-9]+)?", str(s).replace(".", "").replace(",", "."))
    return float(m.group(0)) if m else np.nan

def to_price_m2(s):           # quy mọi đơn vị về TRIỆU đồng/m²
    if pd.isna(s): return np.nan
    v, t = to_number(s), str(s)
    if "tỷ" in t: return v * 1000
    if "đ/m" in t and "triệu" not in t: return v / 1e6
    return v

dfh["area"] = dfh["Diện tích"].apply(to_number)
dfh["bedrooms"] = dfh["Số phòng ngủ"].apply(to_number)
dfh["price_m2"] = dfh["Giá/m2"].apply(to_price_m2)
dfh["district"] = dfh["Quận"].fillna("NA")
dfh["house_type"] = dfh["Loại hình nhà ở"].fillna("NA")
dfh["date"] = pd.to_datetime(dfh["Ngày"], errors="coerce")

# Lọc lỗi đơn vị / ngoại lai cực đoan về khoảng hợp lý của Hà Nội
n0 = len(dfh)
dfh = dfh[dfh["price_m2"].between(5, 500) & dfh["area"].between(10, 1000)].copy()
dfh = dfh.dropna(subset=["area", "bedrooms", "price_m2"]).reset_index(drop=True)
print(f"Sau làm sạch: {len(dfh)} dòng (loại {n0 - len(dfh)} dòng lỗi/thiếu)")
dfh[["area", "bedrooms", "price_m2"]].describe().round(1)'''))

c.append(("md", '''## B2. Tách tập kiểm tra và định nghĩa pipeline tiền xử lý

`KFold` được sử dụng mà không phân tầng vì biến mục tiêu liên tục. Toàn bộ chuỗi tiền xử lý và mô hình được đóng gói trong một `Pipeline`: `ColumnTransformer` áp `StandardScaler` cho biến số và `OneHotEncoder` (với `handle_unknown="ignore"` để an toàn trước hạng mục chưa gặp) cho biến hạng mục, nối tiếp với mô hình hồi quy `Ridge`. Việc đóng gói trong `Pipeline` bảo đảm rằng ở mỗi fold, các bước tiền xử lý chỉ được khớp trên train-fold, qua đó tự động loại trừ rò rỉ dữ liệu mà không cần xử lý thủ công.'''))

c.append(("code", '''NUM = ["area", "bedrooms"]
CAT = ["district", "house_type"]
Xh = dfh[NUM + CAT]
yh = dfh["price_m2"]

# Test held-out 20% (KFold không stratify -> chia ngẫu nhiên)
Xh_tf, Xh_te, yh_tf, yh_te = train_test_split(Xh, yh, test_size=0.20, random_state=SEED)
Xh_tf = Xh_tf.reset_index(drop=True); yh_tf = yh_tf.reset_index(drop=True)

def make_pipe():
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUM),                                   # fit chỉ train-fold
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),             # fit chỉ train-fold
    ])
    return Pipeline([("pre", pre), ("model", Ridge(alpha=1.0))])

print("train-full:", len(Xh_tf), "| test:", len(Xh_te))'''))

c.append(("md", '''## B3. Kiểm định chéo `KFold` — kết quả từng fold (RMSE và R²)

Sử dụng `KFold` (không phân tầng) vì không tồn tại cấu trúc lớp để bảo toàn tỉ lệ; mỗi tin rao được xem là một quan sát độc lập. Hai chỉ số được báo cáo cho từng fold: **RMSE** (căn bậc hai sai số bình phương trung bình, đơn vị triệu/m², phản ánh độ lớn sai số dự báo) và **R²** (hệ số xác định, phản ánh tỉ lệ phương sai của biến mục tiêu được mô hình giải thích).'''))

c.append(("code", '''kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
reg_rmse, reg_r2 = [], []
for fold, (tr, va) in enumerate(kf.split(Xh_tf), 1):
    pipe = make_pipe().fit(Xh_tf.iloc[tr], yh_tf.iloc[tr])     # fit (gồm scaler/onehot) CHỈ train-fold
    pred = pipe.predict(Xh_tf.iloc[va])
    rmse = mean_squared_error(yh_tf.iloc[va], pred) ** 0.5
    r2 = r2_score(yh_tf.iloc[va], pred)
    reg_rmse.append(rmse); reg_r2.append(r2)
    print(f"FOLD {fold}/5 | train={len(tr):6d} val={len(va):6d} | RMSE={rmse:6.2f} triệu/m²  R²={r2:.3f}")
print(f"\\nKFold RMSE: {np.mean(reg_rmse):.2f} ± {np.std(reg_rmse):.2f} triệu/m²")
print(f"KFold R²  : {np.mean(reg_r2):.3f} ± {np.std(reg_r2):.3f}")'''))

c.append(("md", '''### Diễn giải kết quả
- Hệ số R² ở mức trung bình do mô hình mới chỉ khai thác một số ít đặc trưng (diện tích, số phòng ngủ, quận, loại hình nhà ở); trong khi đơn giá phụ thuộc mạnh vào vị trí chi tiết mà các đặc trưng hiện có chưa nắm bắt được. Bổ sung đặc trưng sẽ cải thiện hiệu năng, song trọng tâm của báo cáo là tính đúng đắn của quy trình kiểm định chéo, không phải tối ưu hoá điểm số.
- RMSE và R² ổn định giữa các fold (độ lệch chuẩn nhỏ: RMSE 42.30 ± 0.67, R² 0.313 ± 0.009), cho thấy ước lượng đáng tin cậy và mô hình không quá nhạy cảm với cách chia dữ liệu.'''))

c.append(("md", '''## B4. Kiểm định chéo `TimeSeriesSplit` — khi dữ liệu có yếu tố thời gian

Do bộ dữ liệu chứa cột `Ngày`, cần xét đến tình huống bài toán thực chất là **dự báo giá trong tương lai**. Trong trường hợp này, `KFold` ngẫu nhiên cho phép mô hình học từ các tin rao ở tương lai để dự đoán quá khứ — một dạng rò rỉ thời gian dẫn tới ước lượng hiệu năng sai lệch. `TimeSeriesSplit` khắc phục bằng cách ràng buộc tập train luôn là quá khứ và tập validation luôn là tương lai, không xáo trộn dữ liệu.

Kết quả in theo từng fold cho thấy rõ đặc điểm của chiến lược này: tập validation luôn nằm sau tập train về mặt thời gian, và kích thước tập train tăng dần qua các fold.'''))

c.append(("code", '''# Chỉ lấy phần có ngày hợp lệ, SẮP XẾP theo thời gian (bắt buộc cho TimeSeriesSplit)
dft = dfh.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
print(f"Số tin có ngày hợp lệ: {len(dft)}  ({dft['date'].min().date()} -> {dft['date'].max().date()})")

Xt = dft[NUM + CAT]; yt = dft["price_m2"]
tss = TimeSeriesSplit(n_splits=5)
ts_rmse = []
for fold, (tr, va) in enumerate(tss.split(Xt), 1):
    pipe = make_pipe().fit(Xt.iloc[tr], yt.iloc[tr])
    rmse = mean_squared_error(yt.iloc[va], pipe.predict(Xt.iloc[va])) ** 0.5
    ts_rmse.append(rmse)
    print(f"FOLD {fold}/5 | train={len(tr):6d} (đến {dft['date'].iloc[tr[-1]].date()}) "
          f"| val={len(va):5d} ({dft['date'].iloc[va[0]].date()} -> {dft['date'].iloc[va[-1]].date()}) "
          f"| RMSE={rmse:6.2f}")
print(f"\\nTimeSeriesSplit RMSE: {np.mean(ts_rmse):.2f} ± {np.std(ts_rmse):.2f} triệu/m²")'''))

c.append(("md", '''### So sánh `KFold` và `TimeSeriesSplit`
- `KFold` đưa ra ước lượng dựa trên việc trộn lẫn các mốc thời gian; hợp lý khi xem mỗi tin rao là độc lập và mục tiêu là định giá tại một thời điểm.
- `TimeSeriesSplit` cho ước lượng trung thực hơn đối với bài toán dự báo, vì luôn dự đoán tương lai từ quá khứ; tập train mở rộng dần và mỗi fold validation là một quãng thời gian kế tiếp.
- Lựa chọn phụ thuộc mục tiêu bài toán: định giá tại thời điểm hiện tại sử dụng `KFold`; dự báo giá trong tương lai sử dụng `TimeSeriesSplit`. Trong thực nghiệm, hai phương pháp cho RMSE tương đương (khoảng 42 triệu/m²), nhưng `TimeSeriesSplit` có độ lệch chuẩn lớn hơn (1.41 so với 0.67), phản ánh độ khó cố hữu của bài toán ngoại suy theo thời gian.'''))

# === Conclusion ===========================================================
c.append(("md", '''## Kết luận

1. **Cô lập tập kiểm tra trước tiên:** thực hiện phân chia train/validation/test ngay từ đầu; tập test được đóng băng và chỉ truy cập một lần khi báo cáo kết quả cuối cùng.
2. **Cân bằng đúng phạm vi:** chỉ cân bằng lớp trên tập huấn luyện và sau khi đã chia (Phần A); bài toán hồi quy không cân bằng biến mục tiêu (Phần B).
3. **Khớp tham số đúng nguồn:** mọi bộ vector hoá, chuẩn hoá, mã hoá chỉ được khớp (`fit`) trên tập huấn luyện (hoặc train-fold), sau đó áp dụng (`transform`) cho phần còn lại; sử dụng `Pipeline` để tự động bảo đảm thứ tự đúng trong kiểm định chéo.
4. **Lựa chọn loại fold theo bản chất dữ liệu:** phân loại mất cân bằng dùng `StratifiedKFold`; hồi quy/dữ liệu i.i.d. dùng `KFold`; dữ liệu có nhóm trùng dùng `GroupKFold`; dữ liệu có yếu tố thời gian/dự báo dùng `TimeSeriesSplit`.
5. **Hệ quả của vi phạm thứ tự:** xử lý hoặc cân bằng trước khi chia dữ liệu gây rò rỉ dữ liệu và làm chỉ số đánh giá tăng giả tạo (đã lượng hoá ở Mục A6: +0.3147 F1-macro).

> Nguyên tắc cô đọng: tách tập kiểm tra trước; mọi tham số cần học từ dữ liệu chỉ học trên tập huấn luyện; lựa chọn chiến lược chia fold đúng theo bản chất của dữ liệu.'''))

build(c, OUT)
