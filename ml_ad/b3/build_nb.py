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
c.append(("md", '''# Chia train/val/test, cân bằng & Cross-Validation — trên 2 dataset thật

Dùng đúng **2 file CSV** trong zip (`ViHSD.csv`, `Hanoi_housing_dataset.csv`).

| | PHẦN A — Phân loại | PHẦN B — Hồi quy |
|---|---|---|
| Dataset | ViHSD (bình luận MXH) | Giá nhà Hà Nội |
| Target | 3 lớp `CLEAN/OFFENSIVE/HATE` (mất cân bằng) | `Giá/m²` (số liên tục) |
| Cân bằng | Có (oversample train) | Không (hồi quy) |
| **Loại Fold** | **StratifiedKFold** | **KFold** (+ TimeSeriesSplit) |

> **Quy tắc vàng:** mọi bước **học tham số từ dữ liệu** (`fit` vectorizer/scaler/encoder, cân bằng bằng nhân bản) **chỉ được nhìn TRAIN**. `val`/`test` chỉ đi qua `transform`. Lý do: val/test phải đóng vai "dữ liệu chưa từng thấy" → nếu cho chúng tham gia `fit`, điểm đánh giá **đẹp ảo** (data leakage), ra thực tế là sập.'''))

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
c.append(("md", '''## 0. Nên dùng loại Fold nào? — và VÌ SAO

| Loại Fold | Dùng khi | Vì sao |
|---|---|---|
| **KFold** | Hồi quy / dữ liệu i.i.d., không có lớp | Chia ngẫu nhiên K khối bằng nhau — đơn giản, không thiên lệch khi dữ liệu độc lập. |
| **StratifiedKFold** | **Phân loại**, nhất là **mất cân bằng** | Giữ **tỉ lệ lớp** ở mỗi fold. Nếu dùng KFold thường, fold có thể **thiếu lớp hiếm** → điểm dao động mạnh, ước lượng sai. |
| **GroupKFold** | Có **nhóm/thực thể trùng** (cùng user, cùng căn nhà, cùng phiên) | Giữ cả nhóm về **một phía** → tránh model "thấy" nhóm đó ở cả train lẫn val (leakage theo nhóm). |
| **TimeSeriesSplit** | Dữ liệu có **thời gian**, bài forecasting | Train = **quá khứ**, val = **tương lai**, **không shuffle**. KFold ngẫu nhiên sẽ cho model "thấy tương lai" → leakage thời gian. |

**Áp dụng cho 2 dataset ở đây:**
- **ViHSD** → `StratifiedKFold`. *Vì sao:* lớp `OFFENSIVE` chỉ ~6.8%; KFold thường dễ tạo fold lệch/thiếu lớp này → F1 nhảy loạn. Stratified giữ đúng tỉ lệ 3 lớp ở mọi fold.
- **Hanoi housing** → `KFold` (target liên tục, không có lớp để phân tầng). *Nhưng:* dataset có cột **`Ngày`** → nếu mục tiêu là **dự đoán giá tương lai**, đúng hơn phải dùng `TimeSeriesSplit` (xem cuối Phần B).'''))

# === PART A ================================================================
c.append(("md", '''# PHẦN A — Phân loại ViHSD (StratifiedKFold)'''))

c.append(("md", '''## A1. Nạp & làm sạch
Bỏ bình luận trùng lặp — nếu để trùng, bản sao có thể rơi vào cả train lẫn test (một dạng leakage).'''))

c.append(("code", '''df = pd.read_csv(find_csv("ViHSD.csv"))
print("Kích thước gốc:", df.shape)
df = df.drop_duplicates(subset="free_text").reset_index(drop=True)
print("Sau khi bỏ trùng:", df.shape)
show_dist(df["label_id"], "TOÀN BỘ")
# File có sẵn cột split (data thực tế đôi khi đã chia sẵn) — ở đây ta tự chia để học:
print("Cột split có sẵn:", df["split"].value_counts().to_dict())
df.head(3)'''))

c.append(("md", '''## A2. Bước 1 — Chia `test` (held-out) + `train`/`val`, có `stratify`
`train_test_split` chỉ tách 2 phần/lần → cắt **2 lần**. `stratify=y` để mọi tập giữ đúng tỉ lệ 3 lớp.
*Vì sao chia trước:* để val/test thật sự "chưa từng thấy"; mọi xử lý sau chỉ học trên train.'''))

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

c.append(("md", '''## A3. Bước 2 — Cân bằng CHỈ trên `train`
Oversample (nhân bản) lớp thiểu số cho bằng lớp đa số — **chỉ train**.
*Vì sao chỉ train:* `val/test` phải giữ phân phối gốc thì F1 mới trung thực; cân bằng cả val/test = bóp méo thực tế cần đo, và là leakage (sinh mẫu trên dữ liệu đáng lẽ chưa thấy).'''))

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

c.append(("md", '''## A4. Bước 3–5 — Vector hoá (fit train) → huấn luyện → đánh giá
`TfidfVectorizer` học từ vựng (`fit`) **chỉ trên train đã cân bằng**, rồi `transform` val/test. Đây chính là `fit_transform(train)` vs `transform(val/test)`.'''))

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

c.append(("md", '''## A5. `StratifiedKFold` — IN RA TỪNG FOLD
Thay 1 tập val cố định bằng **K khối**. Mỗi vòng: 1 fold làm val, K−1 fold còn lại làm train; lặp K lần → **mỗi mẫu được validate đúng 1 lần**.
*Vì sao Stratified ở đây:* lớp `OFFENSIVE` hiếm → cần giữ tỉ lệ ở mỗi fold, nếu không có fold thiếu lớp này.
**Quan trọng:** cân bằng + fit vectorizer đặt **trong** vòng lặp, **chỉ** trên train-fold; `test` vẫn held-out.'''))

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

c.append(("md", '''### Giải thích từng fold
- Mỗi vòng đổi khối val khác nhau → sau 5 vòng **mọi mẫu đều từng ở val đúng 1 lần**.
- **Tỉ lệ lớp in ra ở mỗi fold gần như nhau** → tác dụng của `Stratified` (lý do chọn nó cho dữ liệu lệch lớp).
- **`mean ± std`:** trung bình = ước lượng ổn định hơn 1 lần chia; **std** cho biết model nhạy với cách chia tới đâu (std lớn = variance cao, chưa ổn định).
- Cân bằng + fit vectorizer **trong** fold → val-fold luôn sạch, không leakage.'''))

c.append(("md", '''## A6. Làm SAI thứ tự → leakage (đo cụ thể)
Cân bằng (nhân bản) **TRƯỚC khi chia fold**: bản sao giống hệt rơi vào cả train-fold lẫn val-fold → model học thuộc → F1 thổi phồng.'''))

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
c.append(("md", '''# PHẦN B — Hồi quy giá nhà Hà Nội (KFold)

Khác Phần A: target `Giá/m²` **liên tục** → **không cân bằng lớp**, và dùng **`KFold`** (không Stratified, vì không có lớp). Tiền xử lý đổi từ TF-IDF sang **`StandardScaler`** (số) + **`OneHotEncoder`** (Quận, loại hình) — vẫn `fit` **chỉ trên train**.'''))

c.append(("md", '''## B1. Nạp & làm sạch (chuyển text "triệu/m²", "tỷ/m²"... về số)'''))

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

c.append(("md", '''## B2. Chia train/test + định nghĩa pipeline tiền xử lý
`KFold` **không** stratify (vì target liên tục). Pipeline = `ColumnTransformer` (`StandardScaler` cho số + `OneHotEncoder` cho Quận/loại hình) → `Ridge`. Đặt trong **Pipeline** để mỗi fold `fit` preprocessing **chỉ trên train-fold** (chống leakage tự động).'''))

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

c.append(("md", '''## B3. `KFold` — IN RA TỪNG FOLD (RMSE + R²)
*Vì sao KFold (không Stratified):* không có "lớp" để giữ tỉ lệ; mỗi tin nhà coi như độc lập. Báo cáo **RMSE** (sai số, triệu/m²) và **R²** (mức giải thích phương sai) cho từng fold.'''))

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

c.append(("md", '''### Giải thích
- R² ở mức vừa phải vì mới dùng vài đặc trưng (diện tích, phòng ngủ, quận, loại hình); **giá/m² phụ thuộc nhiều vào vị trí chi tiết** — thêm đặc trưng sẽ cải thiện. Trọng tâm ở đây là **quy trình CV**, không phải tối ưu điểm.
- RMSE/R² **ổn định giữa các fold** (std nhỏ) → ước lượng đáng tin, model không quá nhạy với cách chia.'''))

c.append(("md", '''## B4. `TimeSeriesSplit` — vì dataset có cột `Ngày`
*Vì sao cần xét:* nếu bài toán là **dự đoán giá tương lai**, dùng KFold ngẫu nhiên sẽ cho model học từ tin **tương lai** để đoán **quá khứ** → leakage thời gian, điểm ảo. `TimeSeriesSplit` ép **train = quá khứ, val = tương lai**, không shuffle.

In ra từng fold sẽ thấy: val-fold luôn nằm **sau** train-fold về thời gian, và train lớn dần.'''))

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

c.append(("md", '''### So sánh KFold vs TimeSeriesSplit
- KFold: ước lượng "trộn thời gian" — hợp lý nếu coi mỗi tin **độc lập** (bài định giá tại 1 thời điểm).
- TimeSeriesSplit: ước lượng **trung thực hơn cho forecasting** vì luôn đoán tương lai từ quá khứ. Train tăng dần, mỗi fold val là một quãng thời gian kế tiếp.
- **Chọn cái nào?** Theo *mục tiêu*: định giá hiện tại → KFold; dự báo giá tương lai → **TimeSeriesSplit**.'''))

# === Conclusion ===========================================================
c.append(("md", '''## Kết luận chung

1. **CHIA TRƯỚC** mọi thứ: `train/val/test`; `test` khoá lại, chạm 1 lần.
2. **Cân bằng** chỉ trên **train**, **sau** khi chia (Phần A). Hồi quy không cân bằng lớp (Phần B).
3. **`fit`** vectorizer/scaler/encoder **chỉ trên train** (hoặc train-fold), rồi `transform` phần còn lại. Dùng **Pipeline** để tự động đúng thứ tự trong CV.
4. **Chọn loại Fold theo bài toán — và biết lý do:**
   - Phân loại mất cân bằng → **StratifiedKFold** (giữ tỉ lệ lớp).
   - Hồi quy / i.i.d. → **KFold**.
   - Có nhóm trùng → **GroupKFold**. Có thời gian / forecasting → **TimeSeriesSplit**.
5. Sai thứ tự (xử lý/cân bằng trước khi chia) → **data leakage** → điểm đẹp ảo (đo ở A6).

> Một câu: **"Tách test ra trước đã; cái gì cần học từ dữ liệu thì chỉ học trên train; chọn Fold theo đúng bản chất dữ liệu."**'''))

build(c, OUT)
