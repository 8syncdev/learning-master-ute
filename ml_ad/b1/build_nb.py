# -*- coding: utf-8 -*-
"""Assemble two Vietnamese EDA notebooks for social-app / VN-market modelling.

- Classification (NLP, tiếng Việt): ViHSD -> hate speech (CLEAN / OFFENSIVE / HATE)
  from Facebook/YouTube/Instagram/TikTok comments.
- Regression (thị trường VN): Hanoi real-estate -> predict price (triệu/m2).
"""
import os
import nbformat as nbf

BASE = "2611328 - Nguyễn Phương Anh Tú - Exploratory Data Analysis (EDA)"


def build(cells, path):
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    }
    nb.cells = [
        nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src)
        for kind, src in cells
    ]
    nbf.write(nb, path)
    print("wrote", path, "with", len(nb.cells), "cells")


# Vietnamese stopword syllables (stopwords-iso/vi), embedded for offline run
VN_STOP = "a a-lô ai alô amen anh bao bay biến biết bà bài bác bây bèn béng bông bõm bạn bản bất bấy bẩy bập bắt bằng bển bệt bị bỏ bỗng bộ bội bớ bởi bức cao cha chao chi chiếc cho choa chu chui chung chuyện chà chành chí chính chót chùn chú chúng chăn chăng chũn chơi chưa chưng chạnh chả chầm chầy chậc chập chắc chắn chẳng chết chỉ chỉn chốc chớ chợt chủn chứ chừ chừng coi con cu cuối cuốn cuộc càng các cách cái cây còn có cóc cô công cùng căn cũng cơ cơn cả cảm cần cật cậu cắt cổ cục của cứ cực da do duy dà dào dì dù dĩ dưng dưới dạ dần dầu dẫu dễ dịp dở dữ em gian giác giờ giời giữa gì ha hay hoàn hoặc hèn hình hô hơn hầu hậu hẳn hết họ hỏi hồ hự khi khác khói khô không khắc kia kê kì kìa kể kỳ lai le liệt loại loạt luôn luận luật luốt là làm lâu lên lình lí lúc lý lại lần lập lắm lẽ lị lớn lự lực muốn mà mày mình mòi mù mạng mấy mẹ mỗi một mới mợ mực nay ngay nghe nghen nghiễm nghỉm ngoài ngoải ngày ngôi ngõ ngăn ngươi người ngắt ngọn ngọt ngộ nh nhau nhiên nhiêu nhiều nhiệt nhung nhà nhân nhé nhén nhón nhăng như nhưng nhược nhất nhận nhỉ nhỡ những nào này nên nó nóc nói năm nơi nả nấy nếu nền nọ nỗi nớ nở nức nữa oai oái phi pho phui phàm phè phóc phót phăn phương phải phần phắt phết phỉ phỏng phốc phụt phứt qua qui quy quyết quyển quá quít quý quýt quả ra ren riu riêng riệt rày ráo rén rích ríu rón rút răng rất rằng rốt rồi rứa sa sao sau sinh so song suýt sá sì sạch sả sất sắp sẽ số sốt sột sở sợ sức sự ta tang tanh te tha than thanh theo thi thiên thiết thoạt thoảng thoắt thuần thà thành thái tháng tháo thì thình thím thôi thúng thương thường thảo thảy thấy thẩy thậm thật thắng thế thếch thể thỉnh thị thỏm thốc thốt thộc thời thục thử thửa thực tiên tiếp tiện tiệt toà toàn toé toẹt trong trung tráo trên trò trước trạo trếu trển trệt trệu trọi trỏng trời trừ tuy tuyệt tuần tuốt tuồn tuồng tuột tà tàn tán tâm tê tênh tì tình tít tò tôi tông tù tăm tại tả tấm tấn tất tần tật tắp tề tọt tỏ tốc tối tột tớ tới tức từ từng tử tự tựu veo việc vung và vàn vào vâng vèo vì ví vô văng vạn vả vẫn vậy vẻ về vị vốn với vở vụt vừa xa xiết xon xoành xoét xoạch xoẳn xoẹt xuất xuể xuống xá xón xúi xăm xưa xả xắm xềnh xệch xệp xửa à ào á ái áng âu ô ôi ông úi ý đang đi điều đành đán đáng đánh đáo đâu đây đã đó đùng đúng được đạch đại đất đấy đầu đến đều để địa định đồ đỗi độ ơ ơi ơn ư ạ ấy ầu ắt ối ồ ổng ớ ờ ở ủa ứ ừ ử"

NLP_SETUP = '''# Stopwords tiếng Việt (nhúng sẵn để chạy offline) + hàm đếm từ
STOPWORDS = set("""%s""".split())

# Bổ sung một số từ viết tắt/teencode rất phổ biến trên mạng xã hội
STOPWORDS |= {"ko", "k", "kh", "dc", "đc", "vs", "ng", "z", "j", "ji", "r", "mn", "ad"}

import re as _re
# Bắt cụm chữ cái tiếng Việt (âm tiết); bỏ emoji, số, dấu câu
_TOKEN = _re.compile(r"[a-zà-ỹ]+", _re.IGNORECASE)

def word_counts(texts):
    from collections import Counter
    c = Counter()
    for tx in texts:
        for w in _TOKEN.findall(str(tx).lower()):
            if len(w) > 1 and w not in STOPWORDS:
                c[w] += 1
    return c''' % VN_STOP


# =====================================================================
# 1. CLASSIFICATION (NLP, tiếng Việt) — ViHSD hate speech
# =====================================================================
clf = []

clf.append(("code", '''# Import các thư viện cần thiết cho EDA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 200)'''))

clf.append(("md", '''# GIỚI THIỆU DATASET — ViHSD (Vietnamese Hate Speech Detection)

**ViHSD** là bộ dữ liệu chuẩn của nhóm UIT-NLP, gồm hơn **33.000 bình luận tiếng Việt** được thu thập từ các nền tảng mạng xã hội như **Facebook, YouTube, Instagram và TikTok**. Đây là bài toán **NLP — phân loại** phục vụ kiểm duyệt nội dung (content moderation) cho các ứng dụng mạng xã hội ở thị trường Việt Nam.

**Đặc trưng đầu vào:**
- `free_text`: nội dung bình luận của người dùng (văn bản tiếng Việt, có teencode, emoji)

**Biến mục tiêu là `label_id`, gồm 3 nhãn:**
- `0` — **CLEAN**: bình luận bình thường, không thù ghét
- `1` — **OFFENSIVE**: bình luận có tính xúc phạm
- `2` — **HATE**: bình luận thù ghét

Đây là bài toán **classification** (đa lớp). Mục tiêu EDA là hiểu phân bố nhãn, đặc điểm độ dài và từ vựng theo nhãn, để chuẩn bị cho mô hình NLP/LLM tiếng Việt (ví dụ fine-tune **PhoBERT**) khi xây dựng tính năng lọc bình luận cho app.'''))

clf.append(("code", '''# Đọc dataset ViHSD từ file CSV
df = pd.read_csv("ViHSD.csv")

# Gán tên nhãn cho dễ đọc
label_map = {0: "CLEAN", 1: "OFFENSIVE", 2: "HATE"}
df["label_name"] = df["label_id"].map(label_map)

print("5 dòng đầu tiên")
df.head()'''))

clf.append(("code", '''# Xem 5 dòng cuối
df.tail()'''))

clf.append(("code", '''# Kiểm tra số dòng và số cột
print("Kích thước dataset:")
print(df.shape)
print("\\nDataset có", df.shape[0], "dòng và", df.shape[1], "cột.")'''))

clf.append(("code", '''# Thông tin tổng quan
df.info()'''))

clf.append(("code", '''# Danh sách cột
df.columns'''))

clf.append(("code", '''# Kiểm tra giá trị thiếu
print(df.isnull().sum())
print("\\nTổng missing:", df.isnull().sum().sum())'''))

clf.append(("md", '''Nhận xét:

Cột bình luận `free_text` và nhãn `label_id` về cơ bản **không có giá trị thiếu** (các dòng rỗng đã được loại khi chuẩn bị dữ liệu). Vì vậy ta có thể phân tích trực tiếp.'''))

clf.append(("code", '''# Kiểm tra bình luận bị trùng lặp
print("Số bình luận trùng lặp:", df["free_text"].duplicated().sum())'''))

clf.append(("md", '''Nhận xét:

Có một số bình luận **trùng lặp** (ví dụ các bình luận spam giống nhau). Ta loại bỏ để tránh rò rỉ dữ liệu giữa train/test và làm sai lệch thống kê tần suất.'''))

clf.append(("code", '''# Loại bỏ bình luận trùng lặp
df = df.drop_duplicates(subset="free_text").reset_index(drop=True)
print("Kích thước sau khi loại trùng:", df.shape)'''))

clf.append(("code", '''# Phân bố nhãn
print("Số lượng theo nhãn:")
print(df["label_name"].value_counts())
print("\\nTỷ lệ phần trăm:")
print((df["label_name"].value_counts(normalize=True) * 100).round(2))'''))

clf.append(("code", '''# Countplot phân bố nhãn
plt.figure(figsize=(7, 4))
sns.countplot(data=df, x="label_name", order=["CLEAN", "OFFENSIVE", "HATE"])
plt.title("Phân bố nhãn bình luận (ViHSD)")
plt.xlabel("Nhãn")
plt.ylabel("Số lượng")
plt.show()'''))

clf.append(("md", '''### Nhận xét Countplot

Dataset **mất cân bằng nghiêm trọng**: nhãn **CLEAN chiếm ~82.7%**, trong khi **HATE (~10.5%)** và **OFFENSIVE (~6.8%)** rất ít. Đây là đặc điểm điển hình của dữ liệu thực tế — đa số bình luận là bình thường. Khi huấn luyện mô hình cần xử lý mất cân bằng (class weight, oversampling, hoặc dùng F1-macro thay vì accuracy).'''))

clf.append(("md", '''## Đặc trưng độ dài bình luận'''))

clf.append(("code", '''# Tạo đặc trưng độ dài
df["n_words"] = df["free_text"].str.split().str.len()
df["n_chars"] = df["free_text"].str.len()
df[["n_words", "n_chars"]].describe()'''))

clf.append(("md", '''### Nhận xét độ dài

Bình luận khá ngắn: trung vị khoảng **8 từ**, nhưng phân bố **lệch phải rất mạnh** (có bình luận dài tới hơn 1.700 từ). Độ dài ngắn + nhiều teencode là thách thức cho mô hình NLP tiếng Việt.'''))

clf.append(("code", '''# Histogram số từ theo nhãn (giới hạn trục x để dễ nhìn)
plt.figure(figsize=(9, 5))
sns.histplot(data=df, x="n_words", hue="label_name",
             hue_order=["CLEAN", "OFFENSIVE", "HATE"], bins=60, element="step")
plt.xlim(0, 80)
plt.title("Phân bố độ dài bình luận theo nhãn")
plt.xlabel("Số từ")
plt.ylabel("Tần suất")
plt.show()'''))

clf.append(("code", '''# Boxplot độ dài theo nhãn
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="label_name", y="n_words",
            order=["CLEAN", "OFFENSIVE", "HATE"], showfliers=False)
plt.title("Boxplot số từ theo nhãn (ẩn outlier để dễ so sánh)")
plt.xlabel("Nhãn")
plt.ylabel("Số từ")
plt.show()'''))

clf.append(("md", '''### Nhận xét độ dài theo nhãn

Bình luận **HATE dài hơn hẳn** (trung bình ~20 từ) so với **CLEAN (~10.5 từ)** và **OFFENSIVE (~10.7 từ)**. Điều này cho thấy các bình luận thù ghét thường mang tính "chửi rủa, công kích dài dòng", trong khi bình luận xúc phạm thường ngắn gọn. Như vậy **độ dài cũng là một đặc trưng có ích** (dù không đủ một mình) cho việc phân loại.'''))

clf.append(("code", '''# Đếm outlier độ dài bằng IQR
for col in ["n_words", "n_chars"]:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    out = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
    print(f"{col}: {len(out)} outliers")'''))

clf.append(("code", '''# Tương quan giữa độ dài và nhãn (mã hóa số)
num_cols = ["n_words", "n_chars", "label_id"]
plt.figure(figsize=(6, 5))
sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Tương quan giữa độ dài và nhãn")
plt.show()'''))

clf.append(("md", '''### Nhận xét Heatmap

`n_words` và `n_chars` gần như **trùng nhau** (~1.0). Tương quan giữa độ dài và `label_id` ở mức **thấp–trung bình dương**, phù hợp với nhận xét bình luận HATE (nhãn 2) thường dài hơn. Tín hiệu phân loại chính vẫn nằm ở **nội dung từ ngữ**, không phải độ dài.'''))

clf.append(("md", '''## Phân tích từ khóa theo nhãn (NLP tiếng Việt)

So sánh các âm tiết/từ xuất hiện nhiều nhất ở nhóm **CLEAN** và nhóm tiêu cực **HATE + OFFENSIVE**.
*Lưu ý: dữ liệu thù ghét chứa ngôn từ thô tục — đây là bản chất bài toán kiểm duyệt nội dung.*'''))

clf.append(("code", NLP_SETUP))

clf.append(("code", '''# Đếm tần suất từ cho nhóm CLEAN và nhóm tiêu cực (HATE + OFFENSIVE)
clean_top = word_counts(df[df["label_id"] == 0]["free_text"]).most_common(15)
toxic_top = word_counts(df[df["label_id"].isin([1, 2])]["free_text"]).most_common(15)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, top, title, color in [
    (axes[0], clean_top, "Nhóm CLEAN", "#2a9d8f"),
    (axes[1], toxic_top, "Nhóm HATE + OFFENSIVE", "#e76f51"),
]:
    words = [w for w, _ in top][::-1]
    freqs = [c for _, c in top][::-1]
    ax.barh(words, freqs, color=color)
    ax.set_title(f"Top 15 từ — {title}")
    ax.set_xlabel("Tần suất")
plt.tight_layout()
plt.show()'''))

clf.append(("md", '''### Nhận xét Từ khóa

- Nhóm **CLEAN** xoay quanh các từ trung tính: tên riêng (nguyễn, trần, minh), từ lịch sự (xin, nha), chủ đề đời thường (thầy, chị, ăn, xem).
- Nhóm **HATE + OFFENSIVE** nổi bật với **từ ngữ công kích, thô tục** (thằng, ngu, chó, bọn, đéo, chửi, lồn...) và các từ mang tính khái quát/kỳ thị (dân, nước).

Khác biệt từ vựng này rất rõ, cho thấy đặc trưng dạng **n-gram / TF-IDF** hoặc **embedding từ mô hình ngôn ngữ tiếng Việt (PhoBERT)** sẽ rất hiệu quả để phát hiện bình luận độc hại.'''))

clf.append(("md", '''## Kết luận EDA

- Bài toán **NLP — phân loại bình luận độc hại tiếng Việt** (CLEAN / OFFENSIVE / HATE) trên dữ liệu mạng xã hội (FB/YouTube/Instagram/TikTok) — đúng nhu cầu kiểm duyệt nội dung cho app ở thị trường Việt.
- Dữ liệu **mất cân bằng nặng** (CLEAN ~83%) → cần class weight / F1-macro khi mô hình hóa.
- Bình luận **HATE dài hơn** rõ rệt; độ dài là đặc trưng phụ hữu ích.
- Từ vựng phân biệt rất mạnh giữa nhóm sạch và nhóm độc hại.

**Hướng tiếp theo (LLM):** tiền xử lý (chuẩn hóa teencode, tách từ bằng VnCoreNLP/pyvi) → baseline TF-IDF + Logistic Regression → fine-tune **PhoBERT** để triển khai bộ lọc bình luận cho ứng dụng mạng xã hội Việt Nam.'''))

build(clf, os.path.join(BASE, "EDA_ViHSD_HateSpeech.ipynb"))


# =====================================================================
# 2. REGRESSION (thị trường VN) — Hanoi real-estate price
# =====================================================================
reg = []

reg.append(("code", '''# Import các thư viện cần thiết cho EDA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", 120)'''))

reg.append(("md", '''# GIỚI THIỆU DATASET — Giá nhà đất Hà Nội

Dataset gồm hơn **82.000 tin rao bán nhà đất tại Hà Nội**, được thu thập từ **batdongsan.com.vn** (năm 2020). Đây là bài toán **regression** phục vụ xây dựng ứng dụng **định giá bất động sản cho thị trường Việt Nam**.

**Các đặc trưng đầu vào tiêu biểu:**
- `Quận`, `Huyện`, `Địa chỉ`: vị trí bất động sản
- `Loại hình nhà ở`: nhà ngõ/hẻm, nhà mặt phố, nhà phố liền kề, biệt thự
- `Giấy tờ pháp lý`: tình trạng sổ
- `Số tầng`, `Số phòng ngủ`, `Diện tích`, `Dài`, `Rộng`

**Biến mục tiêu là `Giá/m2`** (giá mỗi mét vuông). Đây là dữ liệu **thô, chưa sạch** (giá ghi dạng "86,96 triệu/m²", diện tích "46 m²", phòng ngủ "5 phòng", lẫn nhiều đơn vị khác nhau) nên cần **tiền xử lý** trước khi phân tích — đúng tinh thần dữ liệu thực tế ở thị trường Việt.'''))

reg.append(("code", '''# Đọc dataset và bỏ cột chỉ số thừa
df = pd.read_csv("Hanoi_housing_dataset.csv").drop(columns=["Unnamed: 0"])
df.head()'''))

reg.append(("code", '''# Xem 5 dòng cuối
df.tail()'''))

reg.append(("code", '''# Danh sách cột
df.columns'''))

reg.append(("code", '''# Kích thước và thông tin tổng quan
print("Kích thước:", df.shape)
df.info()'''))

reg.append(("code", '''# Kiểm tra giá trị thiếu ở từng cột
miss = df.isnull().sum().sort_values(ascending=False)
print(miss)
print("\\nTỷ lệ thiếu (%):")
print((miss / len(df) * 100).round(1))'''))

reg.append(("md", '''### Nhận xét missing values

- `Dài` (~76%) và `Rộng` (~57%) thiếu rất nhiều → sẽ **loại bỏ**.
- `Số tầng` thiếu ~56%, `Giấy tờ pháp lý` thiếu ~35% → giữ lại nhưng lưu ý khi mô hình hóa.
- Các cột quan trọng `Diện tích`, `Số phòng ngủ`, `Giá/m2` gần như đầy đủ.'''))

reg.append(("md", '''## Tiền xử lý: chuyển dữ liệu text về dạng số'''))

reg.append(("code", '''# Hàm trích số từ chuỗi (xử lý dấu chấm phần nghìn và dấu phẩy thập phân kiểu VN)
def to_number(s):
    if pd.isna(s):
        return np.nan
    t = str(s).replace(".", "").replace(",", ".")
    m = re.search(r"[0-9]+(?:[.][0-9]+)?", t)
    return float(m.group(0)) if m else np.nan

# Giá/m2 có nhiều đơn vị: "triệu/m²", "đ/m²", "tỷ/m²" -> quy về TRIỆU đồng/m²
def to_price_m2(s):
    if pd.isna(s):
        return np.nan
    v = to_number(s)
    s = str(s)
    if "tỷ" in s:
        return v * 1000        # 1 tỷ = 1000 triệu
    if "đ/m" in s and "triệu" not in s:
        return v / 1e6         # đồng -> triệu
    return v                    # đã là triệu/m²

df["area"] = df["Diện tích"].apply(to_number)        # diện tích (m2)
df["bedrooms"] = df["Số phòng ngủ"].apply(to_number)  # số phòng ngủ
df["floors"] = pd.to_numeric(df["Số tầng"], errors="coerce")  # số tầng
df["price_m2"] = df["Giá/m2"].apply(to_price_m2)      # giá (triệu/m2)

# Bỏ các cột thiếu quá nhiều / không dùng trực tiếp
df = df.drop(columns=["Dài", "Rộng"])

df[["area", "bedrooms", "floors", "price_m2"]].describe()'''))

reg.append(("md", '''### Nhận xét tiền xử lý

Sau khi tách số, ta thấy dữ liệu **lẫn lỗi đơn vị** rất mạnh: có dòng `Giá/m2` lên tới hàng trăm triệu *tỷ*/m² (do ghi sai đơn vị "tỷ/m²") và diện tích tới hơn 100.000 m². Những giá trị này là **lỗi nhập liệu**, làm méo toàn bộ thống kê. Cần lọc về khoảng hợp lý của thị trường Hà Nội.'''))

reg.append(("code", '''# Lọc về khoảng giá trị hợp lý của thị trường Hà Nội
n_before = len(df)
df = df[df["price_m2"].between(5, 500) & df["area"].between(10, 1000)].copy()

# Tạo thêm biến tổng giá trị căn nhà (tỷ đồng) = giá/m2 * diện tích
df["price_total_ty"] = df["price_m2"] * df["area"] / 1000

print("Số dòng trước lọc:", n_before)
print("Số dòng sau lọc:", len(df))
print("Đã loại:", n_before - len(df), "dòng lỗi/ngoại lai cực đoan")'''))

reg.append(("md", '''### Nhận xét lọc outlier

Chỉ loại bỏ khoảng **1.400 dòng** lỗi đơn vị / ngoại lai cực đoan, giữ lại hơn **81.000 tin** hợp lệ. Sau khi lọc, giá/m² tập trung trong khoảng thực tế của Hà Nội.'''))

reg.append(("code", '''# Kiểm tra trùng lặp
print("Số dòng trùng lặp hoàn toàn:", df.duplicated().sum())'''))

reg.append(("code", '''# Thống kê mô tả sau khi làm sạch
df[["area", "bedrooms", "floors", "price_m2", "price_total_ty"]].describe()'''))

reg.append(("md", '''Đây là bài toán **regression** vì biến mục tiêu `price_m2` (giá mỗi m²) là giá trị số liên tục. Ta dùng **histogram** để xem phân bố biến mục tiêu (thay vì countplot như bài phân loại).'''))

reg.append(("md", '''### Histogram của biến mục tiêu (giá/m²)'''))

reg.append(("code", '''plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="price_m2", kde=True, bins=50)
plt.title("Phân bố giá mỗi m² (triệu đồng/m²) - Hà Nội")
plt.xlabel("Giá (triệu/m²)")
plt.ylabel("Tần suất")
plt.show()'''))

reg.append(("md", '''### Nhận xét Histogram biến mục tiêu

Giá/m² tập trung quanh **trung vị ~90 triệu/m²** (trung bình ~100), phân bố **lệch phải**: phần lớn nhà ở mức 50–150 triệu/m², nhưng có những bất động sản ở khu trung tâm lên tới 400–500 triệu/m². Phân bố lệch gợi ý nên **biến đổi log** biến mục tiêu khi mô hình hóa.'''))

reg.append(("md", '''### Histogram các biến đầu vào'''))

reg.append(("code", '''feat_cols = ["area", "bedrooms", "floors", "price_total_ty"]
fig, axes = plt.subplots(2, 2, figsize=(15, 9))
axes = axes.flatten()
for i, col in enumerate(feat_cols):
    sns.histplot(data=df, x=col, kde=True, bins=40, ax=axes[i])
    axes[i].set_title(f"Histogram của {col}")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Tần suất")
axes[0].set_xlim(0, 200)
axes[3].set_xlim(0, 30)
plt.tight_layout()
plt.show()'''))

reg.append(("md", '''### Nhận xét Histogram biến đầu vào

- `area` (diện tích) **lệch phải mạnh**, đa số nhà 30–60 m² (đặc trưng nhà phố Hà Nội).
- `bedrooms` tập trung 3–5 phòng; `floors` tập trung 3–6 tầng.
- `price_total_ty` (tổng giá trị căn nhà) lệch phải, trung vị khoảng **3.5 tỷ**.'''))

reg.append(("md", '''### Boxplot biến mục tiêu và biến đầu vào'''))

reg.append(("code", '''fig, axes = plt.subplots(1, 3, figsize=(18, 4))
sns.boxplot(data=df, x="price_m2", ax=axes[0]); axes[0].set_title("Boxplot giá/m²")
sns.boxplot(data=df, x="area", ax=axes[1]); axes[1].set_title("Boxplot diện tích"); axes[1].set_xlim(0, 300)
sns.boxplot(data=df, x="price_total_ty", ax=axes[2]); axes[2].set_title("Boxplot tổng giá (tỷ)"); axes[2].set_xlim(0, 40)
plt.tight_layout()
plt.show()'''))

reg.append(("md", '''### Nhận xét Boxplot

Cả giá/m², diện tích và tổng giá đều còn **nhiều outlier ở phía trên** (bất động sản cao cấp/khu trung tâm). Đây là **giá trị thật của thị trường**, không phải lỗi, nên giữ lại; khi mô hình hóa có thể dùng log hoặc mô hình cây (Gradient Boosting) ít nhạy với outlier.'''))

reg.append(("md", '''### Correlation heatmap'''))

reg.append(("code", '''num_cols = ["price_m2", "area", "bedrooms", "floors", "price_total_ty"]
plt.figure(figsize=(8, 6))
sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Ma trận tương quan các biến số")
plt.show()'''))

reg.append(("md", '''### Nhận xét Heatmap

- `area` tương quan **mạnh** với tổng giá `price_total_ty` (~0.66) — nhà to thì tổng tiền cao (hiển nhiên).
- `bedrooms` và `floors` tương quan **dương mức trung bình** với `price_m2` (~0.27 và ~0.23): nhà nhiều phòng/tầng thường ở vị trí giá trị cao hơn.
- `area` tương quan **yếu** với `price_m2`: diện tích lớn không có nghĩa đơn giá cao — đơn giá phụ thuộc chủ yếu vào **vị trí** (xem phân tích theo quận bên dưới).'''))

reg.append(("md", '''### Scatter plot với biến mục tiêu'''))

reg.append(("code", '''samp = df.sample(4000, random_state=42)
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
sns.scatterplot(data=samp, x="area", y="price_total_ty", alpha=0.4, ax=axes[0])
axes[0].set_title("Diện tích và tổng giá"); axes[0].set_xlim(0, 300); axes[0].set_ylim(0, 50)
sns.scatterplot(data=samp, x="bedrooms", y="price_m2", alpha=0.3, ax=axes[1])
axes[1].set_title("Số phòng ngủ và giá/m²")
plt.tight_layout()
plt.show()'''))

reg.append(("md", '''### Nhận xét Scatter Plot

Diện tích và tổng giá có quan hệ **tăng tuyến tính** rõ. Số phòng ngủ tăng thì giá/m² nhỉnh lên nhưng **phân tán lớn**, khẳng định đơn giá phụ thuộc nhiều yếu tố (đặc biệt vị trí) chứ không chỉ số phòng.'''))

reg.append(("md", '''### Phân tích giá theo vị trí và loại hình (yếu tố quan trọng nhất)'''))

reg.append(("code", '''# Giá/m² trung vị theo quận
med = df.groupby("Quận")["price_m2"].median().sort_values()
plt.figure(figsize=(9, 9))
plt.barh(med.index, med.values, color="#457b9d")
plt.title("Giá/m² trung vị theo Quận/Huyện")
plt.xlabel("Giá trung vị (triệu/m²)")
plt.tight_layout()
plt.show()'''))

reg.append(("code", '''# Giá/m² theo loại hình nhà ở
plt.figure(figsize=(9, 5))
sns.boxplot(data=df, x="Loại hình nhà ở", y="price_m2", showfliers=False)
plt.title("Giá/m² theo loại hình nhà ở")
plt.xticks(rotation=15)
plt.ylabel("Giá (triệu/m²)")
plt.tight_layout()
plt.show()'''))

reg.append(("md", '''### Nhận xét theo vị trí và loại hình

- **Vị trí quyết định giá**: các quận trung tâm như **Hoàn Kiếm (~185)**, **Cầu Giấy (~103)**, **Tây Hồ (~102)** có giá/m² cao nhất; các huyện ngoại thành như **Ba Vì (~4)**, **Sơn Tây**, **Thạch Thất** thấp nhất. Chênh lệch tới hàng chục lần.
- **Loại hình**: **nhà mặt phố/mặt tiền** đắt nhất (~112 triệu/m²), cao hơn **nhà ngõ/hẻm** (~86 triệu/m²).

Vậy **Quận** và **Loại hình nhà ở** là hai đặc trưng quan trọng nhất — cần mã hóa (one-hot/target encoding) khi xây mô hình.'''))

reg.append(("code", '''# Đếm outlier bằng IQR cho các biến số
for col in ["area", "bedrooms", "floors", "price_m2", "price_total_ty"]:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    out = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
    print(f"{col}: {len(out)} outliers")'''))

reg.append(("md", '''## Kết luận EDA

- Bài toán **regression — dự đoán giá nhà đất Hà Nội** (`Giá/m²`), phục vụ ứng dụng định giá bất động sản cho thị trường Việt.
- Dữ liệu **thô và bẩn**: giá lẫn nhiều đơn vị (triệu/đ/tỷ), số liệu dạng text, nhiều cột thiếu (`Số tầng`, `Dài`, `Rộng`). Đã tiền xử lý: tách số, quy đổi đơn vị, loại ~1.400 dòng lỗi.
- Biến mục tiêu **lệch phải** → nên log-transform khi mô hình hóa.
- **Vị trí (Quận) và loại hình nhà ở** là yếu tố ảnh hưởng giá mạnh nhất; diện tích quyết định tổng giá; số phòng/số tầng ảnh hưởng vừa phải.

**Hướng tiếp theo:** mã hóa biến phân loại (Quận, loại hình, pháp lý) + xử lý thiếu + log(giá) → huấn luyện mô hình hồi quy (Linear/Random Forest/Gradient Boosting) để định giá tự động cho app bất động sản Việt Nam.'''))

build(reg, os.path.join(BASE, "EDA_Hanoi_Housing_Price.ipynb"))
