# -*- coding: utf-8 -*-
"""Sinh notebook chuyên đề SLIDING WINDOW (tiền đề của NMS).

Notebook dạy cơ chế cửa sổ trượt + kim tự tháp ảnh, chạy một bộ phát hiện thật
(HOG + SVM, Dalal-Triggs) trên ảnh Penn-Fudan, rồi đưa các hộp chồng lấp vào
`greedy_nms` của chuyên đề NMS — nối liền hai chủ đề. Notebook được thực thi sẵn
nên mọi hình kết quả đã nhúng trong tệp .ipynb (tiện xem và ghi chú).

Chạy:  python build_sliding_window_nb.py  ->  Sliding_Window_to_NMS.ipynb
Thực thi & nhúng ảnh:  jupyter nbconvert --to notebook --execute --inplace Sliding_Window_to_NMS.ipynb
"""
import nbformat as nbf

OUT = "Sliding_Window_to_NMS.ipynb"


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
c.append(("md", '''# Sliding Window cho phát hiện đối tượng — tiền đề của Non-Maximum Suppression

**Môn học:** Computer Vision · **Học viên:** Nguyễn Phương Anh Tú · **MSHV:** 2611328

Trước khi đến với Non-Maximum Suppression (NMS), cần hiểu *vì sao* một bộ phát hiện lại sinh ra nhiều hộp chồng lấp. Câu trả lời nằm ở chính cách phát hiện đối tượng cổ điển hoạt động: **cửa sổ trượt** (sliding window). Notebook này trình bày tuần tự:

1. Cơ chế cửa sổ trượt và **kim tự tháp ảnh** (image pyramid).
2. Chi phí của phương pháp: số cửa sổ phải phân loại tăng theo cấp số nhân.
3. Một bộ phát hiện thật theo đúng nguyên lý đó — **HOG + SVM** (Dalal & Triggs, 2005) — chạy trên ảnh người đi bộ Penn-Fudan.
4. Đầu ra là một chùm hộp chồng lấp quanh mỗi người, và đây chính là bài toán mà **NMS** giải quyết. Ta đưa thẳng các hộp này vào `greedy_nms` của chuyên đề kế tiếp.

Như vậy, sliding window là *nguồn sinh ra* các phát hiện chồng lấp, còn NMS là *bước làm sạch*; hai chủ đề nối tiếp nhau một cách tự nhiên.

**Nguồn tham khảo**
- HOG + SVM cho phát hiện người — Dalal & Triggs (2005): https://ieeexplore.ieee.org/document/1467360
- Image pyramid — Adelson và cộng sự (1984): https://ieeexplore.ieee.org/document/1456290
- Penn-Fudan Pedestrian Database: https://www.cis.upenn.edu/~jshi/ped_html/
- `cv2.HOGDescriptor` (OpenCV): https://docs.opencv.org/4.x/d5/d33/structcv_1_1HOGDescriptor.html

> Yêu cầu dữ liệu: đã chạy `python download_data.py` để có thư mục `data/PennFudanPed/`.'''))

# === Setup ================================================================
c.append(("md", '''## 0. Chuẩn bị

Nạp thư viện và một ảnh demo. `sliding_window.py` chứa các hàm thuần NumPy (cửa sổ trượt, kim tự tháp); `nms.py` chứa đúng `greedy_nms` dùng ở chuyên đề NMS — nhập cả hai để thấy hai chủ đề liên kết ngay trong mã.'''))

c.append(("code", '''%matplotlib inline
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from sliding_window import image_pyramid, sliding_window, count_windows
from nms import greedy_nms          # tái dùng greedy NMS của chuyên đề kế tiếp

plt.rcParams["figure.dpi"] = 110
FIG = Path("outputs/figures"); FIG.mkdir(parents=True, exist_ok=True)
DATA = Path("data/PennFudanPed/PNGImages")

def load_rgb(name):
    bgr = cv2.imread(str(DATA / name))
    assert bgr is not None, f"Khong doc duoc anh {name} - da chay download_data.py chua?"
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), bgr

WIN = (64, 128)                     # cua so chuan cua HOG nguoi di bo: rong 64, cao 128
DEMO = "FudanPed00001.png"
rgb, bgr = load_rgb(DEMO)
print("Anh demo:", DEMO, "| kich thuoc (H, W):", rgb.shape[:2])'''))

# === 1. Sliding window ====================================================
c.append(("md", '''## 1. Cơ chế cửa sổ trượt

Bộ phát hiện không "nhìn" cả ảnh một lần. Thay vào đó, một **cửa sổ** kích thước cố định (ở đây $64\\times128$ — tỉ lệ điển hình của người đứng) trượt qua mọi vị trí trên ảnh theo một **bước nhảy** (stride). Tại mỗi vị trí, vùng ảnh bên trong cửa sổ được cắt ra và đưa cho một bộ phân loại để trả lời: *"đây có phải người không, và tin cậy bao nhiêu?"*.

Hình bên trái minh hoạ lưới các vị trí cửa sổ; ô màu đỏ là một cửa sổ ví dụ, và bên phải là đúng vùng ảnh mà bộ phân loại nhận được.'''))

c.append(("code", '''fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5),
                               gridspec_kw={"width_ratios": [3, 1]})
ax1.imshow(rgb)
step = 48
for x, y, _ in sliding_window(rgb, step=step, window_size=WIN):
    ax1.add_patch(plt.Rectangle((x, y), WIN[0], WIN[1], fill=False,
                                edgecolor="deepskyblue", lw=0.4, alpha=0.7))
hx, hy = 168, 185                                    # mot cua so dat tren nguoi ben trai
ax1.add_patch(plt.Rectangle((hx, hy), WIN[0], WIN[1], fill=False, edgecolor="red", lw=2.5))
ax1.set_title(f"Cua so {WIN[0]}x{WIN[1]} truot khap anh (buoc {step}px)"); ax1.axis("off")

ax2.imshow(rgb[hy:hy + WIN[1], hx:hx + WIN[0]])
ax2.set_title("Vung anh trong\\n1 cua so -> bo phan loai"); ax2.axis("off")
fig.tight_layout(); fig.savefig(FIG / "sw_01_sliding_window.png", bbox_inches="tight"); plt.show()'''))

# === 2. Image pyramid =====================================================
c.append(("md", '''## 2. Kim tự tháp ảnh — xử lý nhiều kích thước

Cửa sổ có kích thước **cố định**, nhưng người trong ảnh thì **to nhỏ khác nhau**: ô đỏ ở trên rõ ràng nhỏ hơn người thật. Giải pháp kinh điển là **kim tự tháp ảnh**: thu nhỏ ảnh dần theo một hệ số (ví dụ $1.25$ mỗi tầng) rồi lặp lại cửa sổ trượt trên từng tầng. Một người lớn ở ảnh gốc sẽ "vừa khít" cửa sổ ở một tầng đã được thu nhỏ. Nhờ đó, cùng một cửa sổ cố định phát hiện được vật ở mọi tỉ lệ.

Toạ độ hộp tìm thấy ở tầng thu nhỏ được nhân với `factor` để quy về ảnh gốc.'''))

c.append(("code", '''levels = list(image_pyramid(rgb, scale=1.25, min_size=WIN))
H0, W0 = rgb.shape[:2]
n = len(levels)
fig, axes = plt.subplots(1, n, figsize=(2.0 * n, 3.2))
for ax, (factor, lv) in zip(axes, levels):
    h, w = lv.shape[:2]
    ax.imshow(lv, extent=(0, w, h, 0))     # ve dung kich thuoc that
    ax.set_xlim(0, W0); ax.set_ylim(H0, 0); ax.set_aspect("equal")
    ax.set_title(f"{w}x{h}\\n(x{factor:.2f})", fontsize=9); ax.axis("off")
fig.suptitle(f"Kim tu thap anh: {n} tang (he so 1.25)", y=1.02)
fig.tight_layout(); fig.savefig(FIG / "sw_02_pyramid.png", bbox_inches="tight"); plt.show()
print("So tang:", n, "| factor moi tang:", [round(f, 2) for f, _ in levels])'''))

# === 3. Combinatorial cost ===============================================
c.append(("md", '''## 3. Chi phí: số cửa sổ bùng nổ

Tổng số cửa sổ phải phân loại xấp xỉ *(số vị trí mỗi tầng) × (số tầng)*. Bước nhảy càng nhỏ thì càng quét kỹ nhưng số cửa sổ càng lớn — đây là điểm yếu cố hữu về tốc độ của phương pháp cửa sổ trượt, và là động lực cho các kiến trúc hiện đại (region proposal, anchor) sinh đề xuất hiệu quả hơn.'''))

c.append(("code", '''step = 16
per_level = []
for factor, lv in image_pyramid(rgb, scale=1.25, min_size=WIN):
    h, w = lv.shape[:2]
    nx = max(0, (w - WIN[0]) // step + 1)
    ny = max(0, (h - WIN[1]) // step + 1)
    per_level.append(nx * ny)

fig, ax = plt.subplots(figsize=(7, 3.6))
ax.bar(range(len(per_level)), per_level, color="steelblue")
ax.set_xlabel("Tang kim tu thap"); ax.set_ylabel("So cua so")
ax.set_title(f"So cua so moi tang (buoc {step}px) — tong = {sum(per_level)}")
for i, v in enumerate(per_level):
    ax.text(i, v + 3, str(v), ha="center", fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "sw_03_window_count.png", bbox_inches="tight"); plt.show()

print("Tong so cua so tren toan kim tu thap theo buoc nhay:")
for st in (32, 16, 8):
    print(f"  buoc {st:2d}px -> {count_windows(rgb.shape, st, WIN):>6,d} cua so")'''))

# === 4. HOG + SVM real detector ==========================================
c.append(("md", '''## 4. Biến cửa sổ trượt thành bộ phát hiện thật: HOG + SVM

Ta cần một bộ phân loại chấm điểm cho mỗi cửa sổ. Phương pháp kinh điển là **HOG + SVM tuyến tính** của Dalal & Triggs (2005): mỗi cửa sổ được mô tả bằng **biểu đồ hướng gradient** (Histogram of Oriented Gradients), rồi một SVM tuyến tính cho điểm dương nếu giống người.

OpenCV cung cấp sẵn mô hình này qua `cv2.HOGDescriptor`. Hàm `detectMultiScale` thực hiện đúng quy trình ở Mục 1–3: **trượt cửa sổ trên từng tầng của kim tự tháp ảnh và chấm điểm bằng SVM**. Các tham số tương ứng trực tiếp với những khái niệm vừa học:

| Tham số (OpenCV) | Khái niệm | Ý nghĩa |
|---|---|---|
| `winStride` | bước nhảy cửa sổ | bước trượt theo (x, y); nhỏ thì quét kỹ, chậm hơn |
| `scale` | hệ số kim tự tháp | tỉ lệ thu nhỏ giữa hai tầng (1.05 = nhiều tầng, mịn) |
| `hitThreshold` | ngưỡng tin cậy SVM | chỉ giữ cửa sổ có điểm SVM ≥ ngưỡng |
| `finalThreshold` | gộp nhóm nội bộ | đặt 0 để **tắt** gộp, lộ toàn bộ hộp thô |

Đặt `hitThreshold = 0` và tắt gộp nhóm để thấy *toàn bộ* cửa sổ mà SVM cho là người.'''))

c.append(("code", '''hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def hog_raw(bgr_img, hit_threshold=0.0):
    """Sliding window + kim tu thap + SVM. Tra (boxes_xyxy [N,4], scores [N])."""
    # tham so theo thu tu: img, hitThreshold, winStride, padding, scale, finalThreshold, useMeanshift
    rects, weights = hog.detectMultiScale(bgr_img, hit_threshold, (4, 4), (8, 8), 1.05, 0, False)
    if len(rects) == 0:
        return torch.zeros((0, 4)), torch.zeros((0,))
    boxes = torch.tensor([[x, y, x + w, y + h] for (x, y, w, h) in rects], dtype=torch.float)
    scores = torch.tensor(np.asarray(weights).ravel(), dtype=torch.float)
    return boxes, scores

def draw_boxes(ax, image, boxes, color, title, lw=1.4, alpha=1.0):
    ax.imshow(image)
    for b in boxes:
        x1, y1, x2, y2 = [float(v) for v in b]
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                   edgecolor=color, lw=lw, alpha=alpha))
    ax.set_title(title, fontsize=10); ax.axis("off")

boxes_raw, scores_raw = hog_raw(bgr, hit_threshold=0.0)
fig, ax = plt.subplots(figsize=(6.5, 6))
draw_boxes(ax, rgb, boxes_raw, "red",
           f"Dau ra tho cua sliding window: {len(boxes_raw)} hop", lw=0.8, alpha=0.6)
fig.tight_layout(); fig.savefig(FIG / "sw_04_hog_raw.png", bbox_inches="tight"); plt.show()
print(f"So hop tho: {len(boxes_raw)} | diem SVM trong khoang "
      f"[{scores_raw.min():.2f}, {scores_raw.max():.2f}]")'''))

# === 5. The overlap problem -> NMS =======================================
c.append(("md", '''## 5. Vấn đề: chùm hộp chồng lấp → cần NMS

Vì các cửa sổ kề nhau và nhiều tầng tỉ lệ cùng bắt trúng một người, mỗi người bị bao bởi hàng chục hộp gần trùng. Đây chính xác là tình huống mà **Non-Maximum Suppression** xử lý: giữ hộp điểm cao nhất, loại các hộp chồng lấp nhiều với nó, lặp lại. Ta đưa thẳng đầu ra HOG vào `greedy_nms` (cùng hàm dùng ở chuyên đề NMS) với ngưỡng IoU $0.5$.'''))

c.append(("code", '''keep = greedy_nms(boxes_raw, scores_raw, iou_thr=0.5)
boxes_nms = boxes_raw[keep]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 6))
draw_boxes(a1, rgb, boxes_raw, "red",  f"Truoc NMS: {len(boxes_raw)} hop", lw=0.8, alpha=0.6)
draw_boxes(a2, rgb, boxes_nms, "lime", f"Sau greedy_nms@0.5: {len(boxes_nms)} hop", lw=2.2)
fig.tight_layout(); fig.savefig(FIG / "sw_05_before_after_nms.png", bbox_inches="tight"); plt.show()
print(f"Sliding window -> NMS: {len(boxes_raw)} hop chong lap rut ve {len(boxes_nms)} hop.")'''))

# === 6. Two thresholds ====================================================
c.append(("md", '''## 6. Hai tham số điều tiết kết quả

Pipeline có hai tham số quyết định kết quả cuối, đúng bằng hai tham số sẽ phân tích kỹ ở chuyên đề NMS:

- **`hitThreshold`** (ngưỡng tin cậy SVM) — lọc *trước* NMS: tăng ngưỡng thì ít cửa sổ thô hơn, sạch hơn nhưng dễ bỏ sót người mờ/xa. Tương ứng `score_threshold` của detector.
- **`iou_thr`** (ngưỡng IoU của NMS) — gộp *sau* khi đã có cửa sổ: thấp thì gom mạnh, cao thì giữ lại nhiều hộp trùng.

Bảng dưới quét đồng thời hai tham số; hình minh hoạ ảnh hưởng riêng của ngưỡng IoU (cố định `hitThreshold = 0.3`).'''))

c.append(("code", '''print(f"{'hitThreshold':>13} | {'hop tho':>8} | {'nms@0.3':>8} | {'nms@0.5':>8} | {'nms@0.7':>8}")
print("-" * 58)
for ht in (0.0, 0.3, 0.5, 0.7):
    b, s = hog_raw(bgr, hit_threshold=ht)
    cols = [len(greedy_nms(b, s, t)) for t in (0.3, 0.5, 0.7)]
    print(f"{ht:13.1f} | {len(b):8d} | {cols[0]:8d} | {cols[1]:8d} | {cols[2]:8d}")

b03, s03 = hog_raw(bgr, hit_threshold=0.3)
fig, axes = plt.subplots(1, 3, figsize=(13, 5))
for ax, t in zip(axes, (0.3, 0.5, 0.7)):
    k = greedy_nms(b03, s03, t)
    draw_boxes(ax, rgb, b03[k], "lime", f"iou_thr = {t}  ->  {len(k)} hop", lw=2.0)
fig.suptitle("Anh huong nguong IoU cua NMS (hitThreshold = 0.3 co dinh)", y=1.0)
fig.tight_layout(); fig.savefig(FIG / "sw_06_param_grid.png", bbox_inches="tight"); plt.show()'''))

# === 7. Gallery ===========================================================
c.append(("md", '''## 7. Kiểm tra trên nhiều ảnh

Chạy lại toàn bộ pipeline (sliding window → HOG/SVM → NMS) trên vài ảnh khác để xác nhận hành vi nhất quán, không phải may rủi trên một ảnh. HOG là bộ phát hiện cổ điển nên còn lọt vài dương tính giả — điều này hợp lý và cho thấy giới hạn của phương pháp so với detector học sâu ở chuyên đề sau.'''))

c.append(("code", '''samples = ["FudanPed00001.png", "FudanPed00007.png", "FudanPed00025.png"]
fig, axes = plt.subplots(len(samples), 2, figsize=(11, 5.2 * len(samples)))
for row, name in zip(axes, samples):
    rgb_i, bgr_i = load_rgb(name)
    b, s = hog_raw(bgr_i, hit_threshold=0.3)
    k = greedy_nms(b, s, 0.5)
    draw_boxes(row[0], rgb_i, b,    "red",  f"{name}: {len(b)} hop tho", lw=0.8, alpha=0.6)
    draw_boxes(row[1], rgb_i, b[k], "lime", f"sau NMS@0.5: {len(k)} hop", lw=2.2)
fig.tight_layout(); fig.savefig(FIG / "sw_07_gallery.png", bbox_inches="tight"); plt.show()'''))

# === 8. Bridge to NMS =====================================================
c.append(("md", '''## 8. Kết nối sang chuyên đề NMS

Tóm tắt mạch logic:

- **Cửa sổ trượt + kim tự tháp ảnh** quét mọi vị trí và mọi tỉ lệ; bộ phân loại (HOG + SVM) chấm điểm từng cửa sổ.
- Vì nhiều cửa sổ kề nhau cùng trúng một vật, đầu ra là **một chùm hộp chồng lấp** quanh mỗi người — minh hoạ trực tiếp bằng số (ví dụ ảnh demo: hàng trăm hộp thô).
- **NMS** rút chùm hộp đó về một đại diện cho mỗi vật; ở đây tái dùng đúng `greedy_nms` của chuyên đề kế tiếp.
- Detector học sâu (Faster R-CNN, YOLO) thay cửa sổ trượt bằng cơ chế đề xuất vùng/anchor hiệu quả hơn, **nhưng vẫn sinh hộp chồng lấp và vẫn cần NMS** — nên hai chủ đề gắn liền nhau.

Chuyên đề tiếp theo phân tích sâu NMS trên một detector học sâu đã huấn luyện: [`report_NMS_thuc_hanh.md`](report_NMS_thuc_hanh.md); cài đặt thuật toán ở [`nms.py`](nms.py); lý thuyết đầy đủ ở [`../non_maximum_suppression/nghien_cuu_NMS.md`](../non_maximum_suppression/nghien_cuu_NMS.md).'''))

build(c, OUT)
