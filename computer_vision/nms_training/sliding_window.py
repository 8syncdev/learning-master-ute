"""Sliding window + kim tự tháp ảnh — phát hiện đối tượng cổ điển, tiền đề của NMS.

Ý tưởng: trượt một cửa sổ kích thước cố định qua ảnh ở nhiều VỊ TRÍ (bước nhảy
`step`) và nhiều TỈ LỆ (kim tự tháp ảnh); mỗi cửa sổ được một bộ phân loại chấm
điểm. Vì các cửa sổ kề nhau và các tầng tỉ lệ cùng bắt trúng một vật, đầu ra có
rất nhiều hộp chồng lấp — đó chính là lý do phải dùng Non-Maximum Suppression
(xem `nms.py`). Các hàm dưới đây thuần NumPy nên chạy được mà không cần OpenCV;
phần phân loại cửa sổ (HOG + SVM) minh hoạ trong notebook đi kèm.
"""
import numpy as np


def _resize_nn(img, out_hw):
    """Thu/phóng ảnh bằng nội suy lân cận gần nhất (thuần NumPy)."""
    h, w = img.shape[:2]
    oh, ow = out_hw
    ys = np.clip((np.arange(oh) * (h / oh)).astype(int), 0, h - 1)
    xs = np.clip((np.arange(ow) * (w / ow)).astype(int), 0, w - 1)
    return img[ys][:, xs]


def image_pyramid(image, scale=1.25, min_size=(64, 128)):
    """Sinh kim tự tháp ảnh: bắt đầu từ ảnh gốc rồi thu nhỏ dần theo hệ số `scale`.

    Yield (factor, level): `factor` = kích_thước_gốc / kích_thước_hiện_tại (>= 1),
    dùng để quy đổi toạ độ hộp ở tầng hiện tại về ảnh gốc (nhân với `factor`).
    Dừng khi tầng tiếp theo nhỏ hơn `min_size` = (rộng, cao) tối thiểu của cửa sổ.
    """
    yield 1.0, image
    h0 = image.shape[0]
    cur = image
    while True:
        h = int(cur.shape[0] / scale)
        w = int(cur.shape[1] / scale)
        if w < min_size[0] or h < min_size[1]:
            break
        cur = _resize_nn(cur, (h, w))
        yield h0 / h, cur


def sliding_window(image, step, window_size):
    """Trượt cửa sổ (rộng, cao) = `window_size` qua ảnh theo bước `step`.

    Yield (x, y, patch) với (x, y) là góc trên-trái; chỉ sinh các cửa sổ nằm trọn
    trong ảnh (không tràn biên).
    """
    win_w, win_h = window_size
    h, w = image.shape[:2]
    for y in range(0, h - win_h + 1, step):
        for x in range(0, w - win_w + 1, step):
            yield x, y, image[y:y + win_h, x:x + win_w]


def count_windows(image_shape, step, window_size, scale=1.25, min_size=(64, 128)):
    """Đếm tổng số cửa sổ phải phân loại qua toàn bộ kim tự tháp.

    Cho thấy 'sự bùng nổ tổ hợp': chi phí xấp xỉ (số vị trí) × (số tầng tỉ lệ).
    """
    h, w = image_shape[:2]
    win_w, win_h = window_size
    total = 0
    while w >= min_size[0] and h >= min_size[1]:
        nx = max(0, (w - win_w) // step + 1)
        ny = max(0, (h - win_h) // step + 1)
        total += nx * ny
        h = int(h / scale)
        w = int(w / scale)
    return total
