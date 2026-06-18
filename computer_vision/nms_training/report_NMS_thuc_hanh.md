# Báo cáo thực hành: Non-Maximum Suppression (NMS) — gom các đối tượng nhận dạng chồng lấp

**Môn học:** Computer Vision · **Học viên:** Nguyễn Phương Anh Tú · **MSHV:** 2611328
**Loại báo cáo:** thực nghiệm có huấn luyện mô hình thật (Faster R-CNN) trên GPU RTX 5080.

> **Cam kết tính trung thực.** Toàn bộ con số và hình ảnh trong báo cáo được sinh từ **một lần chạy thật** trên máy, lấy trực tiếp từ `outputs/logs/` và `outputs/figures/`. Không có số liệu giả định hay placeholder. Cách tái lập đầy đủ ở Mục 8.

> **Hướng dẫn đọc / thuyết trình.** Mỗi mục lớn có khối **"Điểm nhấn thuyết trình"** tóm tắt 1–2 câu cốt lõi để nói trước đám đông. Phần lý thuyết (Mục 2) và phần phân tích (Mục 5) là trọng tâm học thuật; phần quy trình (Mục 4) là minh chứng "có làm thật".

---

## Mục lục
1. Giới thiệu và động lực
2. Cơ sở lý thuyết: IoU và thuật toán NMS
3. Phương pháp và thiết lập thực nghiệm
4. Toàn bộ quá trình thực nghiệm (minh chứng)
5. Kết quả và phân tích chuyên sâu
6. Các lỗi thường gặp và bài học
7. Kết luận
8. Phụ lục: tái lập, huấn luyện thêm (resume)
9. Tài liệu tham khảo

---

## 1. Giới thiệu và động lực

### 1.1. Bài toán
Phát hiện đối tượng (object detection) trả lời câu hỏi *"có những vật gì, ở đâu"* bằng cách vẽ **hộp giới hạn (bounding box)** quanh mỗi vật kèm điểm tin cậy. Báo cáo này dùng bài toán cụ thể: **phát hiện người đi bộ** trong ảnh đường phố.

### 1.2. Vì sao xuất hiện "đối tượng nhận dạng chồng lấp"?
Mô hình phát hiện **không** sinh ra đúng một hộp cho mỗi người. Nó quét hàng nghìn vị trí/khung neo (anchor) và, do được huấn luyện theo kiểu **gán nhãn một–nhiều** (mỗi vật thật khớp với nhiều dự đoán để học tốt hơn), nên ở đầu ra **mỗi người bị bao bởi cả một chùm hộp gần như trùng nhau**. Thực nghiệm của báo cáo đo trực tiếp: trên một ảnh có 7 người, mô hình xuất **140 hộp chồng lấp** (Mục 5.2).

Vậy cần một bước **hậu xử lý** để *gom mỗi chùm hộp trùng về một đại diện duy nhất*. Đó chính là **Non-Maximum Suppression (NMS)** — "khử các cực đại không phải lớn nhất".

### 1.3. Mục tiêu báo cáo
1. Huấn luyện **thật** một detector trên dữ liệu thật, đạt chất lượng đủ tốt để minh hoạ.
2. Cài đặt **standard (greedy) NMS từ đầu**, chứng minh đúng, và **quan sát hành vi** của nó.
3. Phân tích **ảnh hưởng của ngưỡng IoU** và rút ra **các lỗi cần tránh** khi áp dụng NMS.

> **Điểm nhấn thuyết trình:** *Detector cho ra nhiều hộp chồng nhau quanh một vật; NMS là bước "dọn dẹp" giữ lại một hộp tốt nhất cho mỗi vật. Đây là bước gần như mọi detector (R-CNN, YOLO, SSD) đều cần.*

---

## 2. Cơ sở lý thuyết: IoU và thuật toán NMS

### 2.1. IoU — thước đo độ chồng lấp
Để biết "hai hộp có cùng mô tả một vật không", ta đo **Intersection over Union**:

$$\text{IoU}(A,B)=\frac{\text{diện tích}(A\cap B)}{\text{diện tích}(A\cup B)}\in[0,1].$$

IoU = 0 nghĩa là hai hộp rời nhau; IoU = 1 nghĩa là trùng khít. NMS dùng một **ngưỡng** $N_t$: hai hộp có $\text{IoU}\ge N_t$ bị coi là "cùng một vật".

### 2.2. Greedy NMS — thuật toán
Chạy **riêng từng lớp**. Trong một lớp:
1. Sắp xếp các hộp theo **điểm tin cậy giảm dần**.
2. Lấy hộp điểm cao nhất $M$ → **giữ lại** (đại diện cụm).
3. **Loại bỏ** mọi hộp còn lại có $\text{IoU}(M, b)\ge N_t$.
4. Lặp lại với các hộp chưa bị loại cho tới khi hết.

Mã giả:
```
NMS(B, S, N_t):
    D ← rỗng
    while B khác rỗng:
        m ← hộp có điểm cao nhất;  D ← D ∪ {m};  B ← B \ {m}
        với mỗi b trong B:  nếu IoU(m, b) ≥ N_t  thì  B ← B \ {b}   # XOÁ HẲN
    return D
```
Cài đặt thật (`nms.py`), kèm 3 "quy tắc vàng" được mã hoá ngay trong code:
```python
def greedy_nms(boxes, scores, iou_thr):
    assert boxes.ndim == 2 and boxes.shape[1] == 4, "boxes phải [N,4] xyxy"   # (1) đúng format
    if boxes.numel() == 0:                                                     # (2) guard rỗng
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    order = torch.argsort(scores, descending=True)                            # (3) xét theo điểm GIẢM DẦN
    keep = []
    while order.numel() > 0:
        i = order[0]; keep.append(int(i))
        if order.numel() == 1: break
        rest = order[1:]
        ious = box_iou_pairwise(boxes[i], boxes[rest])
        order = rest[ious < iou_thr]      # chỉ giữ hộp chồng lấp ÍT (chưa bị "nuốt")
    return torch.tensor(keep, ...)
```

### 2.3. Độ phức tạp và tham số
- Độ phức tạp xấu nhất $O(n^2)$ (n = số hộp một lớp sau khi lọc điểm). Bản chất **tuần tự** → khó song song hoá (động lực cho Cluster/Matrix-NMS — xem lý thuyết).
- Ba tham số: `score_threshold` (lọc hộp điểm thấp **trước**), `iou_threshold` $N_t$ (ngưỡng coi là trùng), `top_k` (giới hạn số hộp).

> **Điểm nhấn thuyết trình:** *NMS rất đơn giản — "giữ hộp điểm cao nhất, xoá mọi hộp đè lên nó quá nhiều, lặp lại". Tham số quan trọng nhất là ngưỡng IoU.*

---

## 3. Phương pháp và thiết lập thực nghiệm

| Hạng mục | Lựa chọn | Lý do |
|---|---|---|
| Dữ liệu | **Penn-Fudan Pedestrian** (170 ảnh, 1 lớp `person`) | nhỏ, người **chen/đè nhau** → lý tưởng để thấy NMS gom chồng lấp; train nhanh |
| Mô hình | `fasterrcnn_resnet50_fpn` (COCO-pretrained), thay đầu phân loại → 2 lớp | finetune hội tụ nhanh, có NMS nội bộ để "mở ra" nghiên cứu |
| Chia dữ liệu | seed=1 → **train 120 / val 50** | cố định để tái lập |
| Tối ưu | SGD lr=0.005, momentum=0.9, wd=5e-4; **StepLR(step = epochs//3, γ=0.1)**; AMP | cấu hình chuẩn; lịch LR **scale theo số epoch** để không "đóng băng" sớm |
| Phần cứng | RTX 5080 (sm_120), torch 2.11.0+**cu128** | wheel CUDA 12.8 bắt buộc cho GPU Blackwell |
| Độ đo | **AP@0.5** và **mAP@[.5:.95]** tự cài (`metrics.py`) | minh bạch, không cần `pycocotools`; khớp định nghĩa PASCAL/COCO |

**Vì sao tự viết độ đo thay vì pycocotools?** Để (a) tránh phụ thuộc build C, (b) mọi con số đều **truy vết được** tới mã nguồn mình viết — phù hợp mục tiêu *học hiểu sâu*. Hàm `average_precision` dựng đường Precision–Recall rồi lấy diện tích dưới đường (all-point AP); `map_50_95` trung bình AP qua các ngưỡng IoU 0.50→0.95.

### 3.1. Finetune từ đâu? (điểm khởi đầu của trọng số)
Mô hình **không** train từ số 0. Ta **finetune** từ trọng số đã huấn luyện sẵn:

- **Kiến trúc:** Faster R-CNN, backbone **ResNet-50 + FPN**.
- **Trọng số khởi đầu:** `torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")`, tức `FasterRCNN_ResNet50_FPN_Weights.DEFAULT` (= phiên bản **COCO_V1**) — đã huấn luyện trên **COCO train2017** (80 lớp vật thể). torchvision tự tải tệp `fasterrcnn_resnet50_fpn_coco-258fb6c6.pth` từ `https://download.pytorch.org/models/` (xác nhận trong log train).
- **Phần nào được giữ, phần nào train lại:** backbone ResNet-50, FPN, mạng đề xuất vùng (RPN) và lớp đặc trưng RoI **giữ nguyên trọng số COCO**; chỉ **đầu dự đoán cuối** (`roi_heads.box_predictor`) bị **thay mới** bằng `FastRCNNPredictor(in_features, num_classes=2)` — khởi tạo ngẫu nhiên cho **2 lớp** (nền + `person`) — rồi huấn luyện cùng toàn mạng trên Penn-Fudan.
- **Vì sao finetune từ COCO:** COCO đã có sẵn lớp `person`, nên đặc trưng học được rất phù hợp; nhờ đó AP@0.5 đạt ~0.99 **ngay từ epoch 0** và hội tụ trong ~2 phút thay vì phải train từ đầu (cần rất nhiều dữ liệu/thời gian). Mã: `model.py → build_model()`.

---

## 4. Toàn bộ quá trình thực nghiệm (minh chứng "có làm")

Thực hiện tuần tự; mỗi bước có output thật.

**Bước 1 — Kiểm tra GPU (gate).** `uv run python check_env.py`:
```
torch 2.11.0+cu128 | torchvision 0.26.0+cu128
cuda available: True
device: NVIDIA GeForce RTX 5080 | cap: (12, 0)
cuda matmul OK: True
```

**Bước 2 — Tải dữ liệu.** `uv run python download_data.py` → `images: 170 masks: 170` (zip 53.7 MB, sha256 `9095a96…`).

**Bước 3 — Huấn luyện 20 epoch.** `uv run python train.py --epochs 20` (≈2 phút, ~5.5 s/epoch). Toàn bộ lịch sử (thật):

| epoch | loss | AP@0.5 | mAP | | epoch | loss | AP@0.5 | mAP |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.403 | 0.989 | 0.700 | | 10 | 0.050 | 0.993 | 0.819 |
| 1 | 0.141 | 0.989 | 0.748 | | 11 | 0.047 | 0.992 | 0.810 |
| 2 | 0.105 | 0.990 | 0.793 | | 12 | 0.046 | 0.992 | 0.827 |
| 3 | 0.090 | 0.989 | 0.786 | | 13 | 0.046 | 0.992 | 0.824 |
| 4 | 0.084 | 0.988 | 0.795 | | 14 | 0.045 | 0.992 | 0.827 |
| 5 | 0.075 | 0.993 | 0.743 | | 15 | 0.044 | 0.992 | 0.828 |
| 6 | 0.062 | 0.993 | 0.817 | | 16 | 0.046 | 0.992 | 0.827 |
| 7 | 0.052 | 0.993 | 0.818 | | 17 | 0.046 | 0.992 | 0.829 |
| **8** | 0.053 | 0.993 | **0.831** | | 18 | 0.044 | 0.992 | 0.828 |
| 9 | 0.051 | 0.993 | 0.819 | | 19 | 0.045 | 0.992 | 0.827 |

![Loss và AP theo epoch](outputs/figures/01_loss_curve.png)

**Hai điều chỉnh phương pháp đã thực hiện (và vì sao):**
1. **Lịch LR scale theo số epoch** (`step_size = epochs//3`): khi kéo dài lên 20 epoch, LR giảm dần đều (0.005 → 5e-4 quanh epoch 7 → 5e-5 quanh epoch 14) thay vì giảm hết quá sớm.
2. **Chọn mô hình tốt nhất theo mAP@[.5:.95], không theo AP@0.5:** vì AP@0.5 **bão hoà** quanh 0.99 (xem cột AP@0.5 gần như phẳng) nên không phân biệt được epoch nào tốt hơn; mAP@[.5:.95] mới phản ánh độ chính xác **định vị**. Kết quả: `best.pth` = **epoch 8** (mAP 0.831).

**Bước 4 — Sinh hình & bảng.** `uv run python study_nms.py` → in `NMS self-check OK`, tạo 5 hình trong `outputs/figures/` và `outputs/logs/nms_sweep.csv`.

> **Điểm nhấn thuyết trình:** *Train rất nhanh nhờ finetune (2 phút). Một bài học phương pháp: AP@0.5 bão hoà nên phải chọn mô hình theo mAP@[.5:.95].*

---

## 5. Kết quả và phân tích chuyên sâu

### 5.1. Chất lượng detector
- **AP@0.5 ≈ 0.99** ngay từ epoch đầu: nhờ finetune từ COCO, việc "có người ở đâu" gần như đúng hết ở ngưỡng IoU dễ (0.5).
- **mAP@[.5:.95]** tăng 0.70 → **0.83** rồi **bão hoà** từ ~epoch 8: đây là phần khó — khớp hộp **chính xác** ở các ngưỡng IoU chặt (tới 0.95). Đường cong phẳng dần ⇒ mô hình đã **hội tụ**; train thêm nữa lợi ích không đáng kể (minh chứng cho việc 20 epoch là **đủ**, không cần ép thêm).
- **Phân tích sâu:** AP@0.5 **bão hoà** là tín hiệu rằng nó **không phải** chỉ số tốt để so sánh mô hình trong bài này; mAP@[.5:.95] phân biệt rõ hơn. Đây cũng là lý do COCO dùng mAP@[.5:.95] làm chỉ số chính.

### 5.2. NMS gom đối tượng chồng lấp — trước/sau
![Trước và sau NMS](outputs/figures/02_before_after_nms.png)

Trên ảnh val đông nhất (7 người), tắt NMS cuối của mô hình (`nms_thresh=1.0`, `score_thresh=0.05`) để lộ đầu ra "thô": **140 hộp** chồng chất (trái). Áp **greedy-NMS@0.5**: còn **8 hộp** sạch (phải) — mỗi người ≈ một hộp. *(Trung thực: NMS ở tầng đề xuất vùng RPN vẫn bật; ta nghiên cứu NMS ở đầu phát hiện cuối.)*

### 5.3. Ảnh hưởng ngưỡng IoU — phân tích trade-off
![Ảnh hưởng ngưỡng IoU](outputs/figures/03_iou_threshold_sweep.png)

Trên cùng ảnh, số hộp giữ lại theo ngưỡng: **8 (0.3) → 10 (0.5) → 11 (0.7) → 42 (0.9)**. Bảng định lượng trên **toàn tập val** (`nms_sweep.csv`):

| Ngưỡng IoU | AP@0.5 | Recall | Số hộp trung bình/ảnh sau NMS |
|---|---|---|---|
| 0.3 | 0.9936 | 1.000 | 2.86 |
| 0.4 | 0.9929 | 1.000 | 2.92 |
| 0.5 | 0.9925 | 1.000 | 3.28 |
| 0.6 | 0.9916 | 1.000 | 3.92 |
| 0.7 | 0.9898 | 1.000 | 5.18 |
| 0.9 | 0.9469 | 1.000 | **21.68** |

**Phân tích:**
- **Số hộp giữ lại tăng đơn điệu** theo ngưỡng (2.86 → 21.68): ngưỡng càng cao, NMS càng "khoan dung", càng nhiều hộp trùng sống sót.
- **AP@0.5 giảm** khi ngưỡng tăng (0.9936 → 0.9469): các hộp trùng dư thừa trở thành **dương tính giả (false positive)**, kéo precision xuống.
- ⇒ **Vùng ngưỡng hợp lý ≈ 0.3–0.5** cho dữ liệu này (ít hộp dư, AP cao nhất).

### 5.4. Vì sao Recall = 1.0 ở mọi ngưỡng? (nhận xét trung thực)
Trên Penn-Fudan val, người tách nhau tương đối rõ và detector mạnh, nên **ngay cả ngưỡng thấp 0.3 cũng không xoá nhầm người thật** → recall luôn = 1.0. Hệ quả: ở bộ dữ liệu này, mặt trái "ngưỡng thấp làm **rớt recall**" chưa lộ ra trong con số recall; nó thể hiện gián tiếp qua **số hộp** và **AP/precision**. Mặt trái đó sẽ rõ rệt trên dữ liệu **cảnh đông gắt** (CrowdHuman/CityPersons) — nơi hai người che nhau có IoU hộp cao và bị NMS ngưỡng thấp gộp nhầm. *(Đây là một nhận xét quan trọng: kết luận về NMS phụ thuộc độ "đông" của dữ liệu.)*

> **Điểm nhấn thuyết trình:** *Ngưỡng IoU là "núm vặn" của NMS. Thấp → gom mạnh, sạch nhưng rủi ro gộp nhầm người sát nhau; cao → giữ lại nhiều hộp trùng (rác). Ở dữ liệu này, 0.3–0.5 là tốt nhất.*

---

## 6. Các lỗi thường gặp và bài học ("cái sai cần tránh")

### 6.1. NMS không phân lớp (class-agnostic) xoá nhầm vật khác lớp — đã đo
![Lỗi class-agnostic](outputs/figures/04_pitfall_classagnostic.png)
- **Hiện tượng:** hai hộp đè nhau nhưng khác lớp (person vs dog); NMS gộp chung mọi lớp sẽ xoá mất một (giữ 1/2).
- **Vì sao:** NMS chỉ nhìn IoU + điểm; IoU cao ⇒ bị coi là trùng dù khác lớp.
- **Cách đúng:** chạy NMS **riêng từng lớp** — `torchvision.ops.batched_nms(boxes, scores, class_ids, thr)` (giữ 2/2).

### 6.2. Sai định dạng hộp (xywh vs xyxy) — đã đo
![Lỗi định dạng box](outputs/figures/05_pitfall_format_xywh.png)
- **Hiện tượng:** đưa dữ liệu `xywh` (x, y, rộng, cao) vào hàm vốn mong `xyxy` (hai góc) → hộp phình/tràn khung, IoU vô nghĩa, gom sai.
- **Cách đúng:** chuyển về `xyxy` trước (`torchvision.ops.box_convert`); `greedy_nms` có `assert` chặn nhầm dạng `[N,4]`.

### 6.3. Ngưỡng IoU quá thấp ở cảnh đông → mất vật thật
- Hai người che nhau có IoU hộp cao; ngưỡng thấp gộp nhầm → **giảm recall**. Cách đúng: tăng ngưỡng vùng đông, hoặc dùng **Soft-NMS / Adaptive-NMS / DIoU-NMS** (lý thuyết ở `../non_maximum_suppression/nghien_cuu_NMS.md`).

### 6.4. Quên lọc điểm hoặc quên sắp theo điểm
- Chạy NMS trên cả nghìn hộp điểm thấp → chậm và giữ rác. Không sort giảm dần → "đại diện" cụm sai. Cách đúng: lọc `score_threshold` trước; luôn xét theo điểm giảm dần.

### 6.5. NMS trên tensor rỗng
- Ảnh không có vật → `boxes` rỗng làm vỡ shape/argmax. Cách đúng: guard rỗng (đã có trong `greedy_nms`).

### 6.6. Xếp hạng bằng điểm phân loại thay vì độ chính xác định vị
- Giữ hộp "tự tin về lớp" nhưng định vị lệch, bỏ hộp định vị tốt hơn. Hướng khắc phục: localization confidence (IoU-Net).

### 6.7. Chọn mô hình bằng chỉ số bão hoà (bài học từ chính dự án này)
- Chọn `best` theo **AP@0.5** (bão hoà ~0.99) sẽ chọn nhầm epoch noisy. **Cách đúng:** chọn theo **mAP@[.5:.95]** — chỉ số phân biệt được chất lượng định vị (đã sửa trong `train.py`).

### 6.8. Dùng API AMP đã lỗi thời
- `torch.cuda.amp.autocast/GradScaler` phát cảnh báo deprecated. Cách đúng: `torch.amp.autocast('cuda')` + `torch.amp.GradScaler('cuda')` (đã dùng).

> **Điểm nhấn thuyết trình:** *Hai lỗi "chết người" hay gặp nhất: (1) quên chạy NMS theo từng lớp, (2) nhầm định dạng hộp xywh/xyxy. Cả hai đều làm kết quả sai mà không báo lỗi.*

---

## 7. Kết luận
- Đã **huấn luyện thật** một Faster R-CNN trên Penn-Fudan, 20 epoch (~2 phút trên RTX 5080), đạt **AP@0.5 ≈ 0.99** và **mAP@[.5:.95] ≈ 0.83** (best ở epoch 8); đường cong hội tụ rõ → 20 epoch là đủ.
- **NMS cổ điển gom hiệu quả** các đối tượng chồng lấp: **140 → 8 hộp** trên một ảnh đông.
- **Ngưỡng IoU** chi phối trực tiếp số hộp giữ lại (2.86 → 21.68) và precision (AP@0.5 0.9936 → 0.9469); vùng tốt ≈ **0.3–0.5**.
- Rút ra **8 lỗi cần tránh**, trong đó 2 lỗi được minh hoạ bằng thực nghiệm và 1 lỗi rút ra từ chính quá trình (chọn mô hình theo chỉ số bão hoà).

## 8. Phụ lục — tái lập và huấn luyện thêm
```bash
cd computer_vision/nms_training
uv run python check_env.py                 # kiểm tra GPU
uv run python download_data.py             # tải dữ liệu
uv run python train.py --epochs 20         # huấn luyện thật
uv run python study_nms.py                 # sinh hình + bảng
# Huấn luyện THÊM mà không làm lại từ đầu (nạp lại model/optimizer/scheduler/history):
uv run python train.py --resume outputs/checkpoints/last.pth --epochs 24
```
Nhật ký đầy đủ + bảng từng epoch: xem `his.md`.

## 9. Tài liệu tham khảo
- Lý thuyết NMS đầy đủ (biến thể Soft/Adaptive/DIoU/Cluster/Matrix-NMS, hướng NMS-free DETR/YOLOv10) và danh mục trích dẫn arXiv: [`../non_maximum_suppression/nghien_cuu_NMS.md`](../non_maximum_suppression/nghien_cuu_NMS.md) và [`../non_maximum_suppression/references.md`](../non_maximum_suppression/references.md).
- Penn-Fudan Pedestrian Database; torchvision object detection finetuning tutorial.
