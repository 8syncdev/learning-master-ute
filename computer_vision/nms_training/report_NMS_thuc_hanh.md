# Báo cáo thực hành: Non-Maximum Suppression (NMS) — gom các đối tượng nhận dạng chồng lấp

**Môn học:** Computer Vision · **Học viên:** Nguyễn Phương Anh Tú · **MSHV:** 2611328
**Loại:** thực nghiệm có huấn luyện mô hình thật (Faster R-CNN) trên GPU RTX 5080.

> **Cam kết tính trung thực.** Mọi con số và hình ảnh đều sinh từ **một lần chạy thật**, lấy trực tiếp từ `outputs/logs/` và `outputs/figures/`. Không có số giả định/placeholder. Cách tái lập ở Mục 8.

---

## Nguồn & trích dẫn (links) — tổng hợp ở đầu file

**Mô hình & thành phần đã dùng**
- Faster R-CNN — Ren, He, Girshick, Sun (2015): https://arxiv.org/abs/1506.01497
- Feature Pyramid Network (FPN) — Lin và cộng sự (2016): https://arxiv.org/abs/1612.03144
- ResNet (backbone ResNet-50) — He và cộng sự (2015): https://arxiv.org/abs/1512.03385
- R-CNN (NMS là thành phần chuẩn của detector) — Girshick và cộng sự (2014): https://arxiv.org/abs/1311.2524

**Dữ liệu**
- Penn-Fudan Pedestrian Database — trang chủ & tệp dữ liệu: https://www.cis.upenn.edu/~jshi/ped_html/
- MS COCO (tập pretrain của trọng số khởi đầu) — Lin và cộng sự (2014): https://arxiv.org/abs/1405.0312

**NMS và các biến thể (được nhắc trong báo cáo)**
- NMS hiệu quả (kinh điển) — Neubeck & Van Gool, ICPR 2006: https://ieeexplore.ieee.org/document/1699659
- Soft-NMS — Bodla và cộng sự (2017): https://arxiv.org/abs/1704.04503
- Learning NMS / GossipNet — Hosang và cộng sự (2017): https://arxiv.org/abs/1705.02950
- Adaptive-NMS (cảnh đông) — Liu và cộng sự (2019): https://arxiv.org/abs/1904.03629
- IoU-Net (xếp hạng theo localization) — Jiang và cộng sự (2018): https://arxiv.org/abs/1807.11590
- DIoU/CIoU + DIoU-NMS — Zheng và cộng sự (2020): https://arxiv.org/abs/1911.08287
- Cluster-NMS (+CIoU) — Zheng và cộng sự (2021): https://arxiv.org/abs/2005.03572
- Matrix-NMS (SOLOv2) — Wang và cộng sự (2020): https://arxiv.org/abs/2003.10152
- Fast-NMS (YOLACT) — Bolya và cộng sự (2019): https://arxiv.org/abs/1904.02689
- GIoU — Rezatofighi và cộng sự (2019): https://arxiv.org/abs/1902.09630
- DETR (NMS-free) — Carion và cộng sự (2020): https://arxiv.org/abs/2005.12872
- YOLOv10 (NMS-free) — Wang và cộng sự (2024): https://arxiv.org/abs/2405.14458

**Công cụ & thư viện**
- torchvision — tutorial finetune object detection: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
- torchvision — trọng số Faster R-CNN (COCO_V1): https://pytorch.org/vision/stable/models/faster_rcnn.html
- torchvision.ops — `nms`, `batched_nms`, `box_iou`, `box_convert`: https://pytorch.org/vision/stable/ops.html
- uv (quản lý môi trường): https://docs.astral.sh/uv/
- PyTorch wheels CUDA 12.8 (cu128, cho GPU sm_120): https://download.pytorch.org/whl/cu128

> Danh mục arXiv đầy đủ kèm tên tác giả/hội nghị: [`../non_maximum_suppression/references.md`](../non_maximum_suppression/references.md).

> **Hướng dẫn đọc / thuyết trình.** Mỗi mục lớn có khối **"Điểm nhấn thuyết trình"**. Trọng tâm học thuật: Mục 2 (lý thuyết + trace từng bước) và Mục 5 (phân tích). Mục 3.2 giải thích **ý nghĩa từng tham số**. Mục 4 là minh chứng "có làm thật".

---

## Mục lục
1. Giới thiệu và động lực
2. Cơ sở lý thuyết: IoU, thuật toán NMS, và NMS chạy từng bước (có trace số học)
3. Phương pháp, thiết lập, và **ý nghĩa từng tham số**
4. Toàn bộ quá trình thực nghiệm (minh chứng)
5. Kết quả và phân tích chuyên sâu (nhiều ảnh test thật)
6. Các lỗi thường gặp và bài học
7. Kết luận
8. Phụ lục: tái lập, huấn luyện thêm (resume)

---

## 1. Giới thiệu và động lực

### 1.1. Bài toán
Phát hiện đối tượng trả lời *"có vật gì, ở đâu"* bằng **hộp giới hạn (bounding box)** + điểm tin cậy. Ở đây: **phát hiện người đi bộ** trong ảnh đường phố.

### 1.2. Vì sao có "đối tượng nhận dạng chồng lấp"?
Mô hình quét hàng nghìn khung neo (anchor) và được huấn luyện theo kiểu **gán nhãn một–nhiều** (mỗi vật thật khớp nhiều dự đoán), nên ở đầu ra **mỗi người bị bao bởi một chùm hộp gần trùng**. Thực nghiệm đo trực tiếp: một ảnh 7 người → **140 hộp chồng lấp** (Mục 5.2). Cần bước **hậu xử lý** gom mỗi chùm về một đại diện — đó là **NMS**.

### 1.3. Mục tiêu
1. Huấn luyện **thật** một detector; 2. Cài **greedy NMS từ đầu**, chứng minh đúng; 3. Phân tích **ý nghĩa tham số** và **lỗi cần tránh** để hiểu NMS sâu nhất.

> **Điểm nhấn thuyết trình:** *Detector cho nhiều hộp chồng nhau quanh một vật; NMS là bước "dọn dẹp" giữ một hộp tốt nhất cho mỗi vật. Gần như mọi detector (R-CNN, YOLO, SSD) đều cần.*

---

## 2. Cơ sở lý thuyết

### 2.1. IoU — thước đo độ chồng lấp
$$\text{IoU}(A,B)=\frac{\text{diện tích}(A\cap B)}{\text{diện tích}(A\cup B)}\in[0,1].$$
IoU = 0: rời nhau; = 1: trùng khít. NMS dùng ngưỡng $N_t$: hai hộp có $\text{IoU}\ge N_t$ bị coi là "cùng một vật".

### 2.2. Thuật toán Greedy NMS (chạy riêng từng lớp)
1. Sắp xếp hộp theo **điểm tin cậy giảm dần**.
2. Lấy hộp điểm cao nhất $M$ → **giữ lại**.
3. **Xoá** mọi hộp còn lại có $\text{IoU}(M,b)\ge N_t$.
4. Lặp với phần còn lại tới khi hết.

Cài đặt thật (`nms.py`) mã hoá 3 "quy tắc vàng": (1) `assert` đúng định dạng `[N,4]` xyxy; (2) guard tensor rỗng; (3) xét theo điểm **giảm dần**.

### 2.3. Độ phức tạp & vì sao tuần tự
Xấu nhất $O(n^2)$; bản chất **tuần tự** (mỗi vòng phụ thuộc vòng trước) → khó song song hoá → động lực cho Cluster-NMS/Matrix-NMS (xem links đầu file).

### 2.4. NMS chạy TỪNG BƯỚC — minh chứng số học thật
Đây là phần để **hiểu rõ cơ chế nhất**. Lấy đầu ra thô (score ≥ 0.6 → 107 hộp) trên ảnh val #92 (7 người), chạy greedy-NMS@0.5 và **ghi lại IoU thật từng bước** (`outputs/logs/nms_trace.txt`):

```
Bước 1: GIỮ box điểm=0.997
    - IoU với box (điểm=0.988) = 0.956  -> XOÁ
    - IoU với box (điểm=0.991) = 0.940  -> XOÁ
    - IoU với box (điểm=0.995) = 0.922  -> XOÁ
    ... => còn lại 92 box để xét tiếp
Bước 2: GIỮ box điểm=0.997
    - IoU với box (điểm=0.990) = 0.988  -> XOÁ
    - IoU với box (điểm=0.971) = 0.977  -> XOÁ
    ... => còn lại 75 box để xét tiếp
```
**Đọc hiểu:** mỗi bước, hộp điểm cao nhất "nuốt" tất cả hộp đè lên nó (IoU ≥ 0.5) — đó là một **người**. Sau vài bước, 107 hộp rút về số người thật. Hình hoá quá trình:

![NMS từng bước](outputs/figures/07_nms_step_by_step.png)

Quy ước màu: **xanh dương** = hộp đang được chọn (điểm cao nhất còn lại); **đỏ nét đứt** = hộp bị xoá ở bước này (IoU ≥ 0.5 với hộp xanh dương); **xanh lá** = đã giữ ở các bước trước; **xám chấm** = còn chờ xét.

> **Điểm nhấn thuyết trình:** *NMS = "người điểm cao nhất giữ chỗ, mọi hộp trùng lên người đó bị xoá, lặp lại cho người tiếp theo". Trace cho thấy IoU các hộp trùng lên tới 0.92–0.99 nên bị xoá đúng.*

---

## 3. Phương pháp, thiết lập, và ý nghĩa từng tham số

| Hạng mục | Lựa chọn | Lý do |
|---|---|---|
| Dữ liệu | Penn-Fudan (170 ảnh, 1 lớp `person`) | nhỏ, người chen/đè nhau → lý tưởng cho NMS; train nhanh |
| Mô hình | `fasterrcnn_resnet50_fpn` (COCO-pretrained) → 2 lớp | finetune hội tụ nhanh; có NMS nội bộ để "mở ra" nghiên cứu |
| Chia dữ liệu | seed=1 → train 120 / val 50 | cố định để tái lập |
| Phần cứng | RTX 5080 (sm_120), torch 2.11.0+cu128 | wheel CUDA 12.8 bắt buộc cho GPU Blackwell |

### 3.1. Finetune từ đâu? (điểm khởi đầu của trọng số)
Mô hình **không** train từ số 0. Ta **finetune** từ trọng số sẵn có:
- **Kiến trúc:** Faster R-CNN, backbone **ResNet-50 + FPN**.
- **Trọng số khởi đầu:** `fasterrcnn_resnet50_fpn(weights="DEFAULT")` = `FasterRCNN_ResNet50_FPN_Weights.DEFAULT` (**COCO_V1**) — pretrain trên **COCO train2017** (80 lớp). torchvision tự tải `fasterrcnn_resnet50_fpn_coco-258fb6c6.pth` từ download.pytorch.org/models (thấy trong log epoch 0).
- **Giữ vs thay:** giữ nguyên backbone/FPN/RPN/RoI (trọng số COCO); chỉ **thay đầu** `roi_heads.box_predictor` → `FastRCNNPredictor(in_features, 2)` (nền + `person`), khởi tạo ngẫu nhiên rồi train tiếp. Mã: `model.py`.
- **Vì sao:** COCO đã có lớp `person` → đặc trưng rất phù hợp → AP@0.5 ≈ 0.99 **ngay epoch 0**, hội tụ ~2 phút.

### 3.2. Ý nghĩa từng tham số (đã dùng trong dự án)

**A. Tham số huấn luyện (`train.py`)**
| Tham số | Giá trị | Ý nghĩa | Nếu chỉnh |
|---|---|---|---|
| `epochs` | 20 | số lần quét toàn bộ tập train | quá ít → học chưa tới (underfit); quá nhiều → tốn giờ, có thể overfit. Ở đây hội tụ ~epoch 8 |
| `batch_size` | 2 | số ảnh mỗi bước cập nhật | lớn → gradient mượt nhưng tốn VRAM (ảnh detection rất lớn); 2 là mặc định tutorial |
| `lr` (learning rate) | 0.005 | độ lớn mỗi bước cập nhật trọng số | quá cao → loss dao động/phân kỳ; quá thấp → học chậm |
| `momentum` | 0.9 | quán tính SGD (nhớ hướng cũ) | giúp vượt vùng phẳng, hội tụ mượt; quá cao → vọt lố |
| `weight_decay` | 5e-4 | phạt trọng số lớn (chính quy hoá L2) | chống overfit; quá lớn → mô hình "cứng", underfit |
| `StepLR(step=epochs//3, γ=0.1)` | step=6 | cứ 6 epoch nhân lr ×0.1 | giảm lr giai đoạn cuối để tinh chỉnh; nếu giảm quá sớm → "đóng băng" |
| warmup (epoch 0) | LinearLR 1e-3→1 | tăng dần lr ở đầu | tránh sốc gradient lúc trọng số đầu mới còn ngẫu nhiên |
| AMP (`autocast`+`GradScaler`) | bật | tính toán nửa độ chính xác (fp16) | nhanh hơn, ít VRAM trên 5080; `GradScaler` chống "underflow" gradient fp16 |
| `val_size` / `seed` | 50 / 1 | cố định tập val & cách chia | so sánh công bằng, tái lập được |

**B. Tham số của detector / khâu hậu xử lý (`roi_heads`, dùng ở `study_nms.py`)**
| Tham số | Giá trị (khi "mở raw") | Ý nghĩa | Tác động |
|---|---|---|---|
| `score_thresh` | 0.05 | ngưỡng điểm tối thiểu để giữ một detection | cao → ít hộp, sạch, nhưng dễ **sót vật mờ/xa** (xem Mục 5.5) |
| `nms_thresh` | đặt **1.0** (tắt) | ngưỡng IoU của NMS **nội bộ** mô hình | mặc định 0.5; đặt 1.0 để **lộ toàn bộ hộp thô** phục vụ nghiên cứu |
| `detections_per_img` | 300 | số hộp tối đa giữ mỗi ảnh | mặc định 100; tăng để thấy rõ nhiều hộp chồng |

**C. Tham số NGÔI SAO của NMS (`nms.py`)**
| Tham số | Ý nghĩa | Phân tích |
|---|---|---|
| `iou_thr` ($N_t$) | hai hộp có IoU ≥ $N_t$ thì coi là cùng vật → xoá hộp điểm thấp hơn | quá thấp → gom mạnh, rủi ro gộp nhầm người sát nhau (rớt recall ở cảnh đông); quá cao → giữ lại nhiều hộp trùng (rác). Quét thực nghiệm ở Mục 5.4 |

**D. Độ đo**
- **IoU**: độ chồng lấp hai hộp, [0,1]. **AP@0.5**: AP ở ngưỡng IoU dễ (0.5) — "có đúng vật không". **mAP@[.5:.95]**: trung bình AP qua IoU 0.50→0.95 — đòi hỏi hộp **khít** (chuẩn COCO). **recall**: tỉ lệ vật thật được tìm thấy.

---

## 4. Toàn bộ quá trình thực nghiệm (minh chứng "có làm")

**Bước 1 — Kiểm tra GPU (gate).** `uv run python check_env.py`:
```
torch 2.11.0+cu128 | torchvision 0.26.0+cu128
cuda available: True | device: NVIDIA GeForce RTX 5080 | cap: (12, 0)
cuda matmul OK: True
```
*Ý nghĩa:* xác nhận wheel cu128 chạy được trên GPU Blackwell trước khi train (tránh train nhầm trên CPU).

**Bước 2 — Tải dữ liệu.** `uv run python download_data.py` → `images: 170 masks: 170` (zip 53.7 MB). Box suy ra từ mask (`masks_to_boxes`), lọc hộp suy biến.

**Bước 3 — Huấn luyện 20 epoch.** `uv run python train.py --epochs 20` (≈2 phút, ~5.5 s/epoch). Toàn bộ lịch sử thật:

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

**Hai điều chỉnh phương pháp (và vì sao):**
1. **Lịch LR scale theo epoch** (`step_size=epochs//3`): khi train 20 epoch, lr giảm dần đều thay vì "đóng băng" sớm.
2. **Chọn best theo mAP@[.5:.95], không theo AP@0.5:** vì AP@0.5 **bão hoà ~0.99** (gần như phẳng) nên không phân biệt được epoch tốt; mAP@[.5:.95] mới phản ánh độ chính xác **định vị**. ⇒ `best.pth` = **epoch 8** (mAP 0.831).

**Bước 4 — Sinh hình & bảng.** `uv run python study_nms.py` và `uv run python nms_deep_dive.py` → in `NMS self-check OK`, tạo 8 hình + `nms_sweep.csv` + `nms_trace.txt`.

---

## 5. Kết quả và phân tích chuyên sâu

### 5.1. Chất lượng detector
- **AP@0.5 ≈ 0.99** từ epoch 0: "có người ở đâu" gần như đúng hết ở ngưỡng dễ.
- **mAP@[.5:.95]** tăng 0.70 → **0.83** rồi **bão hoà** từ ~epoch 8 ⇒ mô hình **hội tụ**, train thêm lợi ích không đáng kể (minh chứng 20 epoch là **đủ**).
- **Phân tích:** AP@0.5 bão hoà ⇒ **không** dùng để so sánh mô hình trong bài này; phải dùng mAP@[.5:.95]. Đây cũng là lý do COCO lấy mAP@[.5:.95] làm chỉ số chính.

### 5.2. NMS gom đối tượng chồng lấp — trước/sau
![Trước và sau NMS](outputs/figures/02_before_after_nms.png)

Ảnh val đông nhất (7 người): tắt NMS cuối → **140 hộp** chồng chất (trái); áp greedy-NMS@0.5 → còn **8 hộp** sạch (phải). *(Trung thực: NMS tầng RPN vẫn bật; ta nghiên cứu NMS ở đầu phát hiện cuối.)*

### 5.3. Ảnh test THẬT trên nhiều ảnh (không cherry-pick)
![Gallery 4 ảnh val](outputs/figures/06_gallery_before_after.png)

NMS hoạt động nhất quán trên **4 ảnh val khác nhau**: cột trái chùm hộp thô, cột phải kết quả sạch sau NMS@0.5 — chứng tỏ không phải ăn may trên một ảnh.

### 5.4. Ảnh hưởng ngưỡng IoU — phân tích trade-off
![Sweep IoU](outputs/figures/03_iou_threshold_sweep.png)

Số hộp giữ lại trên ảnh đông theo ngưỡng: **8 (0.3) → 10 (0.5) → 11 (0.7) → 42 (0.9)**. Bảng trên **toàn val** (`nms_sweep.csv`):

| Ngưỡng IoU | AP@0.5 | Recall | Số hộp TB/ảnh sau NMS |
|---|---|---|---|
| 0.3 | 0.9936 | 1.000 | 2.86 |
| 0.4 | 0.9929 | 1.000 | 2.92 |
| 0.5 | 0.9925 | 1.000 | 3.28 |
| 0.6 | 0.9916 | 1.000 | 3.92 |
| 0.7 | 0.9898 | 1.000 | 5.18 |
| 0.9 | 0.9469 | 1.000 | 21.68 |

**Phân tích:** số hộp tăng đơn điệu (2.86 → 21.68) — ngưỡng cao "khoan dung" → nhiều hộp trùng sống sót, thành **dương tính giả** → **AP@0.5 giảm** (0.9936 → 0.9469). ⇒ **vùng tốt ≈ 0.3–0.5**.

### 5.5. Ảnh hưởng score_threshold
![Score threshold](outputs/figures/08_score_threshold_effect.png)

Lọc theo điểm **sau** NMS@0.5: **0.05 → 9 hộp, 0.3 → 8, 0.5 → 8, 0.7 → 7**. Ngưỡng cao loại được hộp điểm thấp (người mờ/xa) → ít dương tính giả nhưng **dễ sót vật**. Đây là "núm vặn" precision↔recall thứ hai bên cạnh ngưỡng IoU.

### 5.6. Vì sao Recall = 1.0 ở mọi ngưỡng? (nhận xét trung thực)
Trên Penn-Fudan val, người tách nhau khá rõ + detector mạnh → ngay cả ngưỡng 0.3 cũng **không xoá nhầm người thật** → recall = 1.0. Mặt trái "ngưỡng thấp làm rớt recall" sẽ rõ trên dữ liệu **cảnh đông gắt** (CrowdHuman/CityPersons), nơi hai người che nhau có IoU hộp cao. *Kết luận về NMS phụ thuộc độ "đông" của dữ liệu.*

> **Điểm nhấn thuyết trình:** *Hai núm vặn: ngưỡng IoU (gom mạnh hay nhẹ) và score_threshold (giữ hộp tự tin tới đâu). Ở dữ liệu này, IoU 0.3–0.5 + score ~0.5 là cân bằng tốt.*

---

## 6. Các lỗi thường gặp và bài học ("cái sai cần tránh")

### 6.1. NMS không phân lớp (class-agnostic) xoá nhầm vật khác lớp — đã đo
![Lỗi class-agnostic](outputs/figures/04_pitfall_classagnostic.png)
Hai hộp đè nhau khác lớp (person vs dog); NMS gộp chung mọi lớp xoá mất một (giữ 1/2). **Đúng:** NMS **riêng từng lớp** — `torchvision.ops.batched_nms` (giữ 2/2).

### 6.2. Sai định dạng hộp (xywh vs xyxy) — đã đo
![Lỗi định dạng](outputs/figures/05_pitfall_format_xywh.png)
Đưa `xywh` vào hàm mong `xyxy` → hộp phình/tràn khung, IoU vô nghĩa. **Đúng:** chuyển `box_convert` trước; `greedy_nms` có `assert` chặn.

### 6.3. Ngưỡng IoU quá thấp ở cảnh đông → mất vật thật
Hai người che nhau IoU hộp cao; ngưỡng thấp gộp nhầm → giảm recall. **Đúng:** tăng ngưỡng vùng đông, hoặc Soft-/Adaptive-/DIoU-NMS (links đầu file).

### 6.4. Quên lọc điểm / quên sắp theo điểm
NMS trên cả nghìn hộp điểm thấp → chậm, giữ rác; không sort → "đại diện" sai. **Đúng:** lọc `score_threshold` trước; luôn xét điểm giảm dần.

### 6.5. NMS trên tensor rỗng
Ảnh không có vật → vỡ shape/argmax. **Đúng:** guard rỗng (đã có).

### 6.6. Xếp hạng bằng điểm phân loại thay vì độ chính xác định vị
Giữ hộp "tự tin về lớp" nhưng định vị lệch. Hướng khắc phục: IoU-Net (link đầu file).

### 6.7. Chọn mô hình bằng chỉ số bão hoà (bài học từ chính dự án)
Chọn `best` theo AP@0.5 (bão hoà ~0.99) → chọn nhầm epoch. **Đúng:** chọn theo mAP@[.5:.95] (đã sửa trong `train.py`).

### 6.8. Dùng API AMP lỗi thời
`torch.cuda.amp.*` deprecated. **Đúng:** `torch.amp.autocast('cuda')` + `torch.amp.GradScaler('cuda')` (đã dùng).

> **Điểm nhấn thuyết trình:** *Hai lỗi hay gặp nhất: (1) quên NMS theo từng lớp, (2) nhầm định dạng hộp xywh/xyxy — đều làm sai mà không báo lỗi.*

---

## 7. Kết luận
- **Huấn luyện thật** Faster R-CNN (finetune từ COCO_V1) trên Penn-Fudan, 20 epoch (~2 phút, RTX 5080): **AP@0.5 ≈ 0.99**, **mAP@[.5:.95] ≈ 0.83** (best epoch 8); hội tụ rõ → 20 epoch đủ.
- **NMS gom hiệu quả** đối tượng chồng lấp: **140 → 8 hộp**; trace số học cho thấy các hộp trùng có IoU 0.92–0.99 nên bị xoá đúng.
- **Tham số chính:** ngưỡng IoU chi phối số hộp (2.86 → 21.68) và precision (AP@0.5 0.9936 → 0.9469); score_threshold chi phối precision↔recall.
- Rút ra **8 lỗi cần tránh**, có minh chứng thực nghiệm.

## 8. Phụ lục — tái lập và huấn luyện thêm
```bash
cd computer_vision/nms_training
uv run python check_env.py
uv run python download_data.py
uv run python train.py --epochs 20
uv run python study_nms.py          # hình 01–05 + nms_sweep.csv
uv run python nms_deep_dive.py      # hình 06–08 + nms_trace.txt
# Train THÊM (không làm lại từ đầu):
uv run python train.py --resume outputs/checkpoints/last.pth --epochs 24
```
Nhật ký đầy đủ + bảng từng epoch: [`his.md`](his.md). Lý thuyết NMS đầy đủ: [`../non_maximum_suppression/nghien_cuu_NMS.md`](../non_maximum_suppression/nghien_cuu_NMS.md).
