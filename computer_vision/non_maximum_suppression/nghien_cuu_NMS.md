# Nghiên cứu kỹ thuật Non-Maximum Suppression (NMS) và ứng dụng gom các đối tượng nhận dạng chồng lấp

> Tài liệu nghiên cứu chuẩn bị cho bài thuyết trình môn Computer Vision.
> Mọi phát biểu về thuật toán/độ phức tạp/kết quả thực nghiệm đều có trích dẫn nguồn — xem `references.md`.

---

## Mục lục
1. [Đặt vấn đề: vì sao cần gom các đối tượng chồng lấp](#1)
2. [Nền tảng: độ đo IoU](#2)
3. [NMS cổ điển (Greedy NMS): thuật toán và độ phức tạp](#3)
4. [Vai trò của NMS trong pipeline phát hiện đối tượng](#4)
5. [Hạn chế của Greedy NMS](#5)
6. [Các biến thể cải tiến](#6)
7. [Ứng dụng cụ thể: gom đối tượng nhận dạng chồng lấp](#7)
8. [Hướng hiện đại: detector không cần NMS (NMS-free)](#8)
9. [Triển khai thực tế và mã nguồn minh hoạ](#9)
10. [Đánh giá và ảnh hưởng của tham số](#10)
11. [Kết luận và hướng phát triển](#11)

---

<a id="1"></a>
## 1. Đặt vấn đề: vì sao cần gom các đối tượng chồng lấp

Một bộ phát hiện đối tượng (object detector) hiện đại không sinh ra **một** hộp giới hạn (bounding box) cho mỗi vật thể, mà sinh ra **rất nhiều** hộp ứng viên phủ chồng lên nhau quanh cùng một vật thể. Nguyên nhân mang tính cấu trúc:

- **Anchor dày đặc / cửa sổ trượt:** các detector như Faster R-CNN, SSD, RetinaNet đặt hàng chục nghìn anchor box ở nhiều vị trí, tỉ lệ và kích thước; nhiều anchor cạnh nhau cùng "bắt" được một vật thể.
- **Dự đoán đa tỉ lệ (multi-scale):** cùng một vật thể được phát hiện ở nhiều tầng đặc trưng (feature pyramid).
- **Gán nhãn một-nhiều khi huấn luyện (one-to-many label assignment):** để mô hình hội tụ tốt và có recall cao, mỗi vật thể thật (ground truth) được gán cho **nhiều** dự đoán dương tính trong lúc huấn luyện. Hệ quả tất yếu: lúc suy luận, mỗi vật thể sinh ra một cụm box trùng lặp.

Vì vậy đầu ra "thô" của detector chứa các **đối tượng nhận dạng chồng lấp** — nhiều box mô tả cùng một vật. Cần một bước **hậu xử lý (post-processing)** để **gom mỗi cụm box trùng về một đại diện duy nhất**, giữ lại đúng số lượng vật thể thực. Đó chính là nhiệm vụ của **Non-Maximum Suppression (NMS)**.

> Định nghĩa kinh điển (Hosang và cộng sự, CVPR 2017): *"NMS là thuật toán hậu xử lý chịu trách nhiệm hợp nhất (merging) tất cả các phát hiện thuộc về cùng một đối tượng."* NMS tiêu chuẩn dựa trên **gom cụm tham lam (greedy clustering)** với một ngưỡng khoảng cách cố định.

Phát biểu lại đúng chủ đề báo cáo: **NMS là kỹ thuật gom các đối tượng nhận dạng chồng lấp.** Mỗi cụm box có độ chồng lấp cao được coi là "cùng một đối tượng" và được rút gọn về một box (hoặc một box hợp nhất). Phần còn lại của tài liệu phân tích kỹ cơ chế gom này, các điểm yếu của nó, và các biến thể cải tiến.

---

<a id="2"></a>
## 2. Nền tảng: độ đo IoU (Intersection over Union)

Để biết "hai box có cùng mô tả một vật hay không", cần một thước đo **độ chồng lấp**. Thước đo chuẩn là **IoU**:

$$\text{IoU}(A, B) = \frac{\text{area}(A \cap B)}{\text{area}(A \cup B)} = \frac{\text{area}(A \cap B)}{\text{area}(A) + \text{area}(B) - \text{area}(A \cap B)}$$

- Miền giá trị: $\text{IoU} \in [0, 1]$. Bằng 0 khi hai box rời nhau, bằng 1 khi trùng khít.
- Với box biểu diễn bởi toạ độ góc $(x_1, y_1, x_2, y_2)$, phần giao là hình chữ nhật:
  $x_1^\cap = \max(x_1^A, x_1^B)$, $y_1^\cap = \max(y_1^A, y_1^B)$, $x_2^\cap = \min(x_2^A, x_2^B)$, $y_2^\cap = \min(y_2^A, y_2^B)$; diện tích giao $= \max(0, x_2^\cap - x_1^\cap)\cdot\max(0, y_2^\cap - y_1^\cap)$.

IoU là **tiêu chí quyết định** trong NMS: hai box có IoU vượt ngưỡng $N_t$ được xem là cùng một vật, nên một trong hai bị loại/giảm điểm.

**Hạn chế của IoU thuần** (động lực cho các biến thể ở Mục 6): khi hai box **không giao nhau**, IoU luôn bằng 0 và không phản ánh "xa hay gần". Các độ đo mở rộng khắc phục điều này:
- **GIoU** (Generalized IoU, Rezatofighi và cộng sự, CVPR 2019): trừ thêm phần diện tích của hộp bao nhỏ nhất $C$ chứa cả hai box nhưng không thuộc hợp của chúng — nhạy với trường hợp không chồng lấp.
- **DIoU** (Distance-IoU, Zheng và cộng sự, AAAI 2020): bổ sung **khoảng cách tâm** đã chuẩn hoá: $\text{DIoU} = \text{IoU} - \rho^2(\mathbf{c}_A, \mathbf{c}_B)/c^2$, với $\rho$ là khoảng cách Euclid giữa hai tâm và $c$ là đường chéo hộp bao nhỏ nhất.
- **CIoU** (Complete IoU, cùng nhóm tác giả): thêm yếu tố **tỉ lệ khung (aspect ratio)** vào DIoU.

Ba yếu tố hình học — **diện tích chồng lấp, khoảng cách tâm, tỉ lệ khung** — sẽ được tái sử dụng làm tiêu chí khử trong DIoU-NMS và Cluster-NMS (Mục 6).

---

<a id="3"></a>
## 3. NMS cổ điển (Greedy NMS): thuật toán và độ phức tạp

### 3.1. Ý tưởng

Greedy NMS xử lý **từng lớp (per-class) độc lập**. Trong một lớp:
1. Sắp xếp tất cả box theo điểm tin cậy (confidence score) giảm dần.
2. Chọn box có điểm cao nhất $M$, đưa vào danh sách giữ lại (keep).
3. Loại bỏ mọi box còn lại có $\text{IoU}(M, b_i) \ge N_t$ (coi như cùng vật với $M$).
4. Lặp lại với các box chưa bị loại cho đến khi hết.

Đây đúng nghĩa là **gom cụm tham lam**: mỗi vòng lặp chọn "đỉnh" có điểm cao nhất làm đại diện cụm, rồi "nuốt" tất cả box chồng lấp đủ nhiều với nó.

### 3.2. Mã giả

```
NMS(B, S, N_t):
    # B: danh sách box, S: điểm tương ứng, N_t: ngưỡng IoU
    D ← ∅                      # tập kết quả giữ lại
    while B ≠ ∅:
        m ← argmax(S)          # chỉ số box điểm cao nhất
        M ← B[m]
        D ← D ∪ {M};  B ← B − {M};  S ← S − {S[m]}
        for b_i in B:
            if IoU(M, b_i) ≥ N_t:
                B ← B − {b_i};  S ← S − {s_i}   # XOÁ HẲN (hard suppression)
    return D
```

Dòng quan trọng nhất là bước **xoá hẳn** (`hard threshold`): điểm của box lân cận bị đặt về 0 (loại khỏi danh sách). Đây vừa là nguồn gốc sự đơn giản, vừa là nguồn gốc các hạn chế ở Mục 5.

### 3.3. Độ phức tạp

- Mỗi vòng lặp loại ít nhất một box và so sánh box đỉnh với các box còn lại. Trường hợp xấu nhất là $O(n^2)$ với $n$ là số box trong một lớp (sau khi đã lọc theo ngưỡng điểm). Thực tế $n$ thường nhỏ nhờ lọc trước theo `score_threshold` và `top-k`.
- Bản chất **tuần tự (sequential)** của vòng lặp tham lam khiến nó **khó song song hoá hiệu quả trên GPU** — đây là động lực trực tiếp cho Matrix-NMS và Cluster-NMS (Mục 6).

### 3.4. Ba tham số then chốt
| Tham số | Vai trò | Ảnh hưởng |
|---|---|---|
| `score_threshold` | Lọc bỏ box điểm thấp **trước** khi chạy NMS | Cao → ít box, nhanh, nhưng dễ bỏ sót vật mờ |
| `iou_threshold` ($N_t$) | Ngưỡng coi hai box là cùng vật | Thấp → khử mạnh (dễ xoá nhầm vật chồng nhau); Cao → giữ nhiều (dễ trùng lặp) |
| `top_k` / `max_detections` | Giới hạn số box giữ lại | Khống chế chi phí và đầu ra |

---

<a id="4"></a>
## 4. Vai trò của NMS trong pipeline phát hiện đối tượng

- **Hai giai đoạn (two-stage) — họ R-CNN:** NMS xuất hiện **hai lần**: (a) trong mạng đề xuất vùng (RPN) để khử trùng các proposal; (b) ở đầu ra cuối cùng để khử trùng các detection theo từng lớp. NMS đã là thành phần chuẩn của detector từ R-CNN (Girshick và cộng sự, 2014).
- **Một giai đoạn (one-stage) — YOLO, SSD, RetinaNet:** NMS là bước hậu xử lý **cuối cùng**, sau khi giải mã anchor thành box và gán điểm.
- **Tính tất yếu:** Với detector dùng **gán nhãn một-nhiều**, mỗi vật thật ứng với nhiều dự đoán → **bắt buộc** phải có NMS để khử trùng. Nói cách khác, sự tồn tại của NMS gắn liền với chiến lược gán nhãn một-nhiều khi huấn luyện. Đây chính là điểm mà các detector NMS-free (Mục 8) tấn công.

---

<a id="5"></a>
## 5. Hạn chế của Greedy NMS

1. **Ngưỡng cứng làm mất vật thể thật chồng lấp (giảm recall).** Khi hai vật thể **thật** đứng sát/che nhau (đám đông người, xe cộ), box của vật thứ hai có IoU cao với box vật thứ nhất nên bị **xoá hẳn**. Soft-NMS chỉ ra: ngưỡng cứng "đặt điểm của các phát hiện lân cận về 0, gây sụt average precision nếu một vật thật rơi vào vùng chồng lấp đó".
2. **Đánh đổi recall–precision bị khoá bởi một ngưỡng $N_t$ duy nhất.** $N_t$ thấp → khử mạnh → bỏ sót vật trong cảnh đông; $N_t$ cao → giữ nhiều → tăng dương tính giả. Không có một $N_t$ tối ưu cho mọi vùng ảnh. Đây là "đánh đổi recall–precision bị ép buộc" mà Hosang và cộng sự nêu ra.
3. **Chỉ dùng điểm tin cậy + IoU,** bỏ qua hình dạng, ngữ cảnh, mật độ — nên không phân biệt được "hai box trùng vì cùng một vật" với "hai box chồng vì hai vật thật che nhau".
4. **Điểm tin cậy phân loại không đồng nghĩa định vị tốt.** Box điểm phân loại cao chưa chắc định vị chính xác nhất; greedy NMS lại luôn ưu tiên điểm phân loại (động lực cho IoU-Net — Mục 6.9).
5. **Khó song song hoá** do bản chất tuần tự (động lực cho Matrix/Cluster-NMS).

Các benchmark cảnh đông như **CrowdHuman** và **CityPersons** là nơi điểm yếu (1)–(2) bộc lộ rõ nhất.

---

<a id="6"></a>
## 6. Các biến thể cải tiến

### 6.1. Soft-NMS (Bodla và cộng sự, ICCV 2017)
Thay vì **xoá hẳn**, Soft-NMS **giảm điểm** của box lân cận theo một hàm liên tục của độ chồng lấp với box đỉnh $M$ — "không vật nào bị loại bỏ trong quá trình". Hai dạng hàm phạt:
- **Tuyến tính:** $s_i \leftarrow s_i\,(1 - \text{IoU}(M, b_i))$ nếu $\text{IoU} \ge N_t$.
- **Gaussian:** $s_i \leftarrow s_i \cdot \exp\!\big(-\text{IoU}(M, b_i)^2 / \sigma\big)$ với mọi box.

Box chồng lấp nhiều bị phạt mạnh nhưng vẫn tồn tại với điểm thấp hơn; nếu thực ra là vật thật, nó còn cơ hội được giữ. Soft-NMS **cùng độ phức tạp** với NMS, **không cần huấn luyện thêm**, gắn vào mọi pipeline; cải thiện ổn định mAP kiểu COCO (báo cáo +1.7% trên PASCAL VOC 2007 cho cả R-FCN và Faster-RCNN; +1.1–1.3% trên MS-COCO).

### 6.2. NMS hợp nhất bằng bỏ phiếu hộp (Weighted / Box-voting NMS)
Thay vì chọn **đúng một** box làm đại diện, ta **hợp nhất cả cụm** box chồng lấp thành một box tinh hơn bằng **trung bình có trọng số** toạ độ (trọng số theo điểm tin cậy và/hoặc IoU). Đây là cách "gom đối tượng" theo nghĩa đen nhất: cụm nhiều box → một box trung bình, ổn định hơn so với chỉ lấy box điểm cao nhất.

### 6.3. Adaptive-NMS (Liu, Huang, Wang — CVPR 2019)
Dùng **ngưỡng khử động theo mật độ**: vùng đông người dùng $N_t$ **cao hơn** (để không xoá nhầm các vật thật chen nhau), vùng thưa dùng $N_t$ thấp. Một **mạng con (subnetwork)** học **điểm mật độ (density)** và có thể nhúng vào cả detector một và hai giai đoạn. Đạt SOTA trên **CityPersons** và **CrowdHuman** — chính các cảnh đông mà greedy NMS yếu nhất.

### 6.4. DIoU-NMS (Zheng và cộng sự, AAAI 2020)
Thay tiêu chí IoU bằng **DIoU** (IoU + khoảng cách tâm). Trực giác: hai box có IoU cao **nhưng tâm cách xa** nhiều khả năng là **hai vật khác nhau** (che nhau), nên **không nên** khử. DIoU-NMS giữ lại các box này tốt hơn, đặc biệt với che khuất (occlusion). Tích hợp dễ dàng vào pipeline hiện có.

### 6.5. Cluster-NMS (Zheng và cộng sự, 2020/2021)
**Gom cụm box một cách ngầm định** thông qua phép toán trên **ma trận IoU**, thường cần **ít vòng lặp hơn** và **chạy thuần trên GPU** nên rất nhanh. Có thể nhúng thêm yếu tố hình học (DIoU) để tăng cả AP lẫn AR. Ví dụ với YOLACT trên MS-COCO: +1.7 AP (detection) và +0.9 AP (instance segmentation), đạt 27.1 FPS trên một GPU GTX 1080Ti.

### 6.6. Matrix-NMS (Wang và cộng sự, SOLOv2, NeurIPS 2020)
Song song hoá tư tưởng Soft-NMS bằng **phép toán ma trận một lần (one-shot)**, áp cho **mask** trong phân đoạn thực thể. Giảm mạnh chi phí thời gian của bước NMS mà vẫn cho kết quả tốt — giải quyết nút cổ chai "mask NMS chậm" của SOLO gốc.

### 6.7. Fast-NMS (YOLACT, Bolya và cộng sự, ICCV 2019)
Dùng ma trận IoU **tam giác trên** để khử song song; chấp nhận loại "dư" một vài box để đổi lấy **tốc độ** — phù hợp suy luận thời gian thực.

### 6.8. Learning NMS / GossipNet (Hosang và cộng sự, CVPR 2017)
Thay vì thuật toán thủ công, **huấn luyện một mạng nơ-ron thực hiện NMS** chỉ từ **box và điểm**. GossipNet khử tốt hơn các cực đại trở thành dương tính giả của greedy NMS (ví dụ box bắt vào một phần người hoặc box quá lớn), cải thiện ~1 điểm AP. Đây là tiền đề cho hướng "NMS học được".

### 6.9. IoU-Net (Jiang và cộng sự, ECCV 2018)
Dự đoán **độ tin cậy định vị (localization confidence)** = IoU dự kiến với ground truth, rồi dùng nó (thay cho điểm phân loại) làm tiêu chí xếp hạng trong NMS → giữ lại box **định vị tốt nhất** chứ không chỉ box "tự tin nhất về lớp".

### Bảng tổng hợp
| Phương pháp | Ý tưởng cốt lõi | Ưu điểm | Hạn chế / Chi phí |
|---|---|---|---|
| Greedy NMS | Xoá hẳn box IoU ≥ $N_t$ | Đơn giản, nhanh, không cần train | Mất vật chồng lấp; ngưỡng cứng; khó song song |
| Soft-NMS | Giảm điểm theo IoU (linear/Gaussian) | +mAP, không train, dễ ghép | Vẫn cần chọn $\sigma$/$N_t$ |
| Weighted/Box-voting | Trung bình có trọng số cả cụm | Box hợp nhất ổn định hơn | Thêm tính toán hợp nhất |
| Adaptive-NMS | Ngưỡng động theo mật độ | SOTA cảnh đông | Cần mạng con học density |
| DIoU-NMS | Tiêu chí IoU + khoảng cách tâm | Tốt khi che khuất | Thêm tính khoảng cách tâm |
| Cluster-NMS | Gom cụm bằng ma trận IoU, GPU | Rất nhanh, +AP/+AR | Cần triển khai ma trận |
| Matrix-NMS | Soft-NMS song song một lần (mask) | Nhanh cho segmentation | Thiết kế cho mask |
| Fast-NMS | Ma trận IoU tam giác trên | Nhanh, thời gian thực | Loại "dư" vài box |
| Learning NMS | Mạng học thực hiện NMS | Xử lý occlusion tốt | Cần huấn luyện, phức tạp |
| IoU-Net | Xếp hạng theo localization confidence | Giữ box định vị tốt | Thêm nhánh dự đoán IoU |

---

<a id="7"></a>
## 7. Ứng dụng cụ thể: gom đối tượng nhận dạng chồng lấp

Đây là phần liên hệ trực tiếp với đề bài. Ba tình huống "chồng lấp" và cách NMS (và biến thể) xử lý:

1. **Trùng lặp quanh một vật (redundant detections).** Nhiều box bám một khuôn mặt/biển báo. → Greedy NMS hoặc Soft-NMS gom về một box; muốn box tinh hơn thì dùng Box-voting để hợp nhất cụm.
2. **Cảnh đông, vật thật che nhau (crowded / occlusion).** Người trong đám đông, xe trong bãi. → Greedy NMS dễ xoá nhầm; nên dùng **Adaptive-NMS** (ngưỡng theo mật độ) hoặc **DIoU-NMS** (phân biệt theo khoảng cách tâm) hoặc **Soft-NMS**.
3. **Phân đoạn thực thể chồng lấp (mask).** → **Matrix-NMS** xử lý nhanh trên mask.

**Quy tắc chọn biến thể theo bài toán:**
- Cảnh thông thường, cần đơn giản: **Greedy NMS** (hoặc **Soft-NMS** để nhặt thêm mAP miễn phí).
- Cảnh đông/che khuất: **Adaptive-NMS** hoặc **DIoU-NMS**.
- Cần tốc độ GPU/thời gian thực: **Cluster-NMS** / **Fast-NMS**.
- Phân đoạn thực thể: **Matrix-NMS**.
- Muốn ưu tiên độ chính xác định vị: kết hợp **IoU-Net** (xếp hạng theo localization confidence).

---

<a id="8"></a>
## 8. Hướng hiện đại: detector không cần NMS (NMS-free)

NMS là bước **không khả vi**, là rào cản cho huấn luyện đầu-cuối (end-to-end) và làm tăng độ trễ suy luận. Hai hướng tiêu biểu loại bỏ NMS bằng cách thay đổi **chiến lược gán nhãn**:

- **DETR (Carion và cộng sự, ECCV 2020):** coi phát hiện đối tượng là bài toán **dự đoán tập hợp (set prediction)** và dùng **so khớp song ánh (bipartite matching)** kiểu Hungarian giữa dự đoán và ground truth khi huấn luyện. Mỗi vật thật khớp **đúng một** truy vấn (query) → không sinh trùng lặp → **không cần NMS**.
- **YOLOv10 (Wang và cộng sự, NeurIPS 2024):** đề xuất **gán đôi nhất quán (consistent dual assignments)**: nhánh **một-nhiều** khi huấn luyện để tăng recall, nhánh **một-một** dùng khi suy luận để bảo đảm độ chính xác — nhất quán giữa huấn luyện và suy luận, nhờ đó **bỏ được NMS** và giảm độ trễ. (YOLOv10-B giảm 46% độ trễ so với YOLOv9-C với cùng hiệu năng.)

Thông điệp phương pháp luận: **NMS gắn liền với gán nhãn một-nhiều**; nếu giải quyết việc khử trùng ngay ở mức huấn luyện (one-to-one), có thể loại bỏ NMS khỏi suy luận.

---

<a id="9"></a>
## 9. Triển khai thực tế và mã nguồn minh hoạ

### 9.1. Greedy NMS từ đầu (NumPy)
```python
import numpy as np

def nms(boxes, scores, iou_thr=0.5):
    """boxes: (N,4) định dạng xyxy; scores: (N,). Trả về chỉ số box giữ lại."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]          # sắp theo điểm giảm dần
    keep = []
    while order.size > 0:
        i = order[0]                        # box điểm cao nhất = đại diện cụm
        keep.append(i)
        # IoU giữa box i và các box còn lại
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1);       h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(iou <= iou_thr)[0] + 1]   # giữ box IoU <= ngưỡng
    return keep
```

### 9.2. Soft-NMS (Gaussian) — ý tưởng giảm điểm
```python
def soft_nms_gaussian(boxes, scores, sigma=0.5, score_thr=0.001):
    boxes, scores = boxes.copy(), scores.copy()
    N = len(boxes); keep_idx = np.arange(N)
    for i in range(N):
        # đưa box điểm cao nhất trong [i:] lên vị trí i
        m = i + np.argmax(scores[i:])
        boxes[[i, m]] = boxes[[m, i]]; scores[[i, m]] = scores[[m, i]]; keep_idx[[i, m]] = keep_idx[[m, i]]
        # giảm điểm các box sau theo IoU với box i (không xoá hẳn)
        for j in range(i + 1, N):
            iou = _iou(boxes[i], boxes[j])
            scores[j] *= np.exp(-(iou ** 2) / sigma)
    sel = scores > score_thr
    return keep_idx[sel], scores[sel]
```

### 9.3. Dùng thư viện (khuyến nghị cho sản phẩm)
```python
import torch
from torchvision.ops import nms, batched_nms

keep = nms(boxes, scores, iou_threshold=0.5)            # một lớp
keep = batched_nms(boxes, scores, class_ids, 0.5)        # NMS riêng theo từng lớp
```
**Lưu ý triển khai:** (1) box định dạng `xyxy`; (2) chạy **NMS riêng theo từng lớp** (`batched_nms`) để không khử nhầm hai vật khác lớp chồng nhau; (3) lọc `score_threshold` và `top_k` **trước** để giảm chi phí; (4) ưu tiên cài đặt GPU sẵn có.

---

<a id="10"></a>
## 10. Đánh giá và ảnh hưởng của tham số

- **Chỉ số chính:** mAP (mean Average Precision) theo các ngưỡng IoU (ví dụ COCO mAP@[.5:.95]), và AR (Average Recall). Cảnh người đông thường thêm $MR^{-2}$ (log-average miss rate).
- **Ảnh hưởng của $N_t$:** $N_t$ nhỏ → loại nhiều box → tăng precision nhưng giảm recall ở cảnh đông; $N_t$ lớn → giữ nhiều box → tăng recall nhưng nhiều dương tính giả. Soft-NMS làm "mềm" sự đánh đổi này; Adaptive-NMS chọn $N_t$ theo mật độ cục bộ.
- **Thông lệ kiểm chứng:** quét $N_t \in \{0.3, 0.5, 0.7\}$, đo mAP/AR; với cảnh đông, so sánh thêm trên CrowdHuman/CityPersons.

---

<a id="11"></a>
## 11. Kết luận và hướng phát triển

- **NMS = kỹ thuật gom các đối tượng nhận dạng chồng lấp**: dựa trên IoU và điểm tin cậy, gom mỗi cụm box trùng về một đại diện.
- **Greedy NMS** đơn giản, hiệu quả, nhưng **ngưỡng cứng** làm mất vật thật trong cảnh đông và bị **khoá ở một đánh đổi recall–precision**.
- Họ biến thể giải quyết từng điểm yếu: **Soft-NMS** (giảm điểm thay vì xoá), **Adaptive-NMS** (ngưỡng theo mật độ), **DIoU-NMS** (thêm khoảng cách tâm), **Cluster/Matrix/Fast-NMS** (song song hoá GPU), **Learning NMS & IoU-Net** (học/định vị).
- **Xu hướng:** detector **NMS-free** (DETR, YOLOv10) khử trùng ngay ở mức huấn luyện bằng gán nhãn một-một, hướng tới suy luận đầu-cuối, độ trễ thấp.

> **Câu chốt:** Greedy NMS vẫn là chuẩn mực thực dụng cho phần lớn bài toán; chọn biến thể theo bối cảnh (cảnh đông → Adaptive/DIoU, tốc độ → Cluster/Matrix, mask → Matrix); và biết rằng các kiến trúc mới nhất đang dần loại bỏ NMS khỏi pipeline suy luận.
