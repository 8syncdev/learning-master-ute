# Non-Maximum Suppression (NMS)
- Kỹ thuật gom các đối tượng nhận dạng chồng lấp
- Môn: Computer Vision
- Học viên: Nguyễn Phương Anh Tú — MSHV 2611328

---

# Nội dung trình bày
- Đặt vấn đề: vì sao xuất hiện đối tượng chồng lấp
- Nền tảng IoU và thuật toán NMS cổ điển
- Hạn chế của Greedy NMS
- Các biến thể cải tiến (Soft / Adaptive / DIoU / Cluster / Matrix ...)
- Ứng dụng gom đối tượng chồng lấp
- Hướng hiện đại: detector không cần NMS
- Triển khai thực tế và kết luận

---

# Đặt vấn đề
- Detector sinh ra RẤT NHIỀU box quanh cùng một vật thể, do:
  - Anchor dày đặc và cửa sổ trượt
  - Dự đoán đa tỉ lệ (multi-scale)
  - Gán nhãn một-nhiều khi huấn luyện → nhiều dự đoán/1 vật
- Hệ quả: đầu ra "thô" chứa các đối tượng nhận dạng chồng lấp
- Cần bước hậu xử lý gom mỗi cụm box trùng về một đại diện

---

# NMS là gì
- NMS = thuật toán hậu xử lý hợp nhất mọi phát hiện thuộc CÙNG một đối tượng (Hosang, CVPR 2017)
- Bản chất: gom cụm tham lam (greedy clustering) theo ngưỡng độ chồng lấp
- Đúng chủ đề: NMS chính là kỹ thuật GOM các đối tượng nhận dạng chồng lấp
- Tiêu chí "cùng một vật": độ đo IoU vượt ngưỡng

---

# Nền tảng: độ đo IoU
- IoU = diện tích giao / diện tích hợp, giá trị trong [0, 1]
- 0 = rời nhau; 1 = trùng khít
- Là tiêu chí quyết định hai box có cùng một vật hay không
- Mở rộng khi cần: GIoU, DIoU (thêm khoảng cách tâm), CIoU (thêm tỉ lệ khung)

---

# Greedy NMS — thuật toán
- Xử lý riêng từng lớp (per-class)
- Bước 1: sắp xếp box theo điểm tin cậy giảm dần
- Bước 2: chọn box điểm cao nhất M làm đại diện, giữ lại
- Bước 3: xoá mọi box có IoU(M, b) ≥ ngưỡng Nt
- Bước 4: lặp với các box còn lại đến khi hết

---

# Greedy NMS — minh hoạ và độ phức tạp
- Mỗi vòng: box điểm cao nhất "nuốt" các box chồng lấp đủ nhiều
- Điểm mấu chốt: XOÁ HẲN (hard threshold) — điểm box lân cận về 0
- Độ phức tạp xấu nhất O(n^2); thực tế nhỏ nhờ lọc trước
- Bản chất tuần tự → khó song song hoá hiệu quả trên GPU

---

# Ba tham số then chốt
- score_threshold: lọc box điểm thấp TRƯỚC khi chạy NMS
- iou_threshold (Nt): ngưỡng coi hai box là cùng vật
  - Nt thấp → khử mạnh, dễ xoá nhầm vật chồng nhau
  - Nt cao → giữ nhiều, dễ trùng lặp
- top_k / max_detections: giới hạn số box đầu ra

---

# NMS trong pipeline phát hiện
- Hai giai đoạn (R-CNN): NMS ở RPN và ở đầu ra cuối
- Một giai đoạn (YOLO, SSD, RetinaNet): NMS là bước hậu xử lý cuối
- NMS gắn liền với gán nhãn một-nhiều khi huấn luyện
- Đây là điểm mà các detector NMS-free nhắm tới loại bỏ

---

# Hạn chế của Greedy NMS
- Ngưỡng cứng làm MẤT vật thật chồng lấp → giảm recall (cảnh đông)
- Đánh đổi recall–precision bị khoá bởi MỘT ngưỡng Nt duy nhất
- Chỉ dùng điểm tin cậy + IoU, bỏ qua mật độ/ngữ cảnh
- Điểm phân loại cao ≠ định vị tốt
- Khó song song hoá (tuần tự)

---

# Cải tiến 1: Soft-NMS
- Thay vì xoá hẳn → GIẢM ĐIỂM box lân cận theo độ chồng lấp
- Tuyến tính: s ← s(1 − IoU); Gaussian: s ← s·exp(−IoU²/σ)
- Box thật bị che vẫn còn cơ hội được giữ
- Cùng độ phức tạp, không cần huấn luyện; +1.1–1.7% mAP (VOC/COCO)

---

# Cải tiến 2: cảnh đông và che khuất
- Adaptive-NMS (CVPR 2019): ngưỡng ĐỘNG theo mật độ
  - Vùng đông → Nt cao hơn để không xoá nhầm
  - Học density bằng mạng con; SOTA CrowdHuman / CityPersons
- DIoU-NMS (AAAI 2020): tiêu chí IoU + khoảng cách tâm
  - IoU cao nhưng tâm xa → có thể là hai vật khác → không khử

---

# Cải tiến 3: tốc độ, GPU và mask
- Cluster-NMS: gom cụm bằng ma trận IoU, thuần GPU, ít vòng lặp
- Matrix-NMS (SOLOv2): Soft-NMS song song một lần cho mask
- Fast-NMS (YOLACT): ma trận IoU tam giác trên, đổi chính xác lấy tốc độ
- Phù hợp suy luận thời gian thực và phân đoạn thực thể

---

# Cải tiến 4: học và định vị
- Learning NMS / GossipNet (CVPR 2017): mạng học thực hiện NMS từ box+điểm
  - Xử lý occlusion tốt hơn, ~+1 điểm AP
- IoU-Net (ECCV 2018): xếp hạng theo localization confidence
  - Giữ box ĐỊNH VỊ tốt nhất, không chỉ box "tự tin" nhất

---

# Chọn biến thể theo bài toán
- Cảnh thường, cần đơn giản → Greedy NMS hoặc Soft-NMS
- Cảnh đông / che khuất → Adaptive-NMS hoặc DIoU-NMS
- Cần tốc độ GPU / thời gian thực → Cluster-NMS / Fast-NMS
- Phân đoạn thực thể (mask) → Matrix-NMS
- Ưu tiên độ chính xác định vị → kết hợp IoU-Net

---

# Hướng hiện đại: detector không cần NMS
- NMS là bước không khả vi, cản trở huấn luyện đầu-cuối và tăng độ trễ
- DETR (ECCV 2020): dự đoán tập hợp + so khớp song ánh → mỗi vật khớp 1 query
- YOLOv10 (NeurIPS 2024): gán đôi nhất quán (one-to-many + one-to-one)
  - Bỏ NMS khi suy luận, giảm 46% độ trễ so với YOLOv9-C
- Thông điệp: khử trùng ngay ở mức huấn luyện

---

# Triển khai thực tế
- torchvision.ops.nms(boxes, scores, iou_threshold)
- batched_nms(boxes, scores, class_ids, iou_threshold) — riêng từng lớp
- Lưu ý: box định dạng xyxy; lọc score_threshold và top_k trước
- Chạy NMS riêng theo lớp để không khử nhầm hai vật khác lớp

---

# Kết luận
- NMS = kỹ thuật gom các đối tượng nhận dạng chồng lấp (IoU + điểm)
- Greedy NMS đơn giản nhưng yếu ở cảnh đông và bị khoá ngưỡng cứng
- Biến thể giải quyết từng điểm yếu: Soft / Adaptive / DIoU / Cluster / Matrix / Learning
- Xu hướng: detector NMS-free (DETR, YOLOv10) hướng tới suy luận đầu-cuối
- Chọn phương pháp theo bối cảnh bài toán
