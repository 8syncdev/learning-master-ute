# Tài liệu tham khảo — Non-Maximum Suppression

Mọi mục dưới đây đã được kiểm chứng nguồn (arXiv / kỷ yếu hội nghị) trong quá trình nghiên cứu.

## Bài báo nền tảng và biến thể NMS

1. **Soft-NMS** — N. Bodla, B. Singh, R. Chellappa, L. S. Davis. *Soft-NMS — Improving Object Detection With One Line of Code.* ICCV 2017, tr. 5562–5570.
   - arXiv: https://arxiv.org/abs/1704.04503
   - ICCV Open Access: https://openaccess.thecvf.com/content_iccv_2017/html/Bodla_Soft-NMS_--_Improving_ICCV_2017_paper.html

2. **Learning NMS (GossipNet)** — J. Hosang, R. Benenson, B. Schiele. *Learning Non-Maximum Suppression.* CVPR 2017.
   - arXiv: https://arxiv.org/abs/1705.02950
   - CVPR Open Access: https://openaccess.thecvf.com/content_cvpr_2017/html/Hosang_Learning_Non-Maximum_Suppression_CVPR_2017_paper.html
   - Mã nguồn: https://github.com/hosang/gossipnet

3. **Adaptive-NMS** — S. Liu, D. Huang, Y. Wang. *Adaptive NMS: Refining Pedestrian Detection in a Crowd.* CVPR 2019.
   - arXiv: https://arxiv.org/abs/1904.03629

4. **DIoU / CIoU + DIoU-NMS** — Z. Zheng, P. Wang, W. Liu, J. Li, R. Ye, D. Ren. *Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression.* AAAI 2020.
   - arXiv: https://arxiv.org/abs/1911.08287
   - Mã nguồn: https://github.com/Zzh-tju/DIoU

5. **Cluster-NMS (+ CIoU)** — Z. Zheng, P. Wang, D. Ren, W. Liu, R. Ye, Q. Hu, W. Zuo. *Enhancing Geometric Factors in Model Learning and Inference for Object Detection and Instance Segmentation.* IEEE TCYB 2021.
   - arXiv: https://arxiv.org/abs/2005.03572
   - Mã nguồn: https://github.com/Zzh-tju/CIoU

6. **Matrix-NMS (SOLOv2)** — X. Wang, R. Zhang, T. Kong, L. Li, C. Shen. *SOLOv2: Dynamic and Fast Instance Segmentation.* NeurIPS 2020.
   - arXiv: https://arxiv.org/abs/2003.10152

7. **Fast-NMS (YOLACT)** — D. Bolya, C. Zhou, F. Xiao, Y. J. Lee. *YOLACT: Real-time Instance Segmentation.* ICCV 2019.
   - arXiv: https://arxiv.org/abs/1904.02689

8. **IoU-Net** — B. Jiang, R. Luo, J. Mao, T. Xiao, Y. Jiang. *Acquisition of Localization Confidence for Accurate Object Detection.* ECCV 2018.
   - arXiv: https://arxiv.org/abs/1807.11590

## Độ đo IoU và mở rộng

9. **GIoU** — H. Rezatofighi, N. Tsoi, J. Gwak, A. Sadeghian, I. Reid, S. Savarese. *Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression.* CVPR 2019.
   - arXiv: https://arxiv.org/abs/1902.09630

## Detector liên quan và hướng NMS-free

10. **R-CNN** — R. Girshick, J. Donahue, T. Darrell, J. Malik. *Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation.* CVPR 2014.
    - arXiv: https://arxiv.org/abs/1311.2524

11. **DETR** — N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov, S. Zagoruyko. *End-to-End Object Detection with Transformers.* ECCV 2020.
    - arXiv: https://arxiv.org/abs/2005.12872

12. **YOLOv10** — A. Wang, H. Chen, L. Liu, K. Chen, Z. Lin, J. Han, G. Ding. *YOLOv10: Real-Time End-to-End Object Detection.* NeurIPS 2024.
    - arXiv: https://arxiv.org/abs/2405.14458

## Thuật toán NMS kinh điển (tiền học sâu)

13. **Efficient NMS** — A. Neubeck, L. Van Gool. *Efficient Non-Maximum Suppression.* ICPR 2006.

## Triển khai / thư viện

14. **torchvision.ops** — `nms`, `batched_nms`, `box_iou`.
    - https://pytorch.org/vision/stable/ops.html
