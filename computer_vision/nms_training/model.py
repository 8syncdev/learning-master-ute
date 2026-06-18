"""Faster R-CNN ResNet50-FPN — FINETUNE từ trọng số COCO-pretrained.

Điểm khởi đầu (KHÔNG train từ 0):
  - weights="DEFAULT" = FasterRCNN_ResNet50_FPN_Weights.DEFAULT (COCO_V1),
    huấn luyện sẵn trên COCO train2017 (80 lớp). torchvision tự tải tệp
    `fasterrcnn_resnet50_fpn_coco-258fb6c6.pth` từ download.pytorch.org/models.
  - Giữ nguyên: backbone ResNet-50 + FPN + RPN + lớp đặc trưng RoI (trọng số COCO).
  - Thay mới: roi_heads.box_predictor -> FastRCNNPredictor(in_features, num_classes=2)
    (nền + person), khởi tạo ngẫu nhiên rồi train tiếp trên Penn-Fudan.
"""
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_model(num_classes=2):
    """num_classes = 2: background + person."""
    m = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")  # COCO_V1
    in_features = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)  # thay đầu -> 2 lớp
    return m
