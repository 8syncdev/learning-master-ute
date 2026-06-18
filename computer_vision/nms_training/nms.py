"""Standard (greedy) Non-Maximum Suppression — cài đặt từ đầu để học nguyên lý.

greedy_nms: thuật toán tham lam thuần (sort theo điểm, giữ box đỉnh, loại box IoU >= thr).
So khớp với torchvision.ops.nms để chứng minh cài đặt đúng (xem study_nms.py).
"""
import torch


def box_iou_pairwise(box, boxes):
    """IoU giữa 1 box [4] và tập boxes [M,4] (xyxy) -> [M]."""
    x1 = torch.maximum(box[0], boxes[:, 0])
    y1 = torch.maximum(box[1], boxes[:, 1])
    x2 = torch.minimum(box[2], boxes[:, 2])
    y2 = torch.minimum(box[3], boxes[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area + areas - inter
    return torch.where(union > 0, inter / union, torch.zeros_like(union))


def greedy_nms(boxes, scores, iou_thr):
    """boxes [N,4] xyxy, scores [N]. Trả index giữ lại, giảm dần theo điểm.

    Quy tắc vàng minh hoạ ngay trong code:
      1) Box phải là xyxy [N,4] (assert) — sai format là lỗi phổ biến nhất.
      2) Guard tensor rỗng — ops.nms/loop trên rỗng dễ vỡ shape.
      3) Luôn xét theo thứ tự điểm GIẢM DẦN.
    """
    assert boxes.ndim == 2 and boxes.shape[1] == 4, "boxes phải là [N,4] định dạng xyxy"
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    order = torch.argsort(scores, descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(int(i))
        if order.numel() == 1:
            break
        rest = order[1:]
        ious = box_iou_pairwise(boxes[i], boxes[rest])
        order = rest[ious < iou_thr]  # giữ lại box chồng lấp ÍT (chưa bị "nuốt")
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)
