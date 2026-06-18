"""Độ đo phát hiện tự viết (minh bạch, đơn lớp 'person') — KHÔNG dùng pycocotools.

- iou_matrix: IoU giữa hai tập box xyxy.
- average_precision: AP all-point ở một ngưỡng IoU, theo kiểu PASCAL VOC 2010+/COCO
  (envelope precision rồi tích phân theo recall). Trả (AP, precision_cuối, recall_cuối).
- map_50_95: trung bình AP qua IoU 0.50:0.95:0.05 (kiểu COCO).
"""
import numpy as np


def iou_matrix(a, b):
    """a:[N,4], b:[M,4] xyxy -> IoU [N,M]."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def _all_point_ap(rec, prec):
    """Diện tích dưới đường PR với envelope precision (đơn điệu giảm)."""
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([1.0], prec, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def average_precision(all_preds, all_gts, iou_thr):
    """all_preds: list[(boxes[N,4], scores[N])]; all_gts: list[boxes[M,4]] (numpy)."""
    entries = []  # (score, is_tp)
    n_gt = 0
    for (pb, ps), gb in zip(all_preds, all_gts):
        n_gt += len(gb)
        pb = np.asarray(pb, dtype=np.float64)
        ps = np.asarray(ps, dtype=np.float64)
        gb = np.asarray(gb, dtype=np.float64)
        matched = set()
        for i in np.argsort(-ps):
            if len(gb) == 0:
                entries.append((ps[i], 0))
                continue
            ious = iou_matrix(pb[i:i + 1], gb)[0].copy()
            for j in matched:
                ious[j] = -1.0
            j = int(np.argmax(ious))
            if ious[j] >= iou_thr:
                entries.append((ps[i], 1))
                matched.add(j)
            else:
                entries.append((ps[i], 0))
    if not entries:
        return 0.0, 0.0, 0.0
    entries.sort(key=lambda e: -e[0])
    tp = fp = 0
    rec, prec = [], []
    for _, is_tp in entries:
        tp += is_tp
        fp += 1 - is_tp
        rec.append(tp / max(n_gt, 1))
        prec.append(tp / (tp + fp))
    ap = _all_point_ap(np.array(rec), np.array(prec))
    return ap, prec[-1], rec[-1]


def map_50_95(all_preds, all_gts):
    return float(np.mean([average_precision(all_preds, all_gts, t)[0]
                          for t in np.arange(0.5, 1.0, 0.05)]))
