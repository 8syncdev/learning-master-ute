"""Nghiên cứu Standard NMS trên detector ĐÃ TRAIN: sinh figures + bảng sweep (số THẬT).

Chạy sau train: uv run python study_nms.py
Đầu ra: outputs/figures/01..05_*.png và outputs/logs/nms_sweep.csv
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.ops import batched_nms
from torchvision.ops import nms as tv_nms

from dataset import PennFudanDataset, get_transform
from metrics import average_precision
from model import build_model
from nms import greedy_nms
from train import split_indices

OUT = Path("outputs")
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda")
SWEEP_THRS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.9]


def load_model():
    m = build_model(num_classes=2)
    ckpt = torch.load(OUT / "checkpoints" / "best.pth", map_location=DEVICE, weights_only=False)
    m.load_state_dict(ckpt["model_state"])
    m.to(DEVICE).eval()
    # Mở "raw": tắt NMS cuối của model để lộ các box chồng lấp (RPN-NMS vẫn bật).
    m.roi_heads.score_thresh = 0.05
    m.roi_heads.nms_thresh = 1.0
    m.roi_heads.detections_per_img = 300
    return m, ckpt


@torch.inference_mode()
def raw_dets(model, img, score_thr):
    out = model([img.to(DEVICE)])[0]
    b, s = out["boxes"].cpu(), out["scores"].cpu()
    keep = s >= score_thr
    return b[keep], s[keep]


def draw(ax, img_chw, boxes, color, title):
    ax.imshow(img_chw.permute(1, 2, 0).clamp(0, 1).numpy())
    for box in boxes:
        x1, y1, x2, y2 = [float(v) for v in box]
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, lw=2))
    ax.set_title(title, fontsize=11)
    ax.axis("off")


def fig01_loss_curve():
    hist = json.loads((OUT / "logs" / "history.json").read_text())
    ep = [h["epoch"] for h in hist]
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(ep, [h["train_loss"] for h in hist], "o-", color="tab:red", label="train loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("train loss", color="tab:red")
    ax2 = ax1.twinx()
    ax2.plot(ep, [h["ap50"] for h in hist], "s-", color="tab:blue", label="AP@0.5")
    ax2.plot(ep, [h["map5095"] for h in hist], "^--", color="tab:green", label="mAP@[.5:.95]")
    ax2.set_ylabel("AP (val)", color="tab:blue"); ax2.set_ylim(0, 1)
    ax1.set_title("Train loss & AP trên val theo epoch")
    fig.legend(loc="upper right", bbox_to_anchor=(0.88, 0.88))
    fig.tight_layout(); fig.savefig(FIG / "01_loss_curve.png", dpi=120); plt.close(fig)


def pick_crowded(ds, idxs):
    best, best_n = idxs[0], -1
    for i in idxs:
        n = len(ds[i][1]["boxes"])
        if n > best_n:
            best, best_n = i, n
    return best, best_n


def fig02_before_after(model, ds, idx):
    img = ds[idx][0]
    boxes, scores = raw_dets(model, img, score_thr=0.3)
    keep = greedy_nms(boxes, scores, 0.5)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    draw(axes[0], img, boxes, "yellow", f"TRƯỚC NMS (raw): {len(boxes)} box chồng lấp")
    draw(axes[1], img, boxes[keep], "lime", f"SAU greedy-NMS@0.5: {len(keep)} box")
    fig.suptitle("NMS gom các đối tượng nhận dạng chồng lấp", fontsize=13)
    fig.tight_layout(); fig.savefig(FIG / "02_before_after_nms.png", dpi=120); plt.close(fig)
    return len(boxes), int(len(keep))


def fig03_sweep_visual(model, ds, idx):
    img = ds[idx][0]
    boxes, scores = raw_dets(model, img, score_thr=0.3)
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    for ax, thr in zip(axes.ravel(), [0.3, 0.5, 0.7, 0.9]):
        keep = greedy_nms(boxes, scores, thr)
        draw(ax, img, boxes[keep], "cyan", f"IoU thr = {thr}  ->  {len(keep)} box giữ lại")
    fig.suptitle("Ảnh hưởng ngưỡng IoU của NMS (thấp = gộp nhầm người sát nhau; cao = còn box trùng)",
                 fontsize=12)
    fig.tight_layout(); fig.savefig(FIG / "03_iou_threshold_sweep.png", dpi=120); plt.close(fig)


def fig04_classagnostic():
    # Dữ liệu tổng hợp có kiểm soát: 2 box đè nhau NHƯNG khác lớp.
    boxes = torch.tensor([[50, 50, 250, 350], [70, 60, 270, 360]], dtype=torch.float)
    scores = torch.tensor([0.95, 0.90])
    labels = torch.tensor([0, 1])  # 0=person, 1=dog (khác lớp)
    keep_agn = tv_nms(boxes, scores, 0.5)                     # class-agnostic: xoá nhầm
    keep_pc = batched_nms(boxes, scores, labels, 0.5)         # per-class: giữ cả hai
    names = {0: "person", 1: "dog"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, keep, title in [(axes[0], keep_agn, "SAI: NMS class-agnostic"),
                            (axes[1], keep_pc, "ĐÚNG: per-class (batched_nms)")]:
        ax.set_xlim(0, 340); ax.set_ylim(420, 0)
        for k in range(len(boxes)):
            x1, y1, x2, y2 = boxes[k].tolist()
            kept = k in set(keep.tolist())
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                       edgecolor=("lime" if kept else "red"),
                                       lw=2.5, linestyle=("-" if kept else "--")))
            ax.text(x1 + 4, y1 + 18, f"{names[labels[k].item()]} {scores[k]:.2f} "
                    f"{'GIỮ' if kept else 'BỊ XOÁ'}", color=("green" if kept else "red"), fontsize=10)
        ax.set_title(f"{title}: giữ {len(keep)}/2"); ax.set_aspect("equal")
    fig.suptitle("Lỗi: NMS không phân lớp xoá nhầm vật KHÁC lớp bị chồng lấp", fontsize=12)
    fig.tight_layout(); fig.savefig(FIG / "04_pitfall_classagnostic.png", dpi=120); plt.close(fig)


def fig05_format():
    # Box ĐÚNG ở dạng xyxy; nếu lỡ đưa dữ liệu xywh vào ops.nms (mong đợi xyxy) -> sai.
    xyxy = torch.tensor([[40, 40, 200, 300], [55, 50, 210, 310], [250, 60, 360, 320]], dtype=torch.float)
    scores = torch.tensor([0.9, 0.8, 0.85])
    keep_ok = greedy_nms(xyxy, scores, 0.5)
    # "Bug": cùng các con số nhưng bị HIỂU NHẦM là xywh -> chuyển sang xyxy sai lệch rồi NMS.
    as_xywh = xyxy.clone()
    wrong_xyxy = torch.stack([as_xywh[:, 0], as_xywh[:, 1],
                              as_xywh[:, 0] + as_xywh[:, 2], as_xywh[:, 1] + as_xywh[:, 3]], dim=1)
    keep_bug = greedy_nms(wrong_xyxy, scores, 0.5)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, bxs, keep, title in [(axes[0], xyxy, keep_ok, "ĐÚNG: box là xyxy"),
                                 (axes[1], wrong_xyxy, keep_bug, "SAI: xem xywh như xyxy")]:
        ax.set_xlim(0, 600); ax.set_ylim(640, 0)
        for k in range(len(bxs)):
            x1, y1, x2, y2 = bxs[k].tolist()
            kept = k in set(keep.tolist())
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                       edgecolor=("lime" if kept else "red"),
                                       lw=2.5, linestyle=("-" if kept else "--")))
        ax.set_title(f"{title}: giữ {len(keep)}"); ax.set_aspect("equal")
    fig.suptitle("Lỗi định dạng box (xywh vs xyxy) làm IoU và NMS sai hoàn toàn", fontsize=12)
    fig.tight_layout(); fig.savefig(FIG / "05_pitfall_format_xywh.png", dpi=120); plt.close(fig)


def nms_sweep_table(model, ds, idxs):
    # Lấy raw detections (score>=0.05, NMS-off) cho toàn val 1 lần.
    raw = []
    gts = []
    for i in idxs:
        img, tgt = ds[i]
        b, s = raw_dets(model, img, score_thr=0.05)
        raw.append((b, s))
        gts.append(tgt["boxes"].numpy())
    rows = []
    for thr in SWEEP_THRS:
        preds, n_after = [], []
        for (b, s) in raw:
            keep = greedy_nms(b, s, thr)
            preds.append((b[keep].numpy(), s[keep].numpy()))
            n_after.append(len(keep))
        ap50, _, recall = average_precision(preds, gts, 0.5)
        rows.append({"iou_thr": thr, "ap50": round(float(ap50), 4),
                     "recall": round(float(recall), 4), "avg_boxes_after": round(float(np.mean(n_after)), 2)})
    with open(OUT / "logs" / "nms_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["iou_thr", "ap50", "recall", "avg_boxes_after"])
        w.writeheader(); w.writerows(rows)
    print("=== NMS IoU-threshold sweep (val) ===")
    for r in rows:
        print(f"  thr={r['iou_thr']}  AP@0.5={r['ap50']}  recall={r['recall']}  avg_boxes={r['avg_boxes_after']}")
    return rows


def nms_self_check(model, ds, idx):
    b, s = raw_dets(model, ds[idx][0], score_thr=0.05)
    for thr in (0.3, 0.5, 0.7):
        g = set(greedy_nms(b, s, thr).tolist())
        t = set(tv_nms(b, s, thr).tolist())
        assert g == t, (thr, len(g), len(t))
    print("NMS self-check OK (greedy_nms == torchvision.ops.nms)")


def main():
    model, ckpt = load_model()
    ds = PennFudanDataset("data/PennFudanPed", get_transform(train=False))
    _, va_idx = split_indices(len(ds), ckpt["args"]["val_size"], ckpt["args"]["seed"])
    crowded, n = pick_crowded(ds, va_idx)
    print(f"Ảnh val đông nhất: idx={crowded} ({n} người)")

    nms_self_check(model, ds, crowded)
    fig01_loss_curve()
    nb, na = fig02_before_after(model, ds, crowded)
    print(f"fig02: raw={nb} box -> sau NMS@0.5={na} box")
    fig03_sweep_visual(model, ds, crowded)
    fig04_classagnostic()
    fig05_format()
    nms_sweep_table(model, ds, va_idx)
    print("Đã sinh figures:", sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()
