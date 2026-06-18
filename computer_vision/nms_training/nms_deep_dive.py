"""Phân tích SÂU NMS bằng ảnh test thật + trace số học (chạy sau train).

Sinh thêm:
  - 06_gallery_before_after.png : 4 ảnh val thật, raw vs sau NMS@0.5
  - 07_nms_step_by_step.png     : NMS chạy TỪNG BƯỚC trên 1 ảnh (giữ/xoá/đang xét)
  - 08_score_threshold_effect.png: ảnh hưởng score_threshold (sau NMS@0.5)
  - outputs/logs/nms_trace.txt   : trace số học (IoU thật từng bước)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from dataset import PennFudanDataset, get_transform
from model import build_model
from nms import box_iou_pairwise, greedy_nms
from train import split_indices

OUT_FIG = "outputs/figures"
DEVICE = torch.device("cuda")


def load_model():
    m = build_model(2)
    ck = torch.load("outputs/checkpoints/best.pth", map_location=DEVICE, weights_only=False)
    m.load_state_dict(ck["model_state"]); m.to(DEVICE).eval()
    m.roi_heads.score_thresh = 0.05
    m.roi_heads.nms_thresh = 1.0
    m.roi_heads.detections_per_img = 300
    return m, ck


@torch.inference_mode()
def raw_dets(model, img, score_thr):
    o = model([img.to(DEVICE)])[0]
    b, s = o["boxes"].cpu(), o["scores"].cpu()
    k = s >= score_thr
    return b[k], s[k]


def draw(ax, img, boxes, color, title, lw=2):
    ax.imshow(img.permute(1, 2, 0).clamp(0, 1).numpy())
    for box in boxes:
        x1, y1, x2, y2 = [float(v) for v in box]
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, lw=lw))
    ax.set_title(title, fontsize=10); ax.axis("off")


def greedy_nms_trace(boxes, scores, iou_thr):
    """Như greedy_nms nhưng ghi lại TỪNG BƯỚC: box được chọn, IoU, box bị xoá."""
    order = torch.argsort(scores, descending=True)
    keep, steps = [], []
    while order.numel() > 0:
        i = int(order[0]); rest = order[1:]
        if rest.numel() > 0:
            ious = box_iou_pairwise(boxes[i], boxes[rest])
            supp = rest[ious >= iou_thr]; surv = rest[ious < iou_thr]
        else:
            ious = torch.empty(0); supp = surv = rest
        steps.append({"pick": i, "score": float(scores[i]),
                      "rest": rest.tolist(), "ious": ious.tolist(),
                      "supp": supp.tolist(), "surv": surv.tolist()})
        keep.append(i); order = surv
    return keep, steps


def fig06_gallery(model, ds, idxs):
    counts = sorted(idxs, key=lambda i: -len(ds[i][1]["boxes"]))[:4]
    fig, axes = plt.subplots(4, 2, figsize=(13, 20))
    for r, idx in enumerate(counts):
        img = ds[idx][0]
        b, s = raw_dets(model, img, 0.3)
        keep = greedy_nms(b, s, 0.5)
        draw(axes[r, 0], img, b, "yellow", f"Ảnh #{idx} — TRƯỚC NMS: {len(b)} box")
        draw(axes[r, 1], img, b[keep], "lime", f"Ảnh #{idx} — SAU NMS@0.5: {len(keep)} box")
    fig.suptitle("Ảnh test THẬT: NMS gom đối tượng chồng lấp (4 ảnh val khác nhau)", fontsize=13)
    fig.tight_layout(); fig.savefig(f"{OUT_FIG}/06_gallery_before_after.png", dpi=110); plt.close(fig)
    return counts


def fig07_step_by_step(model, ds, idx):
    img = ds[idx][0]
    b, s = raw_dets(model, img, 0.6)            # lấy ít box hơn cho dễ nhìn
    keep, steps = greedy_nms_trace(b, s, 0.5)
    n_panel = min(6, len(steps))
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    for t, ax in enumerate(axes.ravel()):
        if t >= n_panel:
            ax.axis("off"); continue
        st = steps[t]
        ax.imshow(img.permute(1, 2, 0).clamp(0, 1).numpy())
        # đã giữ ở các bước trước (xanh lá)
        for j in [steps[u]["pick"] for u in range(t)]:
            x1, y1, x2, y2 = b[j].tolist()
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="lime", lw=2))
        # còn lại chưa xét (xám)
        for j in st["rest"]:
            x1, y1, x2, y2 = b[j].tolist()
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="gray", lw=1, linestyle=":"))
        # bị xoá ở bước này (đỏ nét đứt)
        for j in st["supp"]:
            x1, y1, x2, y2 = b[j].tolist()
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red", lw=2, linestyle="--"))
        # box được chọn (xanh dương, dày)
        x1, y1, x2, y2 = b[st["pick"]].tolist()
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="blue", lw=3.5))
        ax.set_title(f"Bước {t+1}: chọn box điểm {st['score']:.2f} (xanh dương) "
                     f"-> xoá {len(st['supp'])} box IoU≥0.5 (đỏ)", fontsize=10)
        ax.axis("off")
    fig.suptitle("NMS chạy TỪNG BƯỚC: xanh dương=đang chọn, đỏ=bị xoá, xanh lá=đã giữ, xám=chờ xét",
                 fontsize=13)
    fig.tight_layout(); fig.savefig(f"{OUT_FIG}/07_nms_step_by_step.png", dpi=110); plt.close(fig)
    return b, s, keep, steps


def fig08_score_threshold(model, ds, idx):
    img = ds[idx][0]
    b, s = raw_dets(model, img, 0.05)
    keep = greedy_nms(b, s, 0.5)
    bk, sk = b[keep], s[keep]
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    for ax, thr in zip(axes.ravel(), [0.05, 0.3, 0.5, 0.7]):
        m = sk >= thr
        draw(ax, img, bk[m], "magenta", f"score_threshold = {thr}  ->  {int(m.sum())} box")
    fig.suptitle("Ảnh hưởng score_threshold (lọc SAU NMS@0.5): cao = ít box nhưng dễ sót vật mờ",
                 fontsize=12)
    fig.tight_layout(); fig.savefig(f"{OUT_FIG}/08_score_threshold_effect.png", dpi=110); plt.close(fig)


def write_trace(b, s, steps, idx, path="outputs/logs/nms_trace.txt"):
    lines = [f"TRACE số học greedy-NMS@0.5 trên ảnh val #{idx} (raw, score>=0.6, {len(b)} box)",
             "Quy ước: mỗi bước chọn box điểm cao nhất còn lại, xoá mọi box IoU>=0.5 với nó.", ""]
    for t, st in enumerate(steps):
        lines.append(f"Bước {t+1}: GIỮ box điểm={st['score']:.3f}")
        pairs = sorted(zip(st["rest"], st["ious"]), key=lambda x: -x[1])[:6]
        for j, iou in pairs:
            tag = "XOÁ " if iou >= 0.5 else "giữ lại"
            lines.append(f"    - IoU với box (điểm={s[j]:.3f}) = {iou:.3f}  -> {tag}")
        lines.append(f"    => còn lại {len(st['surv'])} box để xét tiếp")
    lines.append("")
    lines.append(f"Kết quả: từ {len(b)} box giữ lại {len(steps)} box.")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines[:14]))
    print("... (đầy đủ ở", path + ")")


def main():
    model, ck = load_model()
    ds = PennFudanDataset("data/PennFudanPed", get_transform(False))
    _, va = split_indices(len(ds), ck["args"]["val_size"], ck["args"]["seed"])
    crowded = max(va, key=lambda i: len(ds[i][1]["boxes"]))
    print("Ảnh đông nhất idx=", crowded, "(", len(ds[crowded][1]["boxes"]), "người )")
    fig06_gallery(model, ds, va)
    b, s, keep, steps = fig07_step_by_step(model, ds, crowded)
    fig08_score_threshold(model, ds, crowded)
    write_trace(b, s, steps, crowded)
    print("Đã sinh: 06_gallery_before_after.png, 07_nms_step_by_step.png, 08_score_threshold_effect.png")


if __name__ == "__main__":
    main()
