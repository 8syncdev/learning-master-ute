"""Train (finetune) Faster R-CNN trên Penn-Fudan — THẬT, có checkpoint/resume + log.

Chạy:
    uv run python train.py --epochs 8
Resume:
    uv run python train.py --resume outputs/checkpoints/last.pth --epochs 12
"""
import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import torch

from dataset import PennFudanDataset, collate_fn, get_transform
from metrics import average_precision, map_50_95
from model import build_model


def split_indices(n, val_size, seed):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    return perm[:-val_size], perm[-val_size:]


def train_one_epoch(model, optimizer, loader, device, scaler, epoch, warmup):
    model.train()
    sched = None
    if warmup:
        wi = min(1000, len(loader) - 1)
        sched = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-3, total_iters=wi)
    running, nb = 0.0, 0
    for images, targets in loader:
        images = [im.to(device) for im in images]
        targets = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in t.items()} for t in targets]
        with torch.amp.autocast("cuda"):
            loss_dict = model(images, targets)
            losses = sum(loss_dict.values())
        lv = losses.item()
        if not math.isfinite(lv):
            print("Loss không hữu hạn:", lv, loss_dict)
            sys.exit(1)
        optimizer.zero_grad()
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()
        if sched is not None:
            sched.step()
        running += lv
        nb += 1
    return running / max(nb, 1)


@torch.inference_mode()
def evaluate_split(model, loader, device, score_thr=0.05):
    model.eval()
    preds, gts = [], []
    for images, targets in loader:
        out = model([images[0].to(device)])[0]
        keep = out["scores"].cpu().numpy() >= score_thr
        preds.append((out["boxes"].cpu().numpy()[keep], out["scores"].cpu().numpy()[keep]))
        gts.append(targets[0]["boxes"].numpy())
    ap50 = average_precision(preds, gts, 0.5)[0]
    return ap50, map_50_95(preds, gts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--val-size", type=int, default=50)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--data", default="data/PennFudanPed")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--resume", default="")
    args = ap.parse_args()

    device = torch.device("cuda")
    out = Path(args.out)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    log_path = out / "logs" / "training_log.jsonl"

    ds_train = PennFudanDataset(args.data, get_transform(train=True))
    ds_val = PennFudanDataset(args.data, get_transform(train=False))
    tr_idx, va_idx = split_indices(len(ds_train), args.val_size, args.seed)
    dl_train = torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds_train, tr_idx),
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate_fn)
    dl_val = torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds_val, va_idx),
        batch_size=1, shuffle=False, num_workers=args.workers, collate_fn=collate_fn)
    print(f"train={len(tr_idx)} val={len(va_idx)} device={device}")

    model = build_model(num_classes=2).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, args.epochs // 3), gamma=0.1)
    scaler = torch.amp.GradScaler("cuda")

    start_epoch, history, best_map = 0, [], -1.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        history = ckpt["history"]
        start_epoch = ckpt["epoch"] + 1
        best_map = max((h["map5095"] for h in history), default=-1.0)
        print(f"RESUME từ {args.resume}: tiếp tục từ epoch {start_epoch} (đã có {len(history)} epoch)")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, optimizer, dl_train, device, scaler, epoch, warmup=(epoch == 0))
        scheduler.step()
        ap50, mAP = evaluate_split(model, dl_val, device)
        sec = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        rec = {"epoch": epoch, "train_loss": round(train_loss, 4), "ap50": round(float(ap50), 4),
               "map5095": round(float(mAP), 4), "sec": round(sec, 1), "lr": lr_now}
        history.append(rec)
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[epoch {epoch}] loss={rec['train_loss']} AP@0.5={rec['ap50']} mAP={rec['map5095']} "
              f"({rec['sec']}s, lr={lr_now:.5f})")

        ckpt = {"epoch": epoch, "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
                "history": history, "args": vars(args)}
        torch.save(ckpt, out / "checkpoints" / "last.pth")
        if mAP > best_map:
            best_map = mAP
            shutil.copyfile(out / "checkpoints" / "last.pth", out / "checkpoints" / "best.pth")
            print(f"  -> best.pth cập nhật (mAP@[.5:.95]={best_map:.4f}, AP@0.5={ap50:.4f})")

    with open(out / "logs" / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("\nXong. Lệnh resume để train thêm:")
    print(f"  uv run python train.py --resume {out}/checkpoints/last.pth --epochs {args.epochs + 4}")


if __name__ == "__main__":
    main()
