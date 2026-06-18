"""Penn-Fudan dataset cho phát hiện người đi bộ.

Box được suy ra từ mask thực thể (mỗi màu = 1 người). Đây là bản chuẩn của
tutorial torchvision, có thêm guard loại box suy biến (w hoặc h = 0).
"""
import os

import torch
from torchvision import tv_tensors
from torchvision.io import ImageReadMode, read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision.transforms import v2 as T


class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.imgs = sorted(os.listdir(os.path.join(root, "PNGImages")))
        self.masks = sorted(os.listdir(os.path.join(root, "PedMasks")))

    def __getitem__(self, idx):
        img = read_image(os.path.join(self.root, "PNGImages", self.imgs[idx]), mode=ImageReadMode.RGB)
        mask = read_image(os.path.join(self.root, "PedMasks", self.masks[idx]))

        obj_ids = torch.unique(mask)[1:]  # bỏ background (id 0)
        masks = (mask == obj_ids[:, None, None]).to(torch.uint8)
        boxes = masks_to_boxes(masks)

        # Guard: loại box suy biến (x2<=x1 hoặc y2<=y1) — tránh crash khi train.
        keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes, masks = boxes[keep], masks[keep]
        n = int(keep.sum())

        labels = torch.ones((n,), dtype=torch.int64)  # 1 = person
        img = tv_tensors.Image(img)
        target = {
            "boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=img.shape[-2:]),
            "masks": tv_tensors.Mask(masks),
            "labels": labels,
            "image_id": idx,
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "iscrowd": torch.zeros((n,), dtype=torch.int64),
        }
        if self.transforms:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.imgs)


def get_transform(train):
    t = []
    if train:
        t.append(T.RandomHorizontalFlip(0.5))
    t += [T.ToDtype(torch.float, scale=True), T.ToPureTensor()]
    return T.Compose(t)


def collate_fn(batch):
    """Thay cho references/detection/utils.collate_fn."""
    return tuple(zip(*batch))
