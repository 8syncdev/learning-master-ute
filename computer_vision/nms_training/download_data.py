"""Tải & giải nén Penn-Fudan Pedestrian (stdlib, idempotent)."""
import hashlib
import urllib.request
import zipfile
from pathlib import Path

URL = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"
dst = Path("data")
dst.mkdir(exist_ok=True)
zp = dst / "PennFudanPed.zip"

if not zp.exists():
    print("Đang tải:", URL)
    urllib.request.urlretrieve(URL, zp)
print("zip bytes:", zp.stat().st_size, "| sha256:", hashlib.sha256(zp.read_bytes()).hexdigest()[:16])

if not (dst / "PennFudanPed").exists():
    with zipfile.ZipFile(zp) as z:
        z.extractall(dst)
    print("Đã giải nén vào", dst / "PennFudanPed")

imgs = sorted((dst / "PennFudanPed" / "PNGImages").glob("*.png"))
masks = sorted((dst / "PennFudanPed" / "PedMasks").glob("*_mask.png"))
print("images:", len(imgs), "masks:", len(masks))
