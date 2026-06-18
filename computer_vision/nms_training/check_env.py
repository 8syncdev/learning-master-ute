"""GATE: xác nhận torch chạy thật trên GPU (sm_120) trước khi train."""
import torch
import torchvision

print("torch", torch.__version__, "| torchvision", torchvision.__version__)
print("cuda available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA không khả dụng — xem contingency CPU/mobilenet trong plan"
print("device:", torch.cuda.get_device_name(0), "| cap:", torch.cuda.get_device_capability(0))
x = torch.randn(2048, 2048, device="cuda")
y = (x @ x).sum().item()  # buộc kernel chạy thật trên sm_120
print("cuda matmul OK:", y == y)
