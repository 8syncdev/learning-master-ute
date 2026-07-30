# FRF-MLP · Demo web

App web minh hoạ **FRF-MLP** (MLP + fuzzy logic Mamdani) phát hiện ngôn từ công kích
trên bình luận mạng xã hội tiếng Việt — 3 lớp **CLEAN / OFFENSIVE / HATE**. Demo trực quan
hoá đầy đủ: **độ tin cậy 3 lớp**, **phân tách 2 kênh** (MLP thống kê + hệ luật mờ), **hàm thành
viên LOW/MED/HIGH** của 3 biến ngôn ngữ, **7 luật Mamdani** (luật nào kích hoạt), và
**token công kích** được tô sáng theo z-score log-odds.

```
┌─ Vite + React + TS (web/, :3000) ────────────┐
│   composer + verdict + giải thích trực quan │
└──────────────────┬──────────────────────────┘
                   │ JSON over /api (vite proxy)
┌──────────────────▼──────────────────────────┐
│ FastAPI + uvicorn (api/, :8000)             │
│   load artifacts.pkl + mlp_feat.pt → predict │
└─────────────────────────────────────────────┘
```

## Chạy demo
```bash
# 0. (một lần) tạo venv + cài deps nếu chưa có
cd ml_ad/final
uv venv .venv
uv pip install -p .venv/bin/python numpy scikit-learn matplotlib pandas scipy torch \
  --index-url https://download.pytorch.org/whl/cpu \
  && uv pip install -p .venv/bin/python fastapi 'uvicorn[standard]' pydantic

# 1. (một lần) sinh artifact: trọng số MLP + vectorizer TF-IDF + lexicon log-odds
#    (≈ 1 phút CPU; reuse train.py + fuzzy.py)
.venv/bin/python save_artifacts.py     # → demo/api/{mlp_feat.pt, artifacts.pkl}

# 2. (một lần) cài deps frontend
cd demo/web && bun install   # hoặc: npm install

# 3. chạy 2 tiến trình (2 terminal)
cd demo/api && ../../.venv/bin/python -m uvicorn main:app --port 8000 --reload   # backend
cd demo/web && bun run dev                                                        # frontend

# mở http://localhost:3000
```

Hoặc dùng script tiện lợi từ `ml_ad/final/`:

```bash
bash demo/run.sh        # khởi động cả 2 nền, in 2 URL
```

## Thử

- Bấm **Mẫu ngẫu nhiên** để lấy bình luận thật từ ViHSD.
- Sửa văn bản, **Ctrl/⌘ + Enter** để phân tích lại.
- Quan sát panel phải: xác suất 3 lớp, fusion `(1−λ)·p_MLP + λ·p_mờ`, 3 biến mờ
  $S$ (độ công kích) / $D$ (mật độ) / $T$ (nhắm đích), 7 luật, và token được tô sáng
  (`z>0` đỏ = công kích, `z<0` xanh = lệch sạch).

## API

| Endpoint | Method | Payload | Return |
|---|---|---|---|
| `/health` | GET | — | trạng thái + `λ`, số mẫu |
| `/sample` | GET | — | một bình luận ngẫu nhiên từ ViHSD |
| `/predict` | POST | `{"text": "..."}` | nhãn + xác suất + giải thích đầy đủ |

## Lưu ý

Demo mang tính **giáo dục và trực quan hoá**, không phải công cụ kiểm duyệt production.
Mô hình huấn luyện trên ViHSD có macro-F1 ≈ 63% (xem báo cáo JTE đi kèm) — sai sót với
teencode mới, ngữ cảnh figurative, và bình luận biên giới CLEAN/OFFENSIVE là điều bình thường.
