#!/usr/bin/env bash
# === FRF-MLP demo · 1 lệnh chạy tất cả (backend :8000 + frontend :3000) ===
# Boot-strap tự động: venv Python → artifact model → node_modules → 2 server.
# Chạy:  bash demo/run.sh      (từ bất kỳ đâu; Ctrl+C để tắt cả hai)
set -euo pipefail
FINAL="$(cd "$(dirname "$0")/.." && pwd)"   # ml_ad/final
cd "$FINAL"

PYTHONBin () { [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3; }
JS_RUN () {
  if command -v bun >/dev/null 2>&1; then echo bun
  elif command -v npm >/dev/null 2>&1; then echo npm
  else echo ""; fi
}

echo "═══════════════════════════════════════════════════════"
echo "  FRF-MLP demo · boot-strap (lần đầu ~2 phút, sau đó ~2 giây)"
echo "═══════════════════════════════════════════════════════"

# ---- 1. Python venv + deps -------------------------------------------------
if ! [ -x .venv/bin/python ]; then
  echo ">> [1/4] Tạo venv + cài Python deps..."
  if ! command -v uv >/dev/null 2>&1; then
    echo "Lỗi: cần 'uv' (curl -LsSf https://astral.sh/uv/install.sh | sh) hoặc tạo .venv thủ công." >&2
    exit 1
  fi
  uv venv .venv -q
  uv pip install -q -p .venv/bin/python numpy scikit-learn matplotlib pandas scipy
  uv pip install -q -p .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
  uv pip install -q -p .venv/bin/python fastapi 'uvicorn[standard]' pydantic
fi
PY="$(PYTHONBin)"

# ---- 2. Model artifact (trọng số MLP + vectorizer) -------------------------
if [ ! -f demo/api/mlp_feat.pt ] || [ ! -f demo/api/artifacts.pkl ]; then
  echo ">> [2/4] Dựng model artifact (retrain mlp_feat ~1 phút CPU)..."
  "$PY" save_artifacts.py
fi

# ---- 3. Frontend deps ------------------------------------------------------
JR="$(JS_RUN)"
[ -n "$JR" ] || { echo "Lỗi: cần 'bun' hoặc 'node+npm'." >&2; exit 1; }
if [ "$JR" = bun ] && [ ! -d demo/web/node_modules ]; then
  echo ">> [3/4] Cài frontend deps (bun)..."
  ( cd demo/web && bun install )
elif [ "$JR" = npm ] && [ ! -d demo/web/node_modules ]; then
  echo ">> [3/4] Cài frontend deps (npm)..."
  ( cd demo/web && npm install )
fi

# ---- 4. Khởi động 2 server nền --------------------------------------------
echo ">> [4/4] Khởi động backend (FastAPI :8000) + frontend (Vite :3000)..."
( cd demo/api && ../../.venv/bin/python -m uvicorn main:app --port 8000 ) &
API=$!
( cd demo/web && "$JR" run dev ) &
WEB=$!
trap 'echo; echo ">> Tắt cả hai server..."; kill $API $WEB 2>/dev/null || true' EXIT

sleep 2
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  ✅ Mở:   http://localhost:3000        (Ctrl+C để tắt)"
echo "  📡 API:  http://localhost:8000/docs    (Swagger UI)"
echo "═══════════════════════════════════════════════════════"
wait
