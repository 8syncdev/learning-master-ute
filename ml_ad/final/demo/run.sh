#!/usr/bin/env bash
# Khoi dong ca backend (FastAPI :8000) va frontend (Vite :3000) o nen.
set -euo pipefail
cd "$(dirname "$0")/.."   # ml_ad/final

if [ ! -f demo/api/mlp_feat.pt ]; then
  echo ">> Chua co artifact. Chay: .venv/bin/python save_artifacts.py"
  .venv/bin/python save_artifacts.py
fi
if [ ! -d demo/web/node_modules ]; then
  echo ">> Cai deps frontend..."
  (cd demo/web && bun install)
fi

echo ">> Backend :8000"
(cd demo/api && ../../.venv/bin/python -m uvicorn main:app --port 8000) &
API=$!
echo ">> Frontend :3000"
(cd demo/web && bun run dev) &
WEB=$!
trap 'kill $API $WEB 2>/dev/null || true' EXIT

echo ">> Mo http://localhost:3000  (Ctrl+C de tat ca)"
wait
