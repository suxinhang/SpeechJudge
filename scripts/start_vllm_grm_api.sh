#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Optional: conda (uncomment / adjust if you use conda like scripts/start.sh)
# source /root/miniconda3/etc/profile.d/conda.sh
# conda activate speechjudge

cd "$REPO_ROOT"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
# Model weights (same convention as infer)
export SPEECHJUDGE_MODEL_PATH="${SPEECHJUDGE_MODEL_PATH:-${REPO_ROOT}/infer/pretrained/SpeechJudge-GRM}"
# JSON tasks vs downloaded audio: separate directories under this root
export SPEECHJUDGE_VLLM_API_DATA="${SPEECHJUDGE_VLLM_API_DATA:-${REPO_ROOT}/data/vllm_grm_jobs}"

LOG_DIR="${REPO_ROOT}/logs/vllm_grm_api"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/server_$(date +%Y%m%d_%H%M%S).log"

echo "[INFO] repo: $REPO_ROOT" | tee -a "$LOG_FILE"
echo "[INFO] SPEECHJUDGE_MODEL_PATH=$SPEECHJUDGE_MODEL_PATH" | tee -a "$LOG_FILE"
echo "[INFO] SPEECHJUDGE_VLLM_API_DATA=$SPEECHJUDGE_VLLM_API_DATA" | tee -a "$LOG_FILE"

echo "[INFO] stopping old uvicorn (vllm_grm_api) if any..." | tee -a "$LOG_FILE"
pkill -f "uvicorn vllm_grm_api.app.main:app" || true
sleep 2

echo "[INFO] starting SpeechJudge vLLM GRM API..." | tee -a "$LOG_FILE"
nohup python -m uvicorn vllm_grm_api.app.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --log-level info \
  >>"$LOG_FILE" 2>&1 &

SERVER_PID=$!
disown "$SERVER_PID" 2>/dev/null || true
echo "[INFO] server pid: $SERVER_PID" | tee -a "$LOG_FILE"

echo "[INFO] waiting for /health ..." | tee -a "$LOG_FILE"
for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:8000/health" >/dev/null; then
    echo "[SUCCESS] API is ready (model load may still be finishing on first request)." | tee -a "$LOG_FILE"
    break
  fi
  sleep 2
done

if command -v cloudflared >/dev/null 2>&1; then
  echo "[INFO] starting cloudflared quick tunnel..." | tee -a "$LOG_FILE"
  cloudflared tunnel --url "http://127.0.0.1:8000" \
    >"$LOG_DIR/cloudflared.log" 2>&1 &

  PUBLIC_URL=""
  for _cf in $(seq 1 60); do
    if [[ -f "$LOG_DIR/cloudflared.log" ]]; then
      PUBLIC_URL="$(
        grep -oE 'https://[^[:space:]]+trycloudflare\.com' "$LOG_DIR/cloudflared.log" 2>/dev/null | head -n 1 || true
      )"
    fi
    if [[ -n "$PUBLIC_URL" ]]; then
      break
    fi
    sleep 2
  done

  if [[ -n "$PUBLIC_URL" ]]; then
    echo "[SUCCESS] Public URL: $PUBLIC_URL" | tee -a "$LOG_FILE"
  else
    echo "[WARN] tunnel URL not detected; see $LOG_DIR/cloudflared.log" | tee -a "$LOG_FILE"
  fi
else
  echo "[WARN] cloudflared not in PATH; skipping tunnel." | tee -a "$LOG_FILE"
fi

wait "$SERVER_PID"
