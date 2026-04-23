#!/usr/bin/env bash
# 远端启动 vLLM GRM Job API：先尽量释放 GPU / 端口，再起 uvicorn，再 cloudflared quick tunnel。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Optional: conda (uncomment / adjust if different on your machine/container)
# source /root/miniconda3/etc/profile.d/conda.sh
# conda activate speechjudge

cd "$REPO_ROOT"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export SPEECHJUDGE_MODEL_PATH="${SPEECHJUDGE_MODEL_PATH:-${REPO_ROOT}/infer/pretrained/SpeechJudge-GRM}"
export SPEECHJUDGE_VLLM_API_DATA="${SPEECHJUDGE_VLLM_API_DATA:-${REPO_ROOT}/data/vllm_grm_jobs}"

# --------- HTTP 端口（uvicorn / curl / cloudflared 一致；改端口只改本行）---------
API_PORT=8000

LOG_DIR="${REPO_ROOT}/logs/vllm_grm_api"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/server_$(date +%Y%m%d_%H%M%S).log"

echo "[INFO] repo: $REPO_ROOT" | tee -a "$LOG_FILE"
echo "[INFO] SPEECHJUDGE_MODEL_PATH=$SPEECHJUDGE_MODEL_PATH" | tee -a "$LOG_FILE"
echo "[INFO] SPEECHJUDGE_VLLM_API_DATA=$SPEECHJUDGE_VLLM_API_DATA" | tee -a "$LOG_FILE"
echo "[INFO] API_PORT=$API_PORT" | tee -a "$LOG_FILE"

# ======================
# 1) 释放 GPU：停常见推理栈 + 本仓库相关 uvicorn，并打 nvidia-smi 便于远端排查
# ======================
echo "[INFO] nvidia-smi (compute apps) before cleanup:" | tee -a "$LOG_FILE"
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv 2>/dev/null | tee -a "$LOG_FILE" || true

echo "[INFO] stopping SGLang (free GPU)..." | tee -a "$LOG_FILE"
pkill -f "sglang.launch_server" || true
pkill -f "sglang\.router" || true

echo "[INFO] stopping standalone vLLM / TGI-style entrypoints (if any)..." | tee -a "$LOG_FILE"
pkill -f "vllm.entrypoints" || true
pkill -f "vllm serve" || true
pkill -f "tgi.entrypoint" || true
pkill -f "text-generation-launcher" || true

echo "[INFO] stopping uvicorn rank_jobs_app / vllm_grm_api..." | tee -a "$LOG_FILE"
pkill -f "uvicorn rank_jobs_app.app.main:app" || true
pkill -f "rank_jobs_app.app.main:app" || true
pkill -f "uvicorn vllm_grm_api.app.main:app" || true

echo "[INFO] stopping old cloudflared quick tunnel..." | tee -a "$LOG_FILE"
pkill -f "cloudflared tunnel" || true

if command -v fuser >/dev/null 2>&1; then
  echo "[INFO] freeing TCP ${API_PORT} if still held (fuser -k)..." | tee -a "$LOG_FILE"
  fuser -k "${API_PORT}/tcp" 2>/dev/null || true
fi

echo "[INFO] waiting for GPU / port to settle..." | tee -a "$LOG_FILE"
sleep 4

echo "[INFO] nvidia-smi (compute apps) after cleanup:" | tee -a "$LOG_FILE"
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv 2>/dev/null | tee -a "$LOG_FILE" || true

# ======================
# 2) 启动 API
# ======================
echo "[INFO] starting SpeechJudge vLLM GRM API..." | tee -a "$LOG_FILE"
nohup python -m uvicorn vllm_grm_api.app.main:app \
  --host 0.0.0.0 \
  --port "$API_PORT" \
  --log-level info \
  >>"$LOG_FILE" 2>&1 &

SERVER_PID=$!
disown "$SERVER_PID" 2>/dev/null || true
echo "[INFO] server pid: $SERVER_PID" | tee -a "$LOG_FILE"

echo "[INFO] waiting for /health on port ${API_PORT} ..." | tee -a "$LOG_FILE"
for _i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:${API_PORT}/health" >/dev/null; then
    echo "[SUCCESS] API is ready (model load may still be finishing on first request)." | tee -a "$LOG_FILE"
    break
  fi
  sleep 2
done

# ======================
# 3) Cloudflare quick tunnel：随机 trycloudflare / cfargotunnel 域名，写入主日志便于远端测试
# ======================
_extract_tunnel_url() {
  local f="$1"
  [[ -f "$f" ]] || return 0
  # trycloudflare.com（经典 quick tunnel）或 cfargotunnel.com（部分版本）
  grep -oE 'https://[^[:space:]]+(trycloudflare\.com|cfargotunnel\.com)' "$f" 2>/dev/null | head -n 1 || true
}

if command -v cloudflared >/dev/null 2>&1; then
  echo "[INFO] starting cloudflared quick tunnel -> http://127.0.0.1:${API_PORT} ..." | tee -a "$LOG_FILE"
  : >"$LOG_DIR/cloudflared.log"
  cloudflared tunnel --url "http://127.0.0.1:${API_PORT}" \
    >>"$LOG_DIR/cloudflared.log" 2>&1 &

  PUBLIC_URL=""
  for _cf in $(seq 1 90); do
    PUBLIC_URL="$(_extract_tunnel_url "$LOG_DIR/cloudflared.log")"
    if [[ -n "$PUBLIC_URL" ]]; then
      break
    fi
    sleep 2
  done

  {
    echo ""
    echo "======== CLOUDFLARE QUICK TUNNEL (random public URL) ========"
    if [[ -n "$PUBLIC_URL" ]]; then
      echo "[SUCCESS] Public URL: $PUBLIC_URL"
      echo "(same line for copy) $PUBLIC_URL"
    else
      echo "[WARN] Tunnel URL not parsed within ~180s."
      echo "[WARN] Inspect: $LOG_DIR/cloudflared.log"
      echo "---- tail cloudflared.log ----"
      tail -n 40 "$LOG_DIR/cloudflared.log" 2>/dev/null || true
    fi
    echo "=============================================================="
    echo ""
  } | tee -a "$LOG_FILE"
else
  echo "[WARN] cloudflared not in PATH; skipping tunnel." | tee -a "$LOG_FILE"
fi

wait "$SERVER_PID"
