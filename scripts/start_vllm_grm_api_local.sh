#!/usr/bin/env bash
# 本机 / Git Bash 启动 vLLM GRM API（逻辑对齐 start_vllm_grm_api.sh）。
# 默认模型目录：<仓库>/infer/pretrained；若 config 在 SpeechJudge-GRM 子目录，可设置环境变量覆盖。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "$REPO_ROOT"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export SPEECHJUDGE_MODEL_PATH="${SPEECHJUDGE_MODEL_PATH:-${REPO_ROOT}/infer/pretrained}"
export SPEECHJUDGE_VLLM_API_DATA="${SPEECHJUDGE_VLLM_API_DATA:-${REPO_ROOT}/data/vllm_grm_jobs}"

HOST="${SPEECHJUDGE_API_HOST:-127.0.0.1}"
PORT="${SPEECHJUDGE_API_PORT:-8000}"

LOG_DIR="${REPO_ROOT}/logs/vllm_grm_api"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/server_$(date +%Y%m%d_%H%M%S).log"

# Git Bash 通常没有 pkill：有则用 pkill，否则用 powershell 按命令行匹配结束进程。
stop_cmdline_substr() {
  local substr=$1
  if command -v pkill >/dev/null 2>&1; then
    pkill -f "$substr" || true
    return 0
  fi
  if ! command -v powershell.exe >/dev/null 2>&1; then
    echo "[WARN] pkill missing and powershell.exe not found; skip stop: $substr" | tee -a "$LOG_FILE" || true
    return 0
  fi
  MSYS2_ARG_CONV_EXCL='*' SPEECHJUDGE_KILL_PATTERN="$substr" powershell.exe -NoProfile -ExecutionPolicy Bypass -Command '
    $p = $env:SPEECHJUDGE_KILL_PATTERN
    if (-not $p) { exit 0 }
    Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
      Where-Object { $_.CommandLine -and $_.CommandLine.Contains($p) } |
      ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
  ' 2>/dev/null || true
}

echo "[INFO] repo: $REPO_ROOT" | tee -a "$LOG_FILE"
echo "[INFO] SPEECHJUDGE_MODEL_PATH=$SPEECHJUDGE_MODEL_PATH" | tee -a "$LOG_FILE"
echo "[INFO] SPEECHJUDGE_VLLM_API_DATA=$SPEECHJUDGE_VLLM_API_DATA" | tee -a "$LOG_FILE"
echo "[INFO] bind ${HOST}:${PORT}" | tee -a "$LOG_FILE"

echo "[INFO] stopping SGLang if running (free GPU)..." | tee -a "$LOG_FILE"
stop_cmdline_substr "sglang.launch_server"
stop_cmdline_substr "sglang.router"

echo "[INFO] stopping uvicorn on rank_jobs_app / vllm_grm_api..." | tee -a "$LOG_FILE"
stop_cmdline_substr "uvicorn rank_jobs_app.app.main:app"
stop_cmdline_substr "rank_jobs_app.app.main:app"
stop_cmdline_substr "uvicorn vllm_grm_api.app.main:app"

echo "[INFO] stopping old cloudflared quick tunnel..." | tee -a "$LOG_FILE"
stop_cmdline_substr "cloudflared tunnel"

sleep 2

echo "[INFO] starting SpeechJudge vLLM GRM API..." | tee -a "$LOG_FILE"
nohup python -m uvicorn vllm_grm_api.app.main:app \
  --host "$HOST" \
  --port "$PORT" \
  --log-level info \
  >>"$LOG_FILE" 2>&1 &

SERVER_PID=$!
# Git Bash / MSYS：disown 后 wait 常会立刻返回，终端像已结束但 uvicorn 仍在跑。
# 仅在非 MinGW 类环境 disown（与 Linux 上 start_vllm_grm_api.sh 行为一致）。
case "$(uname -s 2>/dev/null || echo unknown)" in
  MINGW* | MSYS* | CYGWIN*) ;;
  *) disown "$SERVER_PID" 2>/dev/null || true ;;
esac
echo "[INFO] server pid: $SERVER_PID" | tee -a "$LOG_FILE"

echo "[INFO] waiting for /health ..." | tee -a "$LOG_FILE"
for _i in $(seq 1 120); do
  if curl -sf "http://${HOST}:${PORT}/health" >/dev/null; then
    echo "[SUCCESS] API is ready." | tee -a "$LOG_FILE"
    break
  fi
  sleep 2
done

if curl -sf "http://${HOST}:${PORT}/health" >/dev/null; then
  echo "[INFO] uvicorn keeps running (PID $SERVER_PID). Log: $LOG_FILE" | tee -a "$LOG_FILE"
  echo "[INFO] Verify: curl -sf http://${HOST}:${PORT}/health" | tee -a "$LOG_FILE"
else
  echo "[WARN] /health still not OK after wait loop; check $LOG_FILE" | tee -a "$LOG_FILE"
fi

if command -v cloudflared >/dev/null 2>&1; then
  echo "[INFO] starting cloudflared quick tunnel..." | tee -a "$LOG_FILE"
  cloudflared tunnel --url "http://${HOST}:${PORT}" \
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
  echo "[INFO] Optional: install cloudflared and add to PATH if you need a public trycloudflare URL." | tee -a "$LOG_FILE"
fi

echo "[INFO] Waiting for uvicorn (PID $SERVER_PID) to exit; leave this window open while the API runs." | tee -a "$LOG_FILE"
wait "$SERVER_PID" || true
