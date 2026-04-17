#!/usr/bin/env bash
set -euo pipefail

# ======================
# 1. conda 环境
# ======================
# NOTE: adjust conda install path if different on your machine/container.
source /root/miniconda3/etc/profile.d/conda.sh
conda activate speechjudge

cd /root/SpeechJudge/infer

# ======================
# 2. 环境变量
# ======================
export SPEECHJUDGE_MODEL_PATH="/root/models/SpeechJudge-GRM"
export SPEECHJUDGE_CUDA_DEVICE="0"
# Default when POST /jobs/rank omits form field pairwise_parallel. Prefer setting per request in API.
# export SPEECHJUDGE_PAIRWISE_PARALLEL=5

# ======================
# 3. 日志目录
# ======================
LOG_DIR="/root/SpeechJudge/logs"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/server_$(date +%Y%m%d_%H%M%S).log"

echo "[INFO] starting SpeechJudge..." | tee -a "$LOG_FILE"

# rank_jobs_app 使用本地 JSON 文件存任务（见 SPEECHJUDGE_RANK_JOBS_JSON_DIR），无需 Mongo。

# ======================
# 4. 先停止旧服务（关键）
# ======================
echo "[INFO] stopping SGLang server if running (free GPU before SpeechJudge)..." | tee -a "$LOG_FILE"
# 常见启动：python -m sglang.launch_server ...
pkill -f "sglang.launch_server" || true
pkill -f "sglang\.router" || true

echo "[INFO] stopping old uvicorn process..." | tee -a "$LOG_FILE"

# 杀 uvicorn / rank_jobs_app
pkill -f "uvicorn rank_jobs_app.app.main:app" || true
pkill -f "rank_jobs_app.app.main:app" || true

# 可选：清理 cloudflared
pkill -f "cloudflared tunnel" || true

sleep 2

# ======================
# 5. 启动服务（后台）
# ======================
nohup python -m uvicorn rank_jobs_app.app.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --log-level info \
  >> "$LOG_FILE" 2>&1 &

SERVER_PID=$!
disown "$SERVER_PID" 2>/dev/null || true
echo "[INFO] server pid: $SERVER_PID" | tee -a "$LOG_FILE"

# ======================
# 6. 等待服务 ready
# ======================
echo "[INFO] waiting model loading..." | tee -a "$LOG_FILE"

for i in {1..60}; do
    if curl -s http://127.0.0.1:8000/health >/dev/null; then
        echo "[SUCCESS] SpeechJudge is ready!" | tee -a "$LOG_FILE"
        break
    fi
    sleep 2
done

# ======================
# 7. 启动 Cloudflare Tunnel
# ======================
echo "[INFO] starting cloudflared..." | tee -a "$LOG_FILE"

# 优先使用可写目录中的 cloudflared（例如 /root/bin/cloudflared），
# 回退到 PATH。也可通过环境变量 CLOUDFLARED_BIN 显式指定。
if [[ -z "${CLOUDFLARED_BIN:-}" ]]; then
  if [[ -x "/root/bin/cloudflared" ]]; then
    CLOUDFLARED_BIN="/root/bin/cloudflared"
  else
    CLOUDFLARED_BIN="$(command -v cloudflared || true)"
  fi
fi

# 未找到或未可执行：Linux 下尝试从 GitHub 下载官方二进制到 /root/bin（需 curl 或 wget）。
# 离线或不想自动下载时: export SKIP_CLOUDFLARED_DOWNLOAD=1
if [[ -z "$CLOUDFLARED_BIN" ]] || [[ ! -x "$CLOUDFLARED_BIN" ]]; then
  if [[ -n "$CLOUDFLARED_BIN" ]] && [[ ! -x "$CLOUDFLARED_BIN" ]]; then
    echo "[WARN] cloudflared exists but is not executable: $CLOUDFLARED_BIN" | tee -a "$LOG_FILE"
    echo "[WARN] try: chmod +x \"$CLOUDFLARED_BIN\"" | tee -a "$LOG_FILE"
  fi
  if [[ "${SKIP_CLOUDFLARED_DOWNLOAD:-}" != "1" ]] && [[ "$(uname -s)" == "Linux" ]]; then
    _cf_arch=""
    case "$(uname -m)" in
      x86_64) _cf_arch=amd64 ;;
      aarch64|arm64) _cf_arch=arm64 ;;
    esac
    if [[ -n "$_cf_arch" ]] && { command -v curl >/dev/null 2>&1 || command -v wget >/dev/null 2>&1; }; then
      echo "[INFO] cloudflared missing; downloading cloudflared-linux-${_cf_arch} -> /root/bin/cloudflared ..." | tee -a "$LOG_FILE"
      mkdir -p /root/bin
      _cf_tmp="/root/bin/cloudflared.tmp.$$"
      _cf_url="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-${_cf_arch}"
      set +e
      if command -v curl >/dev/null 2>&1; then
        curl -fsSL "$_cf_url" -o "$_cf_tmp"
      else
        wget -q "$_cf_url" -O "$_cf_tmp"
      fi
      _cf_dl=$?
      set -e
      if [[ "$_cf_dl" -eq 0 ]] && [[ -s "$_cf_tmp" ]]; then
        chmod +x "$_cf_tmp"
        mv -f "$_cf_tmp" /root/bin/cloudflared
        CLOUDFLARED_BIN="/root/bin/cloudflared"
        echo "[SUCCESS] installed cloudflared at $CLOUDFLARED_BIN" | tee -a "$LOG_FILE"
      else
        rm -f "$_cf_tmp"
        echo "[WARN] cloudflared auto-download failed (network or arch?)." | tee -a "$LOG_FILE"
      fi
    elif [[ -z "$_cf_arch" ]]; then
      echo "[WARN] unsupported machine for auto-download: $(uname -m); install cloudflared manually." | tee -a "$LOG_FILE"
    else
      echo "[WARN] need curl or wget to auto-download cloudflared." | tee -a "$LOG_FILE"
    fi
  elif [[ "${SKIP_CLOUDFLARED_DOWNLOAD:-}" == "1" ]]; then
    echo "[INFO] SKIP_CLOUDFLARED_DOWNLOAD=1; not attempting cloudflared download." | tee -a "$LOG_FILE"
  fi
fi

START_TUNNEL=1
if [[ -z "$CLOUDFLARED_BIN" ]]; then
  echo "[WARN] cloudflared not available; skip public tunnel." | tee -a "$LOG_FILE"
  echo "[HINT] Linux x86_64 manual install:" | tee -a "$LOG_FILE"
  echo "  mkdir -p /root/bin && curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /root/bin/cloudflared && chmod +x /root/bin/cloudflared" | tee -a "$LOG_FILE"
  echo "[HINT] Docs: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/" | tee -a "$LOG_FILE"
  START_TUNNEL=0
elif [[ ! -x "$CLOUDFLARED_BIN" ]]; then
  echo "[WARN] cloudflared is not executable: $CLOUDFLARED_BIN — skip tunnel." | tee -a "$LOG_FILE"
  START_TUNNEL=0
fi

if [[ "$START_TUNNEL" -eq 1 ]]; then
  set +e
  "$CLOUDFLARED_BIN" tunnel --url http://127.0.0.1:8000 \
    > "$LOG_DIR/cloudflared.log" 2>&1 &
  CF_PID=$!
  sleep 2
  if ! kill -0 "$CF_PID" 2>/dev/null; then
    echo "[WARN] cloudflared exited immediately; check $LOG_DIR/cloudflared.log" | tee -a "$LOG_FILE"
    START_TUNNEL=0
  fi
  set -e
fi

# ======================
# 8. 提取公网地址（quick tunnel 日志出现较慢，最多轮询约 2 分钟）
# ======================
PUBLIC_URL=""
if [[ "$START_TUNNEL" -eq 1 ]]; then
  for _cf in $(seq 1 60); do
    if [[ -f "$LOG_DIR/cloudflared.log" ]]; then
      PUBLIC_URL=$(
        grep -oE 'https://[^[:space:]]+trycloudflare\.com' "$LOG_DIR/cloudflared.log" 2>/dev/null | head -n 1 || true
      )
    fi
    if [[ -n "$PUBLIC_URL" ]]; then
      break
    fi
    sleep 2
  done
fi

if [[ "$START_TUNNEL" -eq 0 ]]; then
    echo "[INFO] skip tunnel URL detection because cloudflared is unavailable." | tee -a "$LOG_FILE"
elif [ -n "$PUBLIC_URL" ]; then
    echo "[SUCCESS] Public URL: $PUBLIC_URL" | tee -a "$LOG_FILE"
else
    echo "[WARN] cannot detect tunnel URL within ~120s; see: $LOG_DIR/cloudflared.log" | tee -a "$LOG_FILE"
  fi

# ======================
# 9. 保持进程
# ======================
# 在 SSH 前台跑本脚本时，按 Ctrl+C 可能中断 wait 或向进程组发信号，日志里会出现 uvicorn Shutting down。
# 需要常驻请用: tmux/screen/systemd。
wait $SERVER_PID
