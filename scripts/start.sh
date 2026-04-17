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
# Odd-even rank: concurrent pairwise compares per phase (default 5). Use 1 for fully serial + infer lock.
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
START_TUNNEL=1
if [[ -z "$CLOUDFLARED_BIN" ]]; then
  echo "[WARN] cloudflared not found in PATH; skip public tunnel." | tee -a "$LOG_FILE"
  START_TUNNEL=0
elif [[ ! -x "$CLOUDFLARED_BIN" ]]; then
  echo "[WARN] cloudflared exists but is not executable: $CLOUDFLARED_BIN" | tee -a "$LOG_FILE"
  echo "[WARN] fix with: chmod +x \"$CLOUDFLARED_BIN\" (or install as executable), then retry." | tee -a "$LOG_FILE"
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
