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

# ======================
# 3. 日志目录
# ======================
LOG_DIR="/root/SpeechJudge/logs"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/server_$(date +%Y%m%d_%H%M%S).log"

echo "[INFO] starting SpeechJudge..." | tee -a "$LOG_FILE"

# ======================
# 4. 先停止旧服务（关键）
# ======================
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

cloudflared tunnel --url http://127.0.0.1:8000 \
  > "$LOG_DIR/cloudflared.log" 2>&1 &

sleep 5

# ======================
# 8. 提取公网地址
# ======================
PUBLIC_URL=$(grep -o 'https://.*trycloudflare.com' "$LOG_DIR/cloudflared.log" | head -n 1 || true)

if [ -n "$PUBLIC_URL" ]; then
    echo "[SUCCESS] Public URL: $PUBLIC_URL" | tee -a "$LOG_FILE"
else
    echo "[WARN] cannot detect tunnel URL, check logs" | tee -a "$LOG_FILE"
fi

# ======================
# 9. 保持进程
# ======================
wait $SERVER_PID
