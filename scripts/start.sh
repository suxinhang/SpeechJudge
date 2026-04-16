#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_MONGO_SH="${SCRIPT_DIR}/deploy_mongodb.sh"

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
# 3b. 本地 MongoDB（Docker，可选）
# ======================
# 与 infer/rank_jobs_app/core/config.py 默认一致：mongodb://127.0.0.1:27017
# 显式关闭：SPEECHJUDGE_START_LOCAL_MONGO=0|false|no|off
# 端口已可连：视为已有 Mongo（本机服务或其它方式），不再起 Docker 容器。
# 逻辑委托 scripts/deploy_mongodb.sh（容器已存在 / 已运行时会跳过或 docker start）。

_mongo_port="${MONGO_PORT:-27017}"
_skip_local_mongo() {
  local v
  v="$(echo "${SPEECHJUDGE_START_LOCAL_MONGO:-1}" | tr '[:upper:]' '[:lower:]')"
  [[ "$v" == "0" || "$v" == "false" || "$v" == "no" || "$v" == "off" ]]
}

_mongo_port_open() {
  bash -c "echo >/dev/tcp/127.0.0.1/${_mongo_port}" 2>/dev/null
}

_wait_mongo_port() {
  local i
  for i in $(seq 1 30); do
    if _mongo_port_open; then
      return 0
    fi
    sleep 1
  done
  return 1
}

if _skip_local_mongo; then
  echo "[INFO] skip local MongoDB (SPEECHJUDGE_START_LOCAL_MONGO=${SPEECHJUDGE_START_LOCAL_MONGO:-})" | tee -a "$LOG_FILE"
elif _mongo_port_open; then
  echo "[INFO] MongoDB port ${_mongo_port} already accepting connections, skip Docker MongoDB." | tee -a "$LOG_FILE"
elif command -v docker >/dev/null 2>&1 && [[ -f "$DEPLOY_MONGO_SH" ]]; then
  echo "[INFO] ensuring local MongoDB (Docker)..." | tee -a "$LOG_FILE"
  set +e
  bash "$DEPLOY_MONGO_SH" up 2>&1 | tee -a "$LOG_FILE"
  _mongo_up_rc="${PIPESTATUS[0]}"
  set -e
  if [[ "${_mongo_up_rc}" -ne 0 ]]; then
    echo "[WARN] deploy_mongodb.sh exited ${_mongo_up_rc}; check Docker / port binding." | tee -a "$LOG_FILE"
  fi
  if _wait_mongo_port; then
    echo "[INFO] MongoDB port ${_mongo_port} is ready." | tee -a "$LOG_FILE"
  else
    echo "[WARN] MongoDB port ${_mongo_port} not ready after 30s; continuing anyway." | tee -a "$LOG_FILE"
  fi
else
  echo "[WARN] docker not found or missing ${DEPLOY_MONGO_SH}; not starting local MongoDB." | tee -a "$LOG_FILE"
fi

export SPEECHJUDGE_MONGO_URI="${SPEECHJUDGE_MONGO_URI:-mongodb://127.0.0.1:${_mongo_port}}"
export SPEECHJUDGE_MONGO_DB="${SPEECHJUDGE_MONGO_DB:-speechjudge}"
export SPEECHJUDGE_MONGO_COLLECTION="${SPEECHJUDGE_MONGO_COLLECTION:-rank_jobs}"
echo "[INFO] SPEECHJUDGE_MONGO_URI=${SPEECHJUDGE_MONGO_URI}" | tee -a "$LOG_FILE"

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
