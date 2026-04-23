#!/usr/bin/env bash
# 基于 scripts/start.sh 的启动范式：
# 1) 初始化环境 -> 2) 清理旧服务 -> 3) 启动 uvicorn -> 4) 启动 cloudflared 并提取公网地址。
# 本脚本使用本地 JSON 目录存任务，不依赖 MongoDB。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ======================
# 1. conda 环境（默认必需）
# ======================
if [[ "${SKIP_CONDA_ACTIVATE:-0}" != "1" ]]; then
  if [[ -z "${SPEECHJUDGE_CONDA_SH:-}" ]] && [[ -f "/root/miniconda3/etc/profile.d/conda.sh" ]]; then
    SPEECHJUDGE_CONDA_SH="/root/miniconda3/etc/profile.d/conda.sh"
  fi

  if [[ -z "${SPEECHJUDGE_CONDA_SH:-}" ]] || [[ ! -f "$SPEECHJUDGE_CONDA_SH" ]]; then
    echo "[ERROR] conda.sh not found. Set SPEECHJUDGE_CONDA_SH or export SKIP_CONDA_ACTIVATE=1 if you really want to skip."
    exit 1
  fi

  if [[ -n "${SPEECHJUDGE_CONDA_SH:-}" ]] && [[ -f "$SPEECHJUDGE_CONDA_SH" ]]; then
    # shellcheck source=/dev/null
    source "$SPEECHJUDGE_CONDA_SH"
  fi

  if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] conda command not available after sourcing $SPEECHJUDGE_CONDA_SH"
    exit 1
  fi

  _conda_env="${SPEECHJUDGE_CONDA_ENV:-speechjudge}"
  set +e
  conda activate "$_conda_env"
  _conda_rc=$?
  set -e
  if [[ "$_conda_rc" -ne 0 ]]; then
    echo "[ERROR] conda activate ${_conda_env} failed."
    exit 1
  fi
fi

cd "$REPO_ROOT"

# ======================
# 2. 环境变量
# ======================
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export SPEECHJUDGE_MODEL_PATH="${SPEECHJUDGE_MODEL_PATH:-${REPO_ROOT}/infer/pretrained/SpeechJudge-GRM}"
export SPEECHJUDGE_VLLM_API_DATA="${SPEECHJUDGE_VLLM_API_DATA:-${REPO_ROOT}/data/vllm_grm_jobs}"
# 不需要 MongoDB：vllm_grm_api 使用本地 JSON 文件存储任务。

# 统一端口配置：支持外部覆盖，默认 8000。
API_PORT="${SPEECHJUDGE_API_PORT:-8000}"

# ======================
# 3. 日志目录
# ======================
LOG_DIR="${REPO_ROOT}/logs/vllm_grm_api"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/server_$(date +%Y%m%d_%H%M%S).log"

echo "[INFO] repo: $REPO_ROOT" | tee -a "$LOG_FILE"
echo "[INFO] SPEECHJUDGE_MODEL_PATH=$SPEECHJUDGE_MODEL_PATH" | tee -a "$LOG_FILE"
echo "[INFO] SPEECHJUDGE_VLLM_API_DATA=$SPEECHJUDGE_VLLM_API_DATA" | tee -a "$LOG_FILE"
echo "[INFO] API_PORT=$API_PORT" | tee -a "$LOG_FILE"

# ======================
# 4. 先停止旧服务（释放 GPU / 端口）
# ======================
echo "[INFO] nvidia-smi (compute apps) before cleanup:" | tee -a "$LOG_FILE"
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv 2>/dev/null | tee -a "$LOG_FILE" || true

echo "[INFO] stopping SGLang server if running..." | tee -a "$LOG_FILE"
pkill -f "sglang.launch_server" || true
pkill -f "sglang\.router" || true

echo "[INFO] stopping standalone vLLM / TGI-style entrypoints if any..." | tee -a "$LOG_FILE"
pkill -f "vllm.entrypoints" || true
pkill -f "vllm serve" || true
pkill -f "tgi.entrypoint" || true
pkill -f "text-generation-launcher" || true

echo "[INFO] stopping old uvicorn process..." | tee -a "$LOG_FILE"
pkill -f "uvicorn rank_jobs_app.app.main:app" || true
pkill -f "rank_jobs_app.app.main:app" || true
pkill -f "uvicorn vllm_grm_api.app.main:app" || true
pkill -f "vllm_grm_api.app.main:app" || true

echo "[INFO] stopping old cloudflared process..." | tee -a "$LOG_FILE"
pkill -f "cloudflared tunnel" || true

if command -v fuser >/dev/null 2>&1; then
  echo "[INFO] freeing TCP ${API_PORT} if still held (fuser -k)..." | tee -a "$LOG_FILE"
  fuser -k "${API_PORT}/tcp" 2>/dev/null || true
fi

sleep 4

echo "[INFO] nvidia-smi (compute apps) after cleanup:" | tee -a "$LOG_FILE"
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv 2>/dev/null | tee -a "$LOG_FILE" || true

# ======================
# 5. 启动 API（后台）
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

# ======================
# 6. 等待服务 ready
# ======================
HEALTH_POLL_INTERVAL="${SPEECHJUDGE_HEALTH_POLL_INTERVAL:-5}"
HEALTH_MAX_TRIES="${SPEECHJUDGE_HEALTH_MAX_TRIES:-120}"
API_READY=0

echo "[INFO] waiting for /health on port ${API_PORT} (every ${HEALTH_POLL_INTERVAL}s, max ${HEALTH_MAX_TRIES} tries) ..." | tee -a "$LOG_FILE"
for _i in $(seq 1 "$HEALTH_MAX_TRIES"); do
  if curl -sf "http://127.0.0.1:${API_PORT}/health" >/dev/null; then
    echo "[SUCCESS] API is ready (model load may still be finishing on first request)." | tee -a "$LOG_FILE"
    API_READY=1
    break
  fi

  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[ERROR] uvicorn process exited before /health became ready; see $LOG_FILE" | tee -a "$LOG_FILE"
    exit 1
  fi

  echo "[INFO] health check ${_i}/${HEALTH_MAX_TRIES}: not ready yet, retry in ${HEALTH_POLL_INTERVAL}s ..." | tee -a "$LOG_FILE"
  sleep "$HEALTH_POLL_INTERVAL"
done

if [[ "$API_READY" -ne 1 ]]; then
  echo "[WARN] /health did not become ready after $((HEALTH_POLL_INTERVAL * HEALTH_MAX_TRIES))s; continuing startup flow." | tee -a "$LOG_FILE"
fi

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

# 已解析到文件但无执行位：自动 chmod 一次再继续
if [[ -n "${CLOUDFLARED_BIN:-}" ]] && [[ -f "$CLOUDFLARED_BIN" ]] && [[ ! -x "$CLOUDFLARED_BIN" ]]; then
  if chmod +x "$CLOUDFLARED_BIN" 2>/dev/null && [[ -x "$CLOUDFLARED_BIN" ]]; then
    echo "[INFO] chmod +x cloudflared: $CLOUDFLARED_BIN" | tee -a "$LOG_FILE"
  fi
fi

# 未找到或未可执行：Linux 下尝试下载官方二进制到 /root/bin。
if [[ -z "${CLOUDFLARED_BIN:-}" ]] || [[ ! -x "$CLOUDFLARED_BIN" ]]; then
  if [[ -n "${CLOUDFLARED_BIN:-}" ]] && [[ ! -x "$CLOUDFLARED_BIN" ]]; then
    echo "[WARN] cloudflared still not executable after chmod attempt: $CLOUDFLARED_BIN" | tee -a "$LOG_FILE"
  fi

  if [[ "${SKIP_CLOUDFLARED_DOWNLOAD:-}" != "1" ]] && [[ "$(uname -s)" == "Linux" ]]; then
    _cf_arch=""
    case "$(uname -m)" in
      x86_64) _cf_arch="amd64" ;;
      aarch64|arm64) _cf_arch="arm64" ;;
    esac

    if [[ -n "$_cf_arch" ]] && { command -v curl >/dev/null 2>&1 || command -v wget >/dev/null 2>&1; }; then
      echo "[INFO] cloudflared missing; downloading cloudflared-linux-${_cf_arch} -> /root/bin/cloudflared ..." | tee -a "$LOG_FILE"
      mkdir -p /root/bin
      _cf_tmp="/root/bin/cloudflared.tmp.$$"
      _cf_url="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-${_cf_arch}"
      _cf_dl=1

      set +e
      for _try in 1 2 3; do
        rm -f "$_cf_tmp"
        if command -v curl >/dev/null 2>&1; then
          curl -fsSL "$_cf_url" -o "$_cf_tmp" 2>/dev/null && [[ -s "$_cf_tmp" ]] && _cf_dl=0 && break
        fi
        if command -v wget >/dev/null 2>&1; then
          wget -q "$_cf_url" -O "$_cf_tmp" 2>/dev/null && [[ -s "$_cf_tmp" ]] && _cf_dl=0 && break
        fi
        [[ "$_try" -lt 3 ]] && sleep 4
      done
      set -e

      if [[ "$_cf_dl" -eq 0 ]] && [[ -s "$_cf_tmp" ]]; then
        chmod +x "$_cf_tmp"
        mv -f "$_cf_tmp" /root/bin/cloudflared
        CLOUDFLARED_BIN="/root/bin/cloudflared"
        echo "[SUCCESS] installed cloudflared at $CLOUDFLARED_BIN" | tee -a "$LOG_FILE"
      else
        rm -f "$_cf_tmp"
        echo "[WARN] cloudflared auto-download failed after retries (TLS/proxy/GitHub?)." | tee -a "$LOG_FILE"
      fi
    elif [[ -z "$_cf_arch" ]]; then
      echo "[WARN] unsupported machine for cloudflared auto-download: $(uname -m)." | tee -a "$LOG_FILE"
    else
      echo "[WARN] need curl or wget to auto-download cloudflared." | tee -a "$LOG_FILE"
    fi
  elif [[ "${SKIP_CLOUDFLARED_DOWNLOAD:-}" == "1" ]]; then
    echo "[INFO] SKIP_CLOUDFLARED_DOWNLOAD=1; not attempting cloudflared download." | tee -a "$LOG_FILE"
  fi
fi

START_TUNNEL=1
if [[ -z "${CLOUDFLARED_BIN:-}" ]]; then
  echo "[WARN] cloudflared not available; skip public tunnel." | tee -a "$LOG_FILE"
  START_TUNNEL=0
elif [[ ! -x "$CLOUDFLARED_BIN" ]]; then
  echo "[WARN] cloudflared is not executable: $CLOUDFLARED_BIN — skip tunnel." | tee -a "$LOG_FILE"
  START_TUNNEL=0
fi

if [[ "$START_TUNNEL" -eq 1 ]]; then
  : >"$LOG_DIR/cloudflared.log"
  set +e
  "$CLOUDFLARED_BIN" tunnel --url "http://127.0.0.1:${API_PORT}" \
    >>"$LOG_DIR/cloudflared.log" 2>&1 &
  CF_PID=$!
  sleep 2
  if ! kill -0 "$CF_PID" 2>/dev/null; then
    echo "[WARN] cloudflared exited immediately; check $LOG_DIR/cloudflared.log" | tee -a "$LOG_FILE"
    START_TUNNEL=0
  fi
  set -e
fi

# ======================
# 8. 提取公网地址（最多轮询约 3 分钟）
# ======================
PUBLIC_URL=""
if [[ "$START_TUNNEL" -eq 1 ]]; then
  for _cf in $(seq 1 90); do
    if [[ -f "$LOG_DIR/cloudflared.log" ]]; then
      PUBLIC_URL="$(
        grep -oE 'https://[^[:space:]]+(trycloudflare\.com|cfargotunnel\.com)' "$LOG_DIR/cloudflared.log" 2>/dev/null | head -n 1 || true
      )"
    fi
    if [[ -n "$PUBLIC_URL" ]]; then
      break
    fi
    sleep 2
  done
fi

if [[ "$START_TUNNEL" -eq 0 ]]; then
  echo "[INFO] skip tunnel URL detection because cloudflared is unavailable." | tee -a "$LOG_FILE"
elif [[ -n "$PUBLIC_URL" ]]; then
  echo "[SUCCESS] Public URL: $PUBLIC_URL" | tee -a "$LOG_FILE"
else
  echo "[WARN] cannot detect tunnel URL within ~180s; see: $LOG_DIR/cloudflared.log" | tee -a "$LOG_FILE"
fi

# ======================
# 9. 保持进程
# ======================
wait "$SERVER_PID"
