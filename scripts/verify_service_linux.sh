#!/usr/bin/env bash
# Linux-only smoke test: rank_jobs_app 能完成启动（含 Mongo + 模型加载）并响应 /health。
# 不依赖 /root 路径或 conda；使用当前环境中的 python3（线上自行保证 venv/conda 已 activate）。
#
# 用法（在仓库任意目录）:
#   bash scripts/verify_service_linux.sh
#
# 环境变量（可选）:
#   VERIFY_PORT                 默认 8010，避免与已有 8000 冲突
#   PYTHON_BIN                  默认 python3
#   SPEECHJUDGE_START_LOCAL_MONGO  与 scripts/start.sh 相同：0/false/no/off 则不起 Docker Mongo
#   MONGO_PORT                  默认 27017
#   SPEECHJUDGE_MONGO_URI 等    与 rank_jobs_app 一致；未设时在本地 Mongo 场景下会默认 127.0.0.1
#   SPEECHJUDGE_MODEL_PATH      模型目录；未设则仅检查常见默认路径并告警

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
INFER="$REPO_ROOT/infer"
DEPLOY_MONGO_SH="${SCRIPT_DIR}/deploy_mongodb.sh"

VERIFY_PORT="${VERIFY_PORT:-8010}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

_mongo_port="${MONGO_PORT:-27017}"

_skip_local_mongo() {
  local v
  v="$(echo "${SPEECHJUDGE_START_LOCAL_MONGO:-1}" | tr '[:upper:]' '[:lower:]')"
  [[ "$v" == "0" || "$v" == "false" || "$v" == "no" || "$v" == "off" ]]
}

_mongo_port_open() {
  bash -c "echo >/dev/tcp/127.0.0.1/${_mongo_port}" 2>/dev/null
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "error: missing command: $1" >&2
    exit 1
  }
}

need_cmd curl
need_cmd "$PYTHON_BIN"

if [[ ! -d "$INFER" ]]; then
  echo "error: infer directory not found: $INFER" >&2
  exit 1
fi

echo "[verify] repo: $REPO_ROOT"
echo "[verify] infer: $INFER"

# ---- optional local Mongo (Docker), same idea as start.sh ----
if _skip_local_mongo; then
  echo "[verify] skip local MongoDB (SPEECHJUDGE_START_LOCAL_MONGO=${SPEECHJUDGE_START_LOCAL_MONGO:-})"
elif _mongo_port_open; then
  echo "[verify] MongoDB port ${_mongo_port} already open, skip Docker MongoDB."
elif command -v docker >/dev/null 2>&1 && [[ -f "$DEPLOY_MONGO_SH" ]]; then
  echo "[verify] ensuring local MongoDB (Docker)..."
  set +e
  bash "$DEPLOY_MONGO_SH" up
  _mongo_rc=$?
  set -e
  if [[ "$_mongo_rc" -ne 0 ]]; then
    echo "warn: deploy_mongodb.sh exited $_mongo_rc (continuing; ensure Mongo is reachable)" >&2
  fi
  for _i in $(seq 1 30); do
    if _mongo_port_open; then
      echo "[verify] MongoDB port ${_mongo_port} is ready."
      break
    fi
    sleep 1
  done
else
  echo "warn: no docker or missing $DEPLOY_MONGO_SH; assuming Mongo is already available" >&2
fi

export SPEECHJUDGE_MONGO_URI="${SPEECHJUDGE_MONGO_URI:-mongodb://127.0.0.1:${_mongo_port}}"
export SPEECHJUDGE_MONGO_DB="${SPEECHJUDGE_MONGO_DB:-speechjudge}"
export SPEECHJUDGE_MONGO_COLLECTION="${SPEECHJUDGE_MONGO_COLLECTION:-rank_jobs}"
echo "[verify] SPEECHJUDGE_MONGO_URI=$SPEECHJUDGE_MONGO_URI"

_mp="${SPEECHJUDGE_MODEL_PATH:-}"
if [[ -z "$_mp" ]]; then
  if [[ -d "$REPO_ROOT/pretrained/SpeechJudge-GRM" ]]; then
    export SPEECHJUDGE_MODEL_PATH="$REPO_ROOT/pretrained/SpeechJudge-GRM"
    echo "[verify] SPEECHJUDGE_MODEL_PATH=$SPEECHJUDGE_MODEL_PATH"
  else
    echo "warn: SPEECHJUDGE_MODEL_PATH unset and $REPO_ROOT/pretrained/SpeechJudge-GRM not found; load may fail" >&2
  fi
else
  echo "[verify] SPEECHJUDGE_MODEL_PATH=$SPEECHJUDGE_MODEL_PATH"
fi

cd "$INFER"

UVICORN_PID=""
cleanup() {
  if [[ -n "${UVICORN_PID}" ]] && kill -0 "${UVICORN_PID}" 2>/dev/null; then
    kill "${UVICORN_PID}" 2>/dev/null || true
    wait "${UVICORN_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[verify] starting uvicorn on 127.0.0.1:$VERIFY_PORT ..."
"$PYTHON_BIN" -m uvicorn rank_jobs_app.app.main:app \
  --host 127.0.0.1 \
  --port "$VERIFY_PORT" \
  --log-level warning &
UVICORN_PID=$!

echo "[verify] waiting for GET /health (model load may take several minutes) ..."
_ok=0
for _i in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:${VERIFY_PORT}/health" >/dev/null; then
    _ok=1
    break
  fi
  sleep 2
done

if [[ "$_ok" -ne 1 ]]; then
  echo "error: /health did not become ready in time" >&2
  exit 1
fi

echo "[verify] OK — service responds on http://127.0.0.1:${VERIFY_PORT}/health"
curl -sS "http://127.0.0.1:${VERIFY_PORT}/health" | head -c 2000
echo

exit 0
