#!/usr/bin/env bash
# Linux-only smoke test: rank_jobs_app 能完成启动并响应 /health（任务状态存 JSON 文件，无需 Mongo）。
#
# 用法（在仓库任意目录）:
#   bash scripts/verify_service_linux.sh
#
# 环境变量（可选）:
#   VERIFY_PORT                 默认 8010，避免与已有 8000 冲突
#   PYTHON_BIN                  默认 python3
#   SPEECHJUDGE_MODEL_PATH      模型目录；未设则仅检查常见默认路径并告警
#   SPEECHJUDGE_RANK_JOB_DIR    任务工作目录（默认 repo/data/rank_jobs）
#   SPEECHJUDGE_RANK_JOBS_JSON_DIR  任务 JSON 目录（默认 $SPEECHJUDGE_RANK_JOB_DIR/_json_jobs）
#   SPEECHJUDGE_PAIRWISE_PARALLEL   POST 未传 pairwise_parallel 时的默认并发数（1=串行并加锁）
#   SPEECHJUDGE_PREPARE_PARALLEL    POST 未传 prepare_parallel 时，准备阶段 URL 下载/转码默认并发（1–32）
#   SPEECHJUDGE_PREPARE_DOWNLOAD_ATTEMPTS  单 URL HTTP 下载最大尝试次数（含 5xx/429 与网络类重试，默认 5，上限 15）
#   SPEECHJUDGE_PREPARE_DECODE_ATTEMPTS    单文件 ffmpeg/librosa 转 wav 最大尝试次数（默认 3，上限 10）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
INFER="$REPO_ROOT/infer"

VERIFY_PORT="${VERIFY_PORT:-8010}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

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
