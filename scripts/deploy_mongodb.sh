#!/usr/bin/env bash
# Deploy MongoDB via Docker (optional; rank_jobs_app uses JSON files by default).
#
# Default URI if you point other tools at Mongo:
#   SPEECHJUDGE_MONGO_URI=mongodb://127.0.0.1:27017
#   SPEECHJUDGE_MONGO_DB=speechjudge
#
# Requires: Docker
#
# Usage:
#   ./scripts/deploy_mongodb.sh up      # start (default)
#   ./scripts/deploy_mongodb.sh down    # stop and remove container
#   ./scripts/deploy_mongodb.sh status
#   ./scripts/deploy_mongodb.sh logs
#
# Env overrides:
#   MONGO_CONTAINER_NAME  (default: speechjudge-mongo)
#   MONGO_IMAGE           (default: mongo:7)
#   MONGO_PORT            (default: 27017, host -> container 27017)
#   MONGO_DATA_VOLUME     (default: speechjudge_mongo_data) named volume for /data/db

set -euo pipefail

CONTAINER_NAME="${MONGO_CONTAINER_NAME:-speechjudge-mongo}"
IMAGE="${MONGO_IMAGE:-mongo:7}"
PORT="${MONGO_PORT:-27017}"
VOLUME="${MONGO_DATA_VOLUME:-speechjudge_mongo_data}"

cmd="${1:-up}"

die() {
  echo "error: $*" >&2
  exit 1
}

need_docker() {
  command -v docker >/dev/null 2>&1 || die "docker not found; install Docker first."
}

case "$cmd" in
  up|start)
    need_docker
    if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
      if docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
        echo "MongoDB container '$CONTAINER_NAME' is already running."
      else
        echo "Starting existing container '$CONTAINER_NAME'..."
        docker start "$CONTAINER_NAME"
      fi
    else
      echo "Creating MongoDB $IMAGE as '$CONTAINER_NAME' on host port $PORT..."
      docker run -d \
        --name "$CONTAINER_NAME" \
        --restart unless-stopped \
        -p "${PORT}:27017" \
        -v "${VOLUME}:/data/db" \
        "$IMAGE" \
        --bind_ip_all
    fi
    echo ""
    echo "Ready. Point rank_jobs_app at:"
    echo "  export SPEECHJUDGE_MONGO_URI=mongodb://127.0.0.1:${PORT}"
    echo "  # optional:"
    echo "  export SPEECHJUDGE_MONGO_DB=speechjudge"
    echo "  export SPEECHJUDGE_MONGO_COLLECTION=rank_jobs"
    ;;
  down|stop)
    need_docker
    docker rm -f "$CONTAINER_NAME" 2>/dev/null || echo "Container '$CONTAINER_NAME' not found (nothing to remove)."
    ;;
  status)
    need_docker
    docker ps -a --filter "name=^/${CONTAINER_NAME}$" --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
    ;;
  logs)
    need_docker
    docker logs -f "$CONTAINER_NAME"
    ;;
  *)
    die "unknown command: $cmd (use: up | down | status | logs)"
    ;;
esac
