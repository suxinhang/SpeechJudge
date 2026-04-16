#!/usr/bin/env bash
# Start/stop/status for a tarball-style MongoDB install (no Docker).
#
# Defaults match common server layout:
#   MONGO_HOME=/root/MonggoDB4   (your extracted tree with bin/mongod)
#   DB_PATH=/root/mongo-data
#   PORT=27017
#   BIND_IP=127.0.0.1
#
# Usage:
#   bash scripts/start_mongodb_standalone.sh start
#   bash scripts/start_mongodb_standalone.sh stop
#   bash scripts/start_mongodb_standalone.sh status
#   bash scripts/start_mongodb_standalone.sh ping   # mongosh one-liner if available
#
# Env overrides: MONGO_HOME DB_PATH PORT BIND_IP LOG_DIR

set -euo pipefail

MONGO_HOME="${MONGO_HOME:-/root/MonggoDB4}"
DB_PATH="${DB_PATH:-/root/mongo-data}"
PORT="${PORT:-27017}"
BIND_IP="${BIND_IP:-127.0.0.1}"
LOG_DIR="${LOG_DIR:-/root/mongo-logs}"

MONGOD="${MONGO_HOME}/bin/mongod"
PID_FILE="${LOG_DIR}/mongod.pid"
LOG_FILE="${LOG_DIR}/mongod.log"

cmd="${1:-start}"

die() { echo "error: $*" >&2; exit 1; }

ensure_layout() {
  [[ -x "$MONGOD" ]] || die "mongod not found or not executable: $MONGOD (set MONGO_HOME?)"
  mkdir -p "$DB_PATH" "$LOG_DIR"
}

is_listening() {
  bash -c "echo >/dev/tcp/${BIND_IP}/${PORT}" 2>/dev/null
}

do_start() {
  ensure_layout
  if is_listening; then
    echo "[INFO] port ${PORT} already open on ${BIND_IP}; skip start."
    return 0
  fi
  if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "[INFO] mongod pid $(cat "$PID_FILE") still running; skip start."
    return 0
  fi
  echo "[INFO] starting mongod: $MONGOD"
  echo "[INFO] dbpath=$DB_PATH bind=${BIND_IP}:${PORT} log=$LOG_FILE"
  "$MONGOD" \
    --dbpath "$DB_PATH" \
    --bind_ip "$BIND_IP" \
    --port "$PORT" \
    --logpath "$LOG_FILE" \
    --pidfilepath "$PID_FILE" \
    --fork
  for _i in $(seq 1 30); do
    if is_listening; then
      echo "[INFO] mongod is listening on ${BIND_IP}:${PORT}"
      return 0
    fi
    sleep 1
  done
  die "mongod did not open ${BIND_IP}:${PORT} within 30s; tail -n 50 $LOG_FILE"
}

do_stop() {
  ensure_layout
  if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "[INFO] sending shutdown (pidfile)..."
    "$MONGOD" --shutdown --dbpath "$DB_PATH" || true
    rm -f "$PID_FILE" 2>/dev/null || true
  else
    echo "[INFO] no pidfile or process dead; try --shutdown anyway"
    "$MONGOD" --shutdown --dbpath "$DB_PATH" 2>/dev/null || true
  fi
  echo "[INFO] stop requested (if mongod was running it should exit shortly)."
}

do_status() {
  if is_listening; then
    echo "[OK] ${BIND_IP}:${PORT} is accepting connections"
  else
    echo "[WARN] ${BIND_IP}:${PORT} is not accepting connections"
  fi
  if [[ -f "$PID_FILE" ]]; then
    echo "pidfile: $PID_FILE -> $(cat "$PID_FILE" 2>/dev/null || echo '?')"
  else
    echo "pidfile: (none)"
  fi
  if [[ -f "$LOG_FILE" ]]; then
    echo "--- last 10 log lines ---"
    tail -n 10 "$LOG_FILE" || true
  fi
}

do_ping() {
  if command -v mongosh >/dev/null 2>&1; then
    mongosh --quiet "mongodb://${BIND_IP}:${PORT}/" --eval "db.runCommand({ ping: 1 })" || die "mongosh ping failed"
  elif command -v mongo >/dev/null 2>&1; then
    mongo --quiet --host "$BIND_IP" --port "$PORT" --eval "db.runCommand({ ping: 1 })" || die "mongo ping failed"
  else
    die "mongosh/mongo not in PATH; install client or use: bash -c 'echo >/dev/tcp/${BIND_IP}/${PORT}'"
  fi
}

case "$cmd" in
  start) do_start ;;
  stop) do_stop ;;
  status) do_status ;;
  ping) do_ping ;;
  *) die "usage: $0 {start|stop|status|ping}" ;;
esac
