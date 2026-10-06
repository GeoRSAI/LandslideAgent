#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

# Mirrors the process patterns started by start_frontend_all.sh (llm_service)
# and admin_start_services in scripts/llm_service.py (seg_service, cls_service).
declare -A SERVICES=(
  [llm_service]="uvicorn scripts.llm_service:app"
  [seg_service]="uvicorn scripts.seg_service:app"
  [cls_service]="uvicorn scripts.cls_service:app"
)

# Order: seg/cls first (children started via /admin/start_services), then
# llm_service last (the parent process holding the GPU model + both LoRA
# adapters).
ORDER=(seg_service cls_service llm_service)

WAIT_SECONDS="${STOP_WAIT_SECONDS:-10}"

any_stopped=0

for name in "${ORDER[@]}"; do
  pattern="${SERVICES[$name]}"
  pids=$(pgrep -f "$pattern" || true)
  if [[ -z "$pids" ]]; then
    echo "${name}: not running"
    continue
  fi

  echo "${name}: stopping (pid $(echo "$pids" | tr '\n' ' '))..."
  # shellcheck disable=SC2086
  kill $pids 2>/dev/null
  any_stopped=1

  waited=0
  while pgrep -f "$pattern" > /dev/null 2>&1; do
    if (( waited >= WAIT_SECONDS )); then
      echo "${name}: still alive after ${WAIT_SECONDS}s, sending SIGKILL"
      pids=$(pgrep -f "$pattern" || true)
      # shellcheck disable=SC2086
      [[ -n "$pids" ]] && kill -9 $pids 2>/dev/null
      break
    fi
    sleep 1
    waited=$((waited + 1))
  done
  echo "${name}: stopped"
done

if [[ "$any_stopped" -eq 0 ]]; then
  echo "Nothing was running."
else
  echo "All requested services stopped."
fi
