#!/usr/bin/env bash
# Startet einen Schritt der Trainings-Pipeline im Docker-Container.
#   ./run.sh check-gpu | download | import | preview | tts | features | train | evaluate | all
# Ohne GPU:  WWTRAIN_CPU=1 ./run.sh ...
set -euo pipefail
cd "$(dirname "$0")"

export HOST_UID="$(id -u)" HOST_GID="$(id -g)"
export VIDEO_GID="$(getent group video | cut -d: -f3 || true)"
export RENDER_GID="$(getent group render | cut -d: -f3 || true)"
[ -n "$VIDEO_GID" ] || export VIDEO_GID=44
# docker compose verbietet doppelte Einträge; 65534 = nogroup als Platzhalter
if [ -z "$RENDER_GID" ] || [ "$RENDER_GID" = "$VIDEO_GID" ]; then export RENDER_GID=65534; fi

service=wwtrain
[ "${WWTRAIN_CPU:-0}" = "1" ] && service=wwtrain-cpu

compose=(docker compose -f docker-compose.yml)
# Windows/WSL2: GPU über /dev/dxg statt /dev/kfd
if [ -e /dev/dxg ] && [ ! -e /dev/kfd ]; then
  compose+=(-f docker-compose.wsl.yml)
  if [ "$service" = wwtrain ] && [ ! -e "${ROCDXG_LIB:-/opt/rocm/lib/librocdxg.so}" ]; then
    echo "WSL erkannt, aber librocdxg fehlt (${ROCDXG_LIB:-/opt/rocm/lib/librocdxg.so})." >&2
    echo "Siehe README, Abschnitt «Windows». Ohne GPU: WWTRAIN_CPU=1 ./run.sh ..." >&2
    exit 1
  fi
fi

mkdir -p data output recordings/positive recordings/negative
exec "${compose[@]}" --profile cpu run --rm "$service" "$@"
