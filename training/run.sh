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

mkdir -p data output recordings/positive recordings/negative
exec docker compose --profile cpu run --rm "$service" "$@"
