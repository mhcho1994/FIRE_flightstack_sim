#!/usr/bin/env bash
set -uo pipefail

# -----------------------------
# Pass host user info into container
# (used by entrypoint for UID/GID mapping)
# -----------------------------
export HOST_UID="$(id -u)"
export HOST_GID="$(id -g)"
export HOST_USER_NAME="$(id -un)"
export HOST_GROUP_NAME="$(id -gn)"

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="$(cd "${THIS_DIR}/.." && pwd)"
COMPOSE_FILE="${PROJECT_ROOT}/docker/compose.yaml"
COMPOSE=(docker compose -f "${COMPOSE_FILE}" --project-directory "${PROJECT_ROOT}")

# X11 forwarding for GUI apps (QGC, Gazebo)
export DISPLAY="${DISPLAY:-:0}"

RECREATE=false
RESET_SETUP=false
USE_EXEC=false

# -----------------------------
# Optional flags
#   --recreate : force container recreate
#   --exec     : enter container with bash shell
# -----------------------------
for arg in "$@"; do
    case "$arg" in
        --recreate) RECREATE=true; RESET_SETUP=true ;;
        --exec)    USE_EXEC=true ;;
    esac
done

SERVICE="FIRE_flightstack_simulator"
CONTAINER_NAME="fire_flightstack_sim"

# Setup sentinel used by entrypoint
SETUP_DONE=".docker_home/.setup_done"

cleanup() {
    xhost -local:docker || true
}

trap cleanup EXIT
xhost +local:docker

container_display() {
    docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "${CONTAINER_NAME}" 2>/dev/null \
        | awk -F= '$1=="DISPLAY" {print $2; exit}'
}

if docker ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
    EXISTING_DISPLAY="$(container_display || true)"
    if [[ -n "${EXISTING_DISPLAY}" && "${EXISTING_DISPLAY}" != "${DISPLAY}" ]]; then
        echo "[INFO] Container DISPLAY (${EXISTING_DISPLAY}) differs from host DISPLAY (${DISPLAY}); recreating container."
        RECREATE=true
    fi
fi


# -----------------------------
# Enter container either by:
# 1) attach to main process
# 2) exec into interactive shell
# -----------------------------
enter_container() {
    if $USE_EXEC; then
        echo "[INFO] Entering container via exec..."
        "${COMPOSE[@]}" exec -u user -it "${SERVICE}" bash
    else
        echo "[INFO] Attaching to container..."
        "${COMPOSE[@]}" attach "${SERVICE}"
    fi
}


if $RECREATE; then
    # -----------------------------
    # Force container recreate only
    # Remove setup sentinel so entrypoint
    # reruns project setup
    # -----------------------------
    if $RESET_SETUP && [[ -f "${SETUP_DONE}" ]]; then
        echo "[INFO] Removing ${SETUP_DONE}"
        rm -f "${SETUP_DONE}"
    fi

    if "${COMPOSE[@]}" ps -a --services --filter status=running \
        | grep -q "^${SERVICE}$"; then
        "${COMPOSE[@]}" down
    fi

    echo "[INFO] Recreating container..."
    "${COMPOSE[@]}" up --force-recreate

else
    if "${COMPOSE[@]}" ps -a --services --filter status=running \
        | grep -q "^${SERVICE}$"; then

        # -----------------------------
        # Container already running:
        # reuse it (no recreate)
        # -----------------------------
        echo "[INFO] Existing container already running."
        enter_container

    elif "${COMPOSE[@]}" ps -a --services \
        | grep -q "^${SERVICE}$"; then

        # -----------------------------
        # Container exists but stopped:
        # restart existing one only
        # -----------------------------
        echo "[INFO] Starting existing container..."
        "${COMPOSE[@]}" start "${SERVICE}"
        enter_container

    else
        # -----------------------------
        # First run:
        # build image and create container
        # -----------------------------
        if [[ -f "${SETUP_DONE}" ]]; then
            echo "[INFO] Removing ${SETUP_DONE}"
            rm -f "${SETUP_DONE}"
        fi

        echo "[INFO] No container found. Building first time..."
        "${COMPOSE[@]}" up
    fi
fi
