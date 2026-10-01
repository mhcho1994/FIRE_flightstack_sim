#!/usr/bin/env bash
# entrypoint.sh — align container user UID/GID with host, optionally run project setup, drop privileges

DEBUG="false"
USER_NAME="user"
WORKSPACE="/home/${USER_NAME}/FIRE_flightstack_sim"
INSTALL_DIR="${WORKSPACE}/install"
CHOWN_PATHS=()

help() {
  cat <<EOF
Usage:
  entrypoint.sh [options] [--] [command...]

Options:
  -h, --help        Show this help and exit
  --debug           Enable command tracing (set -x)
  --chown PATH      Recursively chown PATH to ${USER_NAME}:${USER_NAME}
                    Can be specified multiple times.
  --skip-setup      Skip first-time project setup

Environment:
  HOST_UID          Required host user id
  HOST_GID          Required host group id
  HOST_USER_NAME    Optional host user name
  HOST_GROUP_NAME   Optional host group name
EOF
}

die() { echo "ERROR: $*" >&2; exit 1; }

SKIP_SETUP="false"
CMD=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) help; exit 0 ;;
    --debug) DEBUG="true"; shift ;;
    --chown)
      [[ $# -ge 2 ]] || die "--chown requires a path"
      CHOWN_PATHS+=("$2")
      shift 2
      ;;
    --skip-setup)
      SKIP_SETUP="true"
      shift
      ;;
    --)
      shift
      CMD=("$@")
      break
      ;;
    -*)
      die "Unknown option: $1 (use --help)"
      ;;
    *)
      CMD=("$@")
      break
      ;;
  esac
done

set -Eeuo pipefail
trap 'echo "[entrypoint.sh] ERROR line=$LINENO cmd=$BASH_COMMAND" >&2' ERR
[[ "${DEBUG}" == "true" ]] && set -x

# Readiness belongs to this start, unlike the persistent project setup flag.
READY_FILE="/tmp/fire_flightstack_sim.ready"
rm -f "${READY_FILE}"

[[ -n "${HOST_UID:-}" ]] || die "please set HOST_UID"
[[ -n "${HOST_GID:-}" ]] || die "please set HOST_GID"

HOST_USER_NAME="${HOST_USER_NAME:-unknown}"
HOST_GROUP_NAME="${HOST_GROUP_NAME:-unknown}"

echo "[ENTRYPOINT] Host user : ${HOST_USER_NAME}"
echo "[ENTRYPOINT] Host group: ${HOST_GROUP_NAME}"
echo "[ENTRYPOINT] Mapping ${USER_NAME} -> ${HOST_UID}:${HOST_GID}"

if [[ "$(id -u)" -ne 0 ]]; then
  die "Entrypoint must run as root"
fi

current_gid="$(id -g "${USER_NAME}")"
if [[ "${current_gid}" != "${HOST_GID}" ]]; then
  groupmod --gid "${HOST_GID}" "${USER_NAME}"
fi

current_uid="$(id -u "${USER_NAME}")"
if [[ "${current_uid}" != "${HOST_UID}" ]]; then
  usermod --uid "${HOST_UID}" "${USER_NAME}"
fi

for path in "${CHOWN_PATHS[@]}"; do
  if [[ -e "${path}" ]]; then
    echo "[ENTRYPOINT] Chowning ${path} -> ${USER_NAME}:${USER_NAME}"
    chown -R "${USER_NAME}:${USER_NAME}" "${path}" || true
  else
    echo "[ENTRYPOINT] WARNING: ${path} does not exist, skipping"
  fi
done

SETUP_FLAG="${WORKSPACE}/.docker_home/.setup_done"

if [[ "${SKIP_SETUP}" != "true" && ! -f "${SETUP_FLAG}" ]]; then
  echo "[ENTRYPOINT] Running first-time setup..."

  sudo -u "${USER_NAME}" -H bash -lc "
    set -euo pipefail
    cd '${WORKSPACE}'

    if [[ -f '${INSTALL_DIR}/autopilot.sh' ]]; then
      echo '[SETUP] autopilot build and environment setup'
      bash '${INSTALL_DIR}/autopilot.sh' --phase build --with-ardupilot --project-root '${WORKSPACE}'
      bash '${INSTALL_DIR}/autopilot.sh' --phase env --with-ardupilot --project-root '${WORKSPACE}'
    fi

    if [[ -f '${INSTALL_DIR}/gazebo.sh' ]]; then
      echo '[SETUP] gazebo environment setup'
      bash '${INSTALL_DIR}/gazebo.sh' --install binary --phase env
    fi

    if [[ -f '${INSTALL_DIR}/extra.sh' ]]; then
      echo '[SETUP] extra environment setup'
      bash '${INSTALL_DIR}/extra.sh' --phase env --project-root '${WORKSPACE}'
    fi

    if [[ -f '${INSTALL_DIR}/usersetup.sh' ]]; then
      echo '[SETUP] user setup'
      bash '${INSTALL_DIR}/usersetup.sh' --project-root '${WORKSPACE}'
    fi
  "

  touch "${SETUP_FLAG}"
  chown "${USER_NAME}:${USER_NAME}" "${SETUP_FLAG}"
  echo "[ENTRYPOINT] Setup completed."
else
  echo "[ENTRYPOINT] Setup already done or skipped."
fi

# Do not interpolate command arguments into shell code: commands such as
# `sleep infinity` and arguments containing spaces must retain their boundaries.
if [[ ${#CMD[@]} -eq 0 ]]; then
  CMD=(bash)
fi
cd "${WORKSPACE}"
touch "${READY_FILE}"
exec sudo -u "${USER_NAME}" -H bash -lc 'cd "$1" && shift && exec "$@"' \
  -- "${WORKSPACE}" "${CMD[@]}"
