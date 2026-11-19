#!/usr/bin/env bash
# get_src.sh — fetch source repos for flightstack_sim
#
# What this script does (idempotent):
#  - Ensures the project workspace layout exists
#  - Clones PX4 into ap/px4 (default ref: v1.16.1) WITH submodules via --recursive (default ON)
#  - Clones px4_msgs into ros/px4_msgs_ws/src/px4_msgs (default ON)
#  - Clones ArduPilot into ap/ardupilot WITH submodules via --recurse-submodules (default OFF)
#
# Repos:
#  - PX4-Autopilot : git clone --recursive
#  - px4_msgs : plain clone
#  - ArduPilot : git clone --recurse-submodules (enabled with --with-ardupilot)
#
# Notes:
#  - This script should be executed (bash get_src.sh), not sourced.
#  - It avoids destructive operations; it only clones when directories are missing/empty.
#
# Final layout (default):
#   <project-root>/
#     ap/px4
#     ap/ardupilot
#     ros/px4_msgs_ws/src/px4_msgs

# --------------------------
# Defaults
# --------------------------
DEBUG="false"
PROJECT_ROOT="$(pwd)"                 # assume invoked from flightstack_sim root

# Clone behavior
UPDATE_SUBMODULES="true"              # always init submodules by default

# PX4
PX4_REF="v1.16.1"
PX4_DIR_REL="ap/px4"

# px4_msgs (ROS2 message package)
PX4_MSGS_REF="main"
PX4_MSGS_DIR_REL="ros/px4_msgs_ws/src/px4_msgs"

# Optional ArduPilot (Default enabled)
WITH_ARDUPILOT="false"
ARDUPILOT_REF="Copter-4.6.2"
ARDUPILOT_DIR_REL="ap/ardupilot"

help() {
  cat <<EOF
Usage:
  bash get_src.sh [options]

Options:
  -h, --help              Show this help and exit
  --debug                 Enable command tracing (set -x)
  --project-root PATH     Project root (default: current directory)

  # Select autopilots
  --no-px4                   Disable PX4 setup (default: off)
  --with-ardupilot           Enable ArduPilot setup (default: off)

  # PX4 controls
  --px4-ref REF           PX4 ref/tag/branch (default: ${PX4_REF})
  --px4-dir PATH          PX4 directory (default: <project-root>/${PX4_DIR_REL})

  # PX4 ROS messages controls
  --no-px4-msgs           Disable px4_msgs setup (default: off)
  --px4-msgs-ref REF      px4_msgs ref/tag/branch (default: ${PX4_MSGS_REF})
  --px4-msgs-dir PATH     px4_msgs directory (default: <project-root>/${PX4_MSGS_DIR_REL})

  # Ardupilot controls
  --ardupilot-ref REF     ArduPilot ref/tag/branch (default: ${ARDUPILOT_REF})
  --ardupilot-dir PATH    ArduPilot dir (default: <project-root>/${ARDUPILOT_DIR_REL})

  # Other controls
  --no-submodules         Skip 'git submodule update --init --recursive'

Examples:
  # Default (PX4 v1.16.1 + px4_msgs)
  bash get_src.sh

  # Add ArduPilot (PX4 v1.16.1 + AP Copter-4.6.2 + px4_msgs)
  bash get_src.sh --with-ardupilot

  # Pin PX4 or ArduPilot
  bash get_src.sh --px4-ref v1.16.0 --ardupilot-ref Copter-4.6.1
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

# Prevent sourcing
(return 0 2>/dev/null) && { echo "Do not source this script. Run: bash $0" >&2; return 1; }

# --------------------------
# Parse args
# --------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) help; exit 0 ;;
    --debug) DEBUG="true"; shift ;;
    --project-root) [[ $# -ge 2 ]] || die "--project-root requires a path"; PROJECT_ROOT="$2"; shift 2 ;;

    # Select autopilots
    --no-px4) WITH_PX4="false"; shift ;;
    --with-ardupilot) WITH_ARDUPILOT="true"; shift ;;

    # PX4 controls
    --px4-ref) [[ $# -ge 2 ]] || die "--px4-ref requires a value"; PX4_REF="$2"; shift 2 ;;
    --px4-dir) [[ $# -ge 2 ]] || die "--px4-dir requires a path"; PX4_DIR_REL=""; PX4_DIR_ABS="$2"; shift 2 ;;

    # px4_msgs controls
    --no-px4-msgs) WITH_PX4_MSGS="false"; shift ;;
    --px4-msgs-ref) [[ $# -ge 2 ]] || die "--px4-msgs-ref requires a value"; PX4_MSGS_REF="$2"; shift 2 ;;
    --px4-msgs-dir) [[ $# -ge 2 ]] || die "--px4-msgs-dir requires a path"; PX4_MSGS_DIR_REL=""; PX4_MSGS_DIR_ABS="$2"; shift 2 ;;

    # ArduPilot controls
    --ardupilot-ref) [[ $# -ge 2 ]] || die "--ardupilot-ref requires a value"; ARDUPILOT_REF="$2"; shift 2 ;;
    --ardupilot-dir) [[ $# -ge 2 ]] || die "--ardupilot-dir requires a path"; ARDUPILOT_DIR_REL=""; ARDUPILOT_DIR_ABS="$2"; shift 2 ;;

    # Other controls
    --no-submodules) UPDATE_SUBMODULES="false"; shift ;;

    *) die "Unknown option: $1 (use --help)" ;;
  esac
done

# Compute default absolute paths
PX4_DIR="${PX4_DIR_ABS:-${PROJECT_ROOT}/${PX4_DIR_REL}}"
PX4_MSGS_DIR="${PX4_MSGS_DIR_ABS:-${PROJECT_ROOT}/${PX4_MSGS_DIR_REL}}"
ARDUPILOT_DIR="${ARDUPILOT_DIR_ABS:-${PROJECT_ROOT}/${ARDUPILOT_DIR_REL}}"

# --------------------------
# Strict mode
# --------------------------
set -Ee
trap 'echo "[get_src.sh] ERROR line=$LINENO cmd=$BASH_COMMAND" >&2' ERR
[[ "${DEBUG}" == "true" ]] && set -x

# --------------------------
# Helpers
# --------------------------
is_dir_empty() {
  local d="$1"
  [[ ! -d "$d" ]] && return 0
  [[ -z "$(ls -A "$d" 2>/dev/null)" ]]
}

align_submodules() {
  local repo="$1"
  [[ "${UPDATE_SUBMODULES}" == "true" ]] || { echo "Skipping submodules (--no-submodules): ${repo}"; return 0; }
  [[ -d "${repo}/.git" ]] || return 0
  echo "Aligning submodules: ${repo}"
  ( cd "${repo}" && git submodule update --init --recursive )
}

# --------------------------
# Ensure base directories
# --------------------------
mkdir -p "${PROJECT_ROOT}/ap" "${PROJECT_ROOT}/ros2"
mkdir -p "$(dirname "${PX4_DIR}")" "$(dirname "${ARDUPILOT_DIR}")" "$(dirname "${PX4_MSGS_DIR}")"

# --------------------------
# Clone PX4 (with --recursive)
# --------------------------
if [[ "${WITH_PX4}" == "true" ]]; then
  if is_dir_empty "${PX4_DIR}"; then
    echo "Cloning PX4-Autopilot (${PX4_REF}) into: ${PX4_DIR}"
    git clone -b "${PX4_REF}" --recursive "https://github.com/mhcho1994/PX4-Autopilot.git" "${PX4_DIR}"
    # align submodules again just in case (some repos have had issues with --recursive)
    align_submodules "${PX4_DIR}"
  else
    echo "PX4 directory exists and is not empty, skipping: ${PX4_DIR}"
  fi
else
  echo "PX4 disabled (--no-px4)."
fi

# --------------------------
# Clone px4_msgs (plain clone)
# --------------------------
if [[ "${WITH_PX4_MSGS}" == "true" ]]; then
  if is_dir_empty "${PX4_MSGS_DIR}"; then
    echo "Cloning px4_msgs (${PX4_MSGS_REF}) into: ${PX4_MSGS_DIR}"
    git clone -b "${PX4_MSGS_REF}" "https://github.com/PX4/px4_msgs.git" "${PX4_MSGS_DIR}"
  else
    echo "px4_msgs directory exists and is not empty, skipping: ${PX4_MSGS_DIR}"
  fi
else
  echo "px4_msgs disabled (--no-px4-msgs)."
fi

# --------------------------
# Clone ArduPilot (with --recurse-submodules)
# --------------------------
if [[ "${WITH_ARDUPILOT}" == "true" ]]; then
  if is_dir_empty "${ARDUPILOT_DIR}"; then
    echo "Cloning ArduPilot (${ARDUPILOT_REF}) into: ${ARDUPILOT_DIR}"
    git clone -b "${ARDUPILOT_REF}" --recurse-submodules "https://github.com/ArduPilot/ardupilot.git" "${ARDUPILOT_DIR}"
    # align submodules again just in case (some repos have had issues with --recursive)
    align_submodules "${ARDUPILOT_DIR}"
  else
    echo "ArduPilot directory exists and is not empty, skipping: ${ARDUPILOT_DIR}"
  fi
else
  echo "ArduPilot disabled by default. Use --with-ardupilot to enable."
fi

echo ""
echo "Source clone DONE. Sources prepared under: ${PROJECT_ROOT}"
echo "  PX4:        ${WITH_PX4} -> ${PX4_DIR}"
echo "  px4_msgs:   ${WITH_PX4_MSGS} -> ${PX4_MSGS_DIR}"
echo "  ArduPilot:  ${WITH_ARDUPILOT} -> ${ARDUPILOT_DIR}"
echo "  Submodules: ${UPDATE_SUBMODULES}"
echo ""