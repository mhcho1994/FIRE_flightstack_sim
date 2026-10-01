#!/usr/bin/env bash
# Local setup for Ubuntu 22.04 / ROS 2 Humble / Gazebo Harmonic (including WSL).
# Sourcing dispatches into a child Bash so options, traps, variables, and the
# working directory of the caller are preserved, including on failure.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    if bash "${BASH_SOURCE[0]}" "$@"; then
        return 0
    else
        return "$?"
    fi
fi
set -euo pipefail

THIS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${THIS_DIR}/.." && pwd)"
CURRENT_USER="$(id -un)"
COMMAND="${1:-help}"
DO_FETCH=1
DO_UPGRADE=0
DEBUG=0
GROUPS_CHANGED=0

log() { echo "[INFO] $*"; }
warn() { echo "[WARN] $*" >&2; }
die() { echo "[ERROR] $*" >&2; exit 1; }

usage() {
    cat <<'EOF_HELP'
Usage: source scripts/setup_local.sh COMMAND [OPTIONS]
       bash scripts/setup_local.sh COMMAND [OPTIONS]

Local setup for Ubuntu 22.04, ROS 2 Humble, and binary Gazebo Harmonic.
PX4 and ArduPilot support components are both enabled.

Commands:
  install   Run deps -> fetch -> build -> env for an initial installation.
  deps      Configure locale/device groups and install system dependencies,
            ROS 2, Gazebo, and supporting tools.
  fetch     Check existing source trees, fetch auxiliary repositories, and
            download QGroundControl. Main source submodules must already exist.
  build     Incrementally build DDS Agent, px4_msgs, and ArduPilot/FIRE Gazebo
            plugins from prepared sources. Does not build autopilot firmware.
            The underlying build script retains its rosdep dependency checks.
  env       Generate environment files, configure QGroundControl access, and
            update the project's managed block in ~/.bashrc.
  help      Show this help; also the default when no command is given.

Options:
  --no-fetch  Skip source/artifact preparation (install only).
  --upgrade   Run apt-get upgrade before dependency installation (install/deps).
  --debug     Trace this script and its installation helpers.
  -h, --help  Show help.

Group overrides (only when creating a missing device group):
  GID_INPUT   Explicit GID for the input group; otherwise allocated by the OS.
  GID_RENDER  Explicit GID for the render group; otherwise allocated by the OS.

Examples:
  source scripts/setup_local.sh install
  source scripts/setup_local.sh install --no-fetch
  source scripts/setup_local.sh deps
  source scripts/setup_local.sh fetch
  source scripts/setup_local.sh build
  source scripts/setup_local.sh env

Commands may invoke sudo through their helpers. System-wide upgrades and
package cleanup are not automatic. env writes configuration files; to apply
shell changes to the current interactive terminal afterwards, run:
  source ~/.bashrc
New device-group memberships require logging out and back in.
EOF_HELP
}

parse_args() {
    case "${COMMAND}" in
        install|deps|fetch|build|env) ;;
        help|-h|--help) usage; exit 0 ;;
        *) die "Unknown command: ${COMMAND}. Use 'help' for usage." ;;
    esac
    if (( $# > 0 )); then shift; fi
    while (( $# > 0 )); do
        case "$1" in
            --no-fetch)
                [[ "${COMMAND}" == install ]] || die "--no-fetch is only valid with install"
                DO_FETCH=0
                ;;
            --upgrade)
                [[ "${COMMAND}" == install || "${COMMAND}" == deps ]] \
                    || die "--upgrade is only valid with install or deps"
                DO_UPGRADE=1
                ;;
            --debug) DEBUG=1 ;;
            -h|--help) usage; exit 0 ;;
            *) die "Unknown option: $1. Use 'help' for usage." ;;
        esac
        shift
    done
}

run_script() {
    local name="$1"
    shift
    local script_path="${PROJECT_ROOT}/install/${name}.sh"
    [[ -f "${script_path}" ]] || die "Required file not found: ${script_path}"
    local -a debug_args=()
    if (( DEBUG )); then debug_args+=(--debug); fi
    log "Running ${name}.sh $*"
    bash "${script_path}" "${debug_args[@]}" "$@"
}

# Preflight all helpers for the requested operation before changing the host.
check_helpers() {
    local -a names
    local name
    case "${COMMAND}" in
        install) names=(base ros2 gazebo autopilot extra usersetup) ;;
        deps) names=(base ros2 gazebo autopilot extra) ;;
        fetch) names=(autopilot extra) ;;
        build) names=(autopilot) ;;
        env) names=(gazebo autopilot extra usersetup) ;;
    esac
    for name in "${names[@]}"; do
        [[ -f "${PROJECT_ROOT}/install/${name}.sh" ]] \
            || die "Required file not found: ${PROJECT_ROOT}/install/${name}.sh"
    done
}

ensure_group() {
    local name="$1" requested_gid="$2"
    if getent group "${name}" >/dev/null; then
        log "Using existing group: ${name}"
    elif [[ -n "${requested_gid}" ]]; then
        sudo groupadd --system --gid "${requested_gid}" "${name}"
    else
        sudo groupadd --system "${name}"
    fi
}

setup_groups_and_user() {
    ensure_group input "${GID_INPUT:-}"
    ensure_group render "${GID_RENDER:-}"
    local memberships group
    local -a missing=()
    memberships=" $(id -nG "${CURRENT_USER}") "
    for group in sudo plugdev dialout input render video; do
        if ! getent group "${group}" >/dev/null; then
            warn "Group ${group} is unavailable; skipping it."
        elif [[ "${memberships}" != *" ${group} "* ]]; then
            missing+=("${group}")
        fi
    done
    if (( ${#missing[@]} )); then
        local IFS=,
        sudo usermod -a -G "${missing[*]}" "${CURRENT_USER}"
        GROUPS_CHANGED=1
    fi
}

install_dependencies() {
    command -v sudo >/dev/null || die "Required command not found: sudo"
    local name requested_gid
    for name in input render; do
        if [[ "${name}" == input ]]; then requested_gid="${GID_INPUT:-}"; else requested_gid="${GID_RENDER:-}"; fi
        if ! getent group "${name}" >/dev/null && [[ -n "${requested_gid}" ]]; then
            [[ "${requested_gid}" =~ ^[0-9]+$ ]] || die "Invalid GID for ${name}: ${requested_gid}"
            if getent group "${requested_gid}" >/dev/null; then
                die "GID ${requested_gid} is already assigned; unset the override for ${name}."
            fi
        fi
    done

    sudo apt-get -y update
    if (( DO_UPGRADE )); then sudo apt-get -y upgrade; fi
    sudo apt-get -y --quiet --no-install-recommends install locales
    sudo locale-gen en_US en_US.UTF-8
    sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
    export LANG=en_US.UTF-8
    setup_groups_and_user

    run_script base
    run_script ros2 --ros-distro humble
    run_script gazebo --install binary --ros-gz binary --phase deps --project-root "${PROJECT_ROOT}"
    # In gazebo.sh's binary mode, "build" installs the Gazebo/ROS-GZ packages.
    # Complete this before building autopilot plugins against those packages.
    run_script gazebo --install binary --ros-gz binary --phase build --project-root "${PROJECT_ROOT}"
    run_script autopilot --phase deps --with-ardupilot --project-root "${PROJECT_ROOT}"
    run_script extra --phase deps --project-root "${PROJECT_ROOT}"
}

fetch_sources() {
    run_script autopilot --phase fetch --with-ardupilot --project-root "${PROJECT_ROOT}"
    run_script extra --phase fetch --project-root "${PROJECT_ROOT}"
}

build_components() {
    run_script autopilot --phase build --with-ardupilot --project-root "${PROJECT_ROOT}"
}

configure_environment() {
    run_script gazebo --install binary --ros-gz binary --phase env --project-root "${PROJECT_ROOT}"
    run_script autopilot --phase env --with-ardupilot --project-root "${PROJECT_ROOT}"
    run_script extra --phase env --project-root "${PROJECT_ROOT}"
    run_script usersetup --ros-distro humble --project-root "${PROJECT_ROOT}"
    log 'Shell configuration updated. Apply it in your terminal with: source ~/.bashrc'
}

parse_args "$@"
if (( DEBUG )); then set -x; fi
check_helpers
cd -- "${PROJECT_ROOT}"
log "Local ${COMMAND}: ${PROJECT_ROOT}"
case "${COMMAND}" in
    install)
        install_dependencies
        if (( DO_FETCH )); then fetch_sources; fi
        build_components
        configure_environment
        ;;
    deps) install_dependencies ;;
    fetch) fetch_sources ;;
    build) build_components ;;
    env) configure_environment ;;
esac
log "Local ${COMMAND} completed."
if (( GROUPS_CHANGED )); then
    warn "Log out and back in to apply new device-group memberships."
fi
