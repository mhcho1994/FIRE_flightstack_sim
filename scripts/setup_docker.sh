#!/usr/bin/env bash
# Manage the development image, container, and independent Bash sessions.
# Run in a child Bash even when sourced: never change the caller's options,
# variables, functions, working directory, or traps.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    if bash "${BASH_SOURCE[0]}" "$@"; then
        return 0
    else
        return "$?"
    fi
fi
set -euo pipefail

THIS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${THIS_DIR}/setup_docker.sh"
PROJECT_ROOT="$(cd -- "${THIS_DIR}/.." && pwd)"
DOCKERFILE="${PROJECT_ROOT}/docker/Dockerfile"
WORKSPACE="/home/user/FIRE_flightstack_sim"
READY_FILE="/tmp/fire_flightstack_sim.ready"

HOST_UID="$(id -u)"
HOST_GID="$(id -g)"
HOST_USER_NAME="$(id -un)"
HOST_GROUP_NAME="$(id -gn)"
IMAGE_NAME="${IMAGE_NAME:-fire_flightstack_sim}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
CONTAINER_NAME="${CONTAINER_NAME:-fire_flightstack_sim}"

DO_FETCH=1
FETCH_ONLY=0
NO_CACHE=0
RECREATE=0
NEW_TERMINAL=0
USE_GPU=0
USE_X11=auto
USE_HOST_NET=1
COMMAND="${1:-help}"

log() { echo "[INFO] $*"; }
warn() { echo "[WARN] $*" >&2; }
die() { echo "[ERROR] $*" >&2; exit 1; }
require_cmd() { command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"; }

usage() {
    cat <<EOF_HELP
Usage: source scripts/setup_docker.sh COMMAND [OPTIONS]
       bash scripts/setup_docker.sh COMMAND [OPTIONS]

Commands:
  build     Fetch sources and build the image using Docker's build cache.
  run       Start in the background; reuse an existing container.
  open      Open an independent Bash session in the running container.
  stop      Stop the container without removing it.
  remove    Remove the container (including a running one), keeping the image
            and bind-mounted project files. Container-only files are deleted.
  help      Show this help; also the default when no command is given.

Build options:
  --no-fetch       Skip host-side source/artifact fetch steps.
  --fetch-only     Fetch sources/artifacts without building an image.
  --no-cache       Build without the Docker layer cache.

Run options (creation settings require --recreate for an existing container):
  --recreate       Replace the container using the current image and rerun setup.
                  Container-only files are deleted; project files are retained.
  --gpu            Enable NVIDIA GPU access.
  --x11            Enable X11 forwarding (default: enabled when DISPLAY is set).
  --no-x11         Disable X11 forwarding.
  --no-host-net    Use Docker's default networking instead of host networking.

Open options:
  --new-terminal   Open in a new Linux terminal or Windows Terminal tab (WSL).

Common options:
  --image NAME     Image name (default: ${IMAGE_NAME}).
  --tag TAG        Image tag (default: ${IMAGE_TAG}).
  --container NAME Container name (default: ${CONTAINER_NAME}).
  --debug          Trace script commands.
  -h, --help       Show help.

Examples:
  source scripts/setup_docker.sh build
  source scripts/setup_docker.sh run
  source scripts/setup_docker.sh open
  source scripts/setup_docker.sh open --new-terminal
  source scripts/setup_docker.sh stop
  source scripts/setup_docker.sh remove

After changing the Dockerfile or image dependencies:
  source scripts/setup_docker.sh build --no-fetch
  source scripts/setup_docker.sh run --recreate

Container build outputs are stored separately under .docker_home/artifacts/NAME.
Host build outputs are preserved; existing containers need run --recreate to use this.

First-time setup (including after recreation) runs in the background and may
take several minutes or longer. Wait until it completes before using open.
Follow progress with:
  docker logs -f ${CONTAINER_NAME}
EOF_HELP
}

parse_args() {
    case "${COMMAND}" in
        build|run|open|stop|remove) ;;
        help|-h|--help) usage; exit 0 ;;
        *) die "Unknown command: ${COMMAND}. Use 'help' for usage." ;;
    esac
    if (( $# > 0 )); then shift; fi
    while (( $# > 0 )); do
        case "$1" in
            --image|--tag|--container)
                [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || die "$1 requires a value"
                case "$1" in
                    --image) IMAGE_NAME="$2" ;;
                    --tag) IMAGE_TAG="$2" ;;
                    --container) CONTAINER_NAME="$2" ;;
                esac
                shift 2
                ;;
            --no-fetch|--fetch-only|--no-cache)
                [[ "${COMMAND}" == build ]] || die "$1 is only valid with build"
                case "$1" in
                    --no-fetch) DO_FETCH=0 ;;
                    --fetch-only) FETCH_ONLY=1 ;;
                    --no-cache) NO_CACHE=1 ;;
                esac
                shift
                ;;
            --recreate|--gpu|--x11|--no-x11|--no-host-net)
                [[ "${COMMAND}" == run ]] || die "$1 is only valid with run"
                case "$1" in
                    --recreate) RECREATE=1 ;;
                    --gpu) USE_GPU=1 ;;
                    --x11) USE_X11=1 ;;
                    --no-x11) USE_X11=0 ;;
                    --no-host-net) USE_HOST_NET=0 ;;
                esac
                shift
                ;;
            --new-terminal)
                [[ "${COMMAND}" == open ]] || die "$1 is only valid with open"
                NEW_TERMINAL=1
                shift
                ;;
            --debug) set -x; shift ;;
            -h|--help) usage; exit 0 ;;
            *) die "Unknown option: $1. Use 'help' for usage." ;;
        esac
    done
    [[ "${CONTAINER_NAME}" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] \
        || die "Invalid container name: ${CONTAINER_NAME}"
    if (( FETCH_ONLY && ! DO_FETCH )); then
        die "--fetch-only and --no-fetch cannot be combined"
    fi
}

require_docker() {
    require_cmd docker
    docker info >/dev/null 2>&1 || die "Docker daemon is unavailable or permission was denied."
}

container_exists() {
    docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1
}

container_status() {
    docker container inspect --format '{{.State.Status}}' "${CONTAINER_NAME}"
}

prepare_host_dirs() {
    local directory
    for directory in ap data gz ros2 tools ws .docker_home; do
        mkdir -p "${PROJECT_ROOT}/${directory}"
    done
}

# CMake/colcon outputs contain absolute paths and cannot be shared with a
# native build at a different project path. Overlay only generated directories,
# leaving source files shared and the host's existing build outputs untouched.
prepare_build_mounts() {
    local artifact_root="${PROJECT_ROOT}/.docker_home/artifacts/${CONTAINER_NAME}"
    local relative workspace output
    local -a paths=(build gz/env)
    for relative in ap/px4 ap/ardupilot gz/ardupilot_gazebo tools/Micro-XRCE-DDS-Agent; do
        if [[ -d "${PROJECT_ROOT}/${relative}" ]]; then paths+=("${relative}/build"); fi
    done
    for workspace in "${PROJECT_ROOT}"/ros2/* "${PROJECT_ROOT}"/gz/*; do
        [[ -d "${workspace}/src" ]] || continue
        relative="${workspace#"${PROJECT_ROOT}/"}"
        for output in build install log; do paths+=("${relative}/${output}"); done
    done
    BUILD_MOUNTS=()
    for relative in "${paths[@]}"; do
        mkdir -p "${PROJECT_ROOT}/${relative}" "${artifact_root}/${relative}"
        BUILD_MOUNTS+=(--volume "${artifact_root}/${relative}:${WORKSPACE}/${relative}:rw")
    done
}

container_ready() {
    docker exec --user user "${CONTAINER_NAME}" test -f "${READY_FILE}" >/dev/null 2>&1
}

recent_startup_logs() {
    local started_at
    started_at="$(docker container inspect --format '{{.State.StartedAt}}' "${CONTAINER_NAME}")" || return
    # A reused container retains previous runs' logs. Only inspect this start.
    docker logs --since "${started_at}" --tail 80 "${CONTAINER_NAME}" 2>&1
}

print_setup_progress() {
    log "Container initialization is still in progress."
    log "First-time setup or setup after recreation may take several minutes or longer."
    log "Follow progress: docker logs -f ${CONTAINER_NAME}"
    log "Wait for setup to finish, then retry open (first-time setup: '[ENTRYPOINT] Setup completed.')."
}

require_running_container() {
    local status exit_code
    status="$(container_status)"
    [[ "${status}" != running ]] || return 0
    if [[ "${status}" == exited || "${status}" == dead ]]; then
        exit_code="$(docker container inspect --format '{{.State.ExitCode}}' "${CONTAINER_NAME}")"
        if [[ "${exit_code}" != 0 ]]; then
            warn "Container ${CONTAINER_NAME} exited with code ${exit_code}. Recent logs:"
            recent_startup_logs >&2 || true
            die "Container startup failed. Full log: docker logs ${CONTAINER_NAME}"
        fi
    fi
    die "Container is ${status}. Run 'run' first."
}

build_image() {
    prepare_host_dirs
    if (( DO_FETCH )); then
        log "Fetching host-side sources and artifacts"
        bash "${PROJECT_ROOT}/install/autopilot.sh" \
            --phase fetch --with-ardupilot --project-root "${PROJECT_ROOT}"
        bash "${PROJECT_ROOT}/install/extra.sh" \
            --phase fetch --project-root "${PROJECT_ROOT}"
    fi
    if (( FETCH_ONLY )); then return 0; fi

    local -a args=(docker build)
    if (( NO_CACHE )); then args+=(--no-cache); fi
    log "Building ${IMAGE_NAME}:${IMAGE_TAG}"
    "${args[@]}" \
        --build-arg "UID_USER=${HOST_UID}" \
        --build-arg "GID_USER=${HOST_GID}" \
        --build-arg "GID_INPUT=${GID_INPUT:-107}" \
        --build-arg "GID_RENDER=${GID_RENDER:-110}" \
        --build-arg "HOST_USER_NAME=${HOST_USER_NAME}" \
        --build-arg "HOST_USER_ID=${HOST_UID}" \
        --build-arg "HOST_GROUP_NAME=${HOST_GROUP_NAME}" \
        --build-arg "HOST_GROUP_ID=${HOST_GID}" \
        --tag "${IMAGE_NAME}:${IMAGE_TAG}" --file "${DOCKERFILE}" "${PROJECT_ROOT}"
    log "Image built. Use 'run --recreate' to apply it to an existing container."
}

allow_x11() {
    # Keep access after this script returns: detached GUI clients connect later.
    # Limit the fallback grant to the host user mapped into the container.
    if [[ -n "${DISPLAY:-}" ]] && command -v xhost >/dev/null 2>&1; then
        xhost "+si:localuser:${HOST_USER_NAME}" >/dev/null 2>&1 \
            || warn "Could not enable X11 access on ${DISPLAY}."
    fi
}

run_container() {
    local status existing_display
    if container_exists && (( ! RECREATE )); then
        status="$(container_status)"
        case "${status}" in
            running|exited|created) ;;
            *) die "Container is ${status}; resolve that state before running it." ;;
        esac
        existing_display="$(docker container inspect --format \
            '{{range .Config.Env}}{{println .}}{{end}}' "${CONTAINER_NAME}" \
            | sed -n 's/^DISPLAY=//p')"
        if [[ -n "${existing_display}" ]]; then
            if [[ "${existing_display}" != "${DISPLAY:-}" ]]; then
                warn "Container DISPLAY=${existing_display} differs from the host; use run --recreate to update it."
            else
                allow_x11
            fi
        fi
        if [[ "${status}" == running ]]; then
            log "Container is already running: ${CONTAINER_NAME}"
        else
            docker container start "${CONTAINER_NAME}" >/dev/null
            log "Startup requested for existing container: ${CONTAINER_NAME}"
        fi
        log "Creation options are unchanged. Use 'run --recreate' to change them."
    else
        docker image inspect "${IMAGE_NAME}:${IMAGE_TAG}" >/dev/null 2>&1 \
            || die "Image ${IMAGE_NAME}:${IMAGE_TAG} does not exist. Run 'build' first."
        prepare_host_dirs
        if [[ "${USE_X11}" == auto ]]; then
            USE_X11=0
            if [[ -n "${DISPLAY:-}" ]]; then USE_X11=1; fi
        fi
        if (( USE_X11 )); then
            [[ -n "${DISPLAY:-}" ]] || die "DISPLAY must be set for --x11"
        fi

        prepare_build_mounts
        local -a args=(docker run --detach --init --name "${CONTAINER_NAME}"
            --ipc host --shm-size 4g
            --env "HOST_UID=${HOST_UID}" --env "HOST_GID=${HOST_GID}"
            --env "HOST_USER_NAME=${HOST_USER_NAME}" --env "HOST_GROUP_NAME=${HOST_GROUP_NAME}"
            --env "TZ=${TZ:-America/New_York}" --env XDG_RUNTIME_DIR=/tmp/runtime-docker
            --volume "${PROJECT_ROOT}:${WORKSPACE}:rw"
            --volume "${PROJECT_ROOT}/.docker_home:/home/user/.host_persist"
            --workdir "${WORKSPACE}" "${BUILD_MOUNTS[@]}")
        if (( USE_HOST_NET )); then args+=(--network host); fi
        if (( USE_GPU )); then
            args+=(--gpus all --env NVIDIA_VISIBLE_DEVICES=all --env NVIDIA_DRIVER_CAPABILITIES=all)
        fi
        local device
        for device in /dev/dri /dev/ttyUSB0 /dev/ttyACM0; do
            if [[ -e "${device}" ]]; then args+=(--device "${device}"); fi
        done
        if [[ -d /dev/input ]]; then args+=(--volume /dev/input:/dev/input:ro); fi
        if (( USE_X11 )); then
            args+=(--env "DISPLAY=${DISPLAY}" --env QT_X11_NO_MITSHM=1
                --volume /tmp/.X11-unix:/tmp/.X11-unix:ro)
            if [[ -n "${XAUTHORITY:-}" && -f "${XAUTHORITY}" ]]; then
                args+=(--env XAUTHORITY=/tmp/.docker.xauth --volume "${XAUTHORITY}:/tmp/.docker.xauth:ro")
            else
                allow_x11
            fi
        fi
        if container_exists; then
            log "Replacing container: ${CONTAINER_NAME}"
            docker container rm --force "${CONTAINER_NAME}" >/dev/null
        fi
        # A new container has a new user home and needs its environment set up.
        rm -f "${PROJECT_ROOT}/.docker_home/.setup_done"
        "${args[@]}" "${IMAGE_NAME}:${IMAGE_TAG}" -- sleep infinity >/dev/null
        log "Startup requested for new container: ${CONTAINER_NAME}"
    fi
    require_running_container
    if container_ready; then
        log "Container initialization is complete. Ready to open a shell."
    else
        require_running_container
        log "Setup runs in the background; this command returns without waiting."
        print_setup_progress
    fi
    log "Open a shell: bash \"${SCRIPT_PATH}\" open --container \"${CONTAINER_NAME}\""
}

open_new_terminal() {
    local -a shell_command=(bash "${SCRIPT_PATH}" open --container "${CONTAINER_NAME}")
    local -a terminal_command
    if [[ -n "${WSL_DISTRO_NAME:-}" ]] && command -v wt.exe >/dev/null 2>&1; then
        terminal_command=(wt.exe -w 0 new-tab wsl.exe --distribution "${WSL_DISTRO_NAME}" --exec "${shell_command[@]}")
    elif [[ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]]; then
        if command -v x-terminal-emulator >/dev/null 2>&1; then
            terminal_command=(x-terminal-emulator -e "${shell_command[@]}")
        elif command -v gnome-terminal >/dev/null 2>&1; then
            terminal_command=(gnome-terminal -- "${shell_command[@]}")
        elif command -v konsole >/dev/null 2>&1; then
            terminal_command=(konsole -e "${shell_command[@]}")
        elif command -v xterm >/dev/null 2>&1; then
            terminal_command=(xterm -e "${shell_command[@]}")
        else
            die "No supported terminal found. Use 'open' in another terminal."
        fi
    else
        die "No graphical terminal session found. Use 'open' in another terminal."
    fi
    local terminal_log terminal_pid
    terminal_log="$(mktemp "${TMPDIR:-/tmp}/fire-docker-terminal.XXXXXX.log")"
    nohup "${terminal_command[@]}" >"${terminal_log}" 2>&1 < /dev/null &
    terminal_pid=$!
    sleep 0.2
    if ! kill -0 "${terminal_pid}" 2>/dev/null; then
        wait "${terminal_pid}" || die "Terminal launch failed. See ${terminal_log}"
    fi
    log "Opened a new terminal. Launcher output: ${terminal_log}"
}

open_container() {
    container_exists || die "Container does not exist. Run 'run' first."
    require_running_container
    # This per-start marker is written by entrypoint only after setup succeeds.
    if ! container_ready; then
        local setup_logs
        if ! setup_logs="$(recent_startup_logs)"; then
            printf '%s\n' "${setup_logs}" >&2
            die "Could not read initialization logs. Check 'docker logs ${CONTAINER_NAME}' and retry open."
        fi
        # Initialization may finish or the container may exit while logs are read.
        require_running_container
        if ! container_ready; then
            # Match only the fatal entrypoint trap. Dependency tools can print
            # nonfatal ERROR messages, which are not proof of failed setup.
            if grep -Eq '^\[entrypoint\.sh\] ERROR( |$)' <<< "${setup_logs}"; then
                printf '%s\n' "${setup_logs}" >&2
                die "Container initialization failed. Full log: docker logs ${CONTAINER_NAME}"
            fi
            print_setup_progress
            return 1
        fi
    fi
    if (( NEW_TERMINAL )); then
        open_new_terminal
    else
        exec docker exec --interactive --tty --user user --workdir "${WORKSPACE}" \
            "${CONTAINER_NAME}" /bin/bash
    fi
}

stop_container() {
    if ! container_exists; then
        log "Container does not exist: ${CONTAINER_NAME}"
    else
        case "$(container_status)" in
            exited|created) log "Container is already stopped: ${CONTAINER_NAME}" ;;
            *) docker container stop "${CONTAINER_NAME}" >/dev/null
               log "Stopped container: ${CONTAINER_NAME}" ;;
        esac
    fi
}

remove_container() {
    if container_exists; then
        docker container rm --force "${CONTAINER_NAME}" >/dev/null
        log "Removed container: ${CONTAINER_NAME}. Image and project files were kept."
    else
        log "Container does not exist: ${CONTAINER_NAME}"
    fi
}

parse_args "$@"
if [[ "${COMMAND}" != build ]] || (( ! FETCH_ONLY )); then require_docker; fi
case "${COMMAND}" in
    build) build_image ;;
    run) run_container ;;
    open) open_container ;;
    stop) stop_container ;;
    remove) remove_container ;;
esac
