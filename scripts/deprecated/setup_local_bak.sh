#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# setup_local.sh
# Local environment setup script.
# This script has only been tested on Windows Subsystem for Linux (WSL) with Ubuntu 22.04. 
# Successful installation on newer Ubuntu versions is not guaranteed.
# -----------------------------------------------------------------------------

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$THIS_DIR/.." && pwd)"

HOST_UID="$(id -u)"
HOST_GID="$(id -g)"
GID_INPUT="${GID_INPUT:-107}"
GID_RENDER="${GID_RENDER:-110}"
CURRENT_USER="${USER:-$(id -un)}"

log() {
    echo "[INFO] $*"
}

warn() {
    echo "[WARN] $*" >&2
}

die() {
    echo "[ERROR] $*" >&2
    exit 1
}

require_file() {
    local file="$1"
    [[ -f "$file" ]] || die "Required file not found: $file"
}

run_script() {
    local script_path="$1"
    shift
    require_file "$script_path"

    log "Preparing script: $script_path"
    sudo chown "$CURRENT_USER:$CURRENT_USER" "$script_path"
    sudo chmod +x "$script_path"

    log "Running script: $script_path $*"
    bash "$script_path" "$@"
}

ensure_group() {
    local group_name="$1"
    local group_gid="$2"

    if getent group "$group_name" >/dev/null 2>&1; then
        log "Group '$group_name' already exists. Skipping."
    else
        log "Creating group '$group_name' with GID $group_gid"
        sudo groupadd -r -g "$group_gid" "$group_name"
    fi
}

install_locales() {
    log "Updating package index and upgrading packages"
    sudo apt-get -y update
    sudo apt-get -y upgrade

    log "Installing base locale package"
    sudo apt-get -y --quiet --no-install-recommends install locales

    log "Generating locale en_US.UTF-8"
    sudo locale-gen en_US en_US.UTF-8
    sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
    export LANG=en_US.UTF-8
}

setup_groups_and_user() {
    log "Host UID: $HOST_UID"
    log "Host GID: $HOST_GID"

    ensure_group "input" "$GID_INPUT"
    ensure_group "render" "$GID_RENDER"

    log "Adding user '$CURRENT_USER' to required groups"
    sudo usermod -a -G sudo,plugdev,dialout,input,render,video "$CURRENT_USER"
}

main() {
    log "Starting local setup from: $PROJECT_ROOT"

    cd "$PROJECT_ROOT"

    install_locales
    setup_groups_and_user

    # Install required dependencies and packages
    run_script "$PROJECT_ROOT/install/base.sh"
    run_script "$PROJECT_ROOT/install/ros2.sh" --ros-distro humble
    run_script "$PROJECT_ROOT/install/gazebo.sh" --install binary --phase all
    run_script "$PROJECT_ROOT/install/autopilot.sh" --with-ardupilot --phase all
    run_script "$PROJECT_ROOT/install/extra.sh" --phase all

    # Optional cleanup
    run_script "$PROJECT_ROOT/install/clean.sh"

    # User setup
    run_script "$PROJECT_ROOT/install/usersetup.sh"

    log "Local setup completed successfully."
    warn "You may need to log out and log back in for new group memberships to take effect."
}

main "$@"