#!/usr/bin/env bash
# gazebo.sh
#
# Install or build Gazebo Harmonic and optional ros_gz for Ubuntu 22.04 / ROS 2.
#
# Supported workflows
# -------------------
#
# 1) Binary installation
#    - fetch : optional source fetch for ros_gz when --ros-gz source
#    - deps  : install common dependencies and apt repositories
#    - env   : write environment helper script
#    - build : install Gazebo binary and either install/build ros_gz
#
# 2) Source installation
#    - fetch : fetch source trees on the host before docker build/runtime
#    - deps  : install build dependencies inside Dockerfile/container
#    - env   : write environment helper script
#    - build : build source trees inside the container
#
# Recommended usage
# -----------------
#
# Binary mode:
#   bash install/gazebo.sh --install binary --phase all
#
# Binary Gazebo + source ros_gz:
#   bash install/gazebo.sh --install binary --ros-gz source --phase fetch
#   bash install/gazebo.sh --install binary --ros-gz source --phase deps
#   bash install/gazebo.sh --install binary --ros-gz source --phase env
#   bash install/gazebo.sh --install binary --ros-gz source --phase build
#
# Source mode:
#   # On host:
#   bash install/gazebo.sh --install source --phase fetch
#
#   # In Dockerfile:
#   bash install/gazebo.sh --install source --phase deps
#   bash install/gazebo.sh --install source --phase env
#
#   # Inside container:
#   bash install/gazebo.sh --install source --phase build
#
# Notes
# -----
# - We intentionally do NOT run apt-get upgrade for reproducibility.
# - ROS 2 Humble + Gazebo Harmonic is a non-default pairing and should be used carefully.

# ------------------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------------------
DEBUG="false"
DO_UPGRADE="false"

INSTALL_MODE="binary"   # binary | source
PHASE="all"             # fetch | deps | env | build | all

BUILD_TYPE="RelWithDebInfo"
ROS_DISTRO="humble"
GZ_VERSION="harmonic"

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${THIS_DIR}/.." && pwd)"

GZ_WS_DIR="${PROJECT_ROOT}/gz/${GZ_VERSION}_ws"
ROS_GZ_WS_DIR="${PROJECT_ROOT}/ros2/ros_gz_ws"

# ros_gz control:
#  - ROS_GZ_MODE=auto  => Gazebo=binary -> ros_gz=binary, Gazebo=source -> ros_gz=source
#  - ROS_GZ_MODE=binary|source => force that mode regardless of Gazebo install
#  - --no-ros-gz disables ros_gz entirely
INSTALL_ROS_GZ="true"
ROS_GZ_MODE="auto"      # auto | binary | source
ROS_GZ_REPO_URL="https://github.com/gazebosim/ros_gz.git"
ROS_GZ_REPO_BRANCH="${ROS_DISTRO}"

# Enable OSRF rosdep rules for Gazebo keys (optional).
ENABLE_GZ_ROSDEP_RULES="false"

# Where to fetch Gazebo repos from in source mode:
GZ_REPOS_YAML="https://raw.githubusercontent.com/gazebo-tooling/gazebodistro/master/collection-harmonic.yaml"

# ------------------------------------------------------------------------------
# Help / error handling
# ------------------------------------------------------------------------------
help() {
  cat <<EOF
Usage:
  bash gazebo.sh [options]

Options:
  -h, --help
      Show this help and exit.

  --debug
      Enable shell tracing (set -x).

  -i, --install MODE
      Gazebo install mode: binary | source
      Default: ${INSTALL_MODE}

  --phase PHASE
      Phase: fetch | deps | env | build | all
      Default: ${PHASE}

  --ros-gz MODE
      ros_gz mode: auto | binary | source
      Default: ${ROS_GZ_MODE}
      auto => follows Gazebo install mode

  --no-ros-gz
      Skip ros_gz entirely.

  --enable-gz-rosdep-rules
      Install OSRF Gazebo rosdep rules for source builds.

  --project-root PATH
      Override project root.
      Default: ${PROJECT_ROOT}

  --build-type TYPE
      CMake build type.
      Default: ${BUILD_TYPE}

  --gz-repos-yaml URL_OR_PATH
      Gazebo collection YAML used for vcs import.
      Default: ${GZ_REPOS_YAML}

  --ros-gz-repo-url URL
      ros_gz repository URL.
      Default: ${ROS_GZ_REPO_URL}

  --ros-gz-branch BRANCH
      ros_gz branch name.
      Default: ${ROS_GZ_REPO_BRANCH}

Examples:
  # Binary install
  bash install/gazebo.sh --install binary --phase all

  # Binary Gazebo + source ros_gz
  bash install/gazebo.sh --install binary --ros-gz source --phase fetch
  bash install/gazebo.sh --install binary --ros-gz source --phase deps
  bash install/gazebo.sh --install binary --ros-gz source --phase env
  bash install/gazebo.sh --install binary --ros-gz source --phase build

  # Source fetch on host
  bash install/gazebo.sh --install source --phase fetch

  # Source deps/env in Dockerfile
  bash install/gazebo.sh --install source --phase deps
  bash install/gazebo.sh --install source --phase env

  # Source build in container
  bash install/gazebo.sh --install source --phase build
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

# Prevent sourcing.
(return 0 2>/dev/null) && {
  echo "Do not source this script. Run: bash $0" >&2
  return 1
}

# ------------------------------------------------------------------------------
# Parse args
# ------------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      help
      exit 0
      ;;
    --debug)
      DEBUG="true"
      shift
      ;;
    -i|--install)
      [[ $# -ge 2 ]] || die "--install requires binary|source"
      INSTALL_MODE="$2"
      shift 2
      ;;
    --phase)
      [[ $# -ge 2 ]] || die "--phase requires fetch|deps|env|build|all"
      PHASE="$2"
      shift 2
      ;;
    --ros-gz)
      [[ $# -ge 2 ]] || die "--ros-gz requires auto|binary|source"
      ROS_GZ_MODE="$2"
      shift 2
      ;;
    --no-ros-gz)
      INSTALL_ROS_GZ="false"
      shift
      ;;
    --enable-gz-rosdep-rules)
      ENABLE_GZ_ROSDEP_RULES="true"
      shift
      ;;
    --project-root)
      [[ $# -ge 2 ]] || die "--project-root requires a path"
      PROJECT_ROOT="$2"
      GZ_WS_DIR="${PROJECT_ROOT}/gz/${GZ_VERSION}_ws"
      ROS_GZ_WS_DIR="${PROJECT_ROOT}/ros2/ros_gz_ws"
      shift 2
      ;;
    --build-type)
      [[ $# -ge 2 ]] || die "--build-type requires a value"
      BUILD_TYPE="$2"
      shift 2
      ;;
    --gz-repos-yaml)
      [[ $# -ge 2 ]] || die "--gz-repos-yaml requires a URL or path"
      GZ_REPOS_YAML="$2"
      shift 2
      ;;
    --ros-gz-repo-url)
      [[ $# -ge 2 ]] || die "--ros-gz-repo-url requires a URL"
      ROS_GZ_REPO_URL="$2"
      shift 2
      ;;
    --ros-gz-branch)
      [[ $# -ge 2 ]] || die "--ros-gz-branch requires a value"
      ROS_GZ_REPO_BRANCH="$2"
      shift 2
      ;;
    *)
      die "Unknown option: $1"
      ;;
  esac
done

[[ "${INSTALL_MODE}" == "binary" || "${INSTALL_MODE}" == "source" ]] \
  || die "Invalid --install mode: ${INSTALL_MODE}"

[[ "${PHASE}" == "fetch" || "${PHASE}" == "deps" || "${PHASE}" == "env" || "${PHASE}" == "build" || "${PHASE}" == "all" ]] \
  || die "Invalid --phase: ${PHASE}"

case "${ROS_GZ_MODE}" in
  auto|binary|source) ;;
  *) die "Invalid --ros-gz mode: ${ROS_GZ_MODE}" ;;
esac

# ------------------------------------------------------------------------------
# Strict mode
# ------------------------------------------------------------------------------
set -Eeuo pipefail
trap 'echo "[gazebo.sh] ERROR line=$LINENO cmd=$BASH_COMMAND" >&2' ERR

if [[ "${DEBUG}" == "true" ]]; then
  set -x
fi

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
get_effective_ros_gz_mode() {
  local mode="${ROS_GZ_MODE}"
  if [[ "${mode}" == "auto" ]]; then
    mode="${INSTALL_MODE}"
  fi
  echo "${mode}"
}

require_command() {
  local cmd="$1"
  command -v "${cmd}" >/dev/null 2>&1 || die "Required command not found: ${cmd}"
}

require_file() {
  local path="$1"
  [[ -f "${path}" ]] || die "Required file not found: ${path}"
}

require_dir() {
  local path="$1"
  [[ -d "${path}" ]] || die "Required directory not found: ${path}"
}

fetch_to_file() {
  local src="$1"
  local out="$2"

  if [[ "${src}" =~ ^https?:// ]]; then
    require_command curl
    curl -L --fail -o "${out}" "${src}"
  else
    require_file "${src}"
    cp -f "${src}" "${out}"
  fi
}

add_osrf_repo() {
  echo "==> Adding OSRF Gazebo apt repository"
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    wget

  sudo mkdir -p /usr/share/keyrings
  sudo wget -q https://packages.osrfoundation.org/gazebo.gpg \
    -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg

  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
    | sudo tee /etc/apt/sources.list.d/gazebo-stable.list >/dev/null

  sudo apt-get update

  if [[ "${DO_UPGRADE}" == "true" ]]; then
    sudo apt-get -y upgrade
  fi
}

install_gz_rosdep_rules() {
  [[ "${ENABLE_GZ_ROSDEP_RULES}" == "true" ]] || return 0

  echo "==> Installing OSRF Gazebo rosdep rules"
  require_command rosdep

  local dst="/etc/ros/rosdep/sources.list.d/00-gazebo.list"
  sudo mkdir -p "$(dirname "${dst}")"
  sudo bash -c "wget -q https://raw.githubusercontent.com/osrf/osrf-rosdep/master/gz/00-gazebo.list -O '${dst}'"

  sudo rosdep init 2>/dev/null || true
  rosdep update
}

source_ros_setup_if_exists() {
  local ros_setup="/opt/ros/${ROS_DISTRO}/setup.bash"
  [[ -f "${ros_setup}" ]] || die "ROS setup not found: ${ros_setup}"
  # shellcheck disable=SC1090
  source "${ros_setup}"
}

source_gz_overlay_if_exists() {
  local gz_setup="${GZ_WS_DIR}/install/setup.bash"
  if [[ -f "${gz_setup}" ]]; then
    echo "==> Sourcing Gazebo overlay: ${gz_setup}"
    # shellcheck disable=SC1090
    source "${gz_setup}"
  fi
}

write_env_script() {
  local env_dir="${PROJECT_ROOT}/gz/env"
  local env_file="${env_dir}/gz_env.sh"

  mkdir -p "${env_dir}"

  cat > "${env_file}" <<EOF
#!/usr/bin/env bash
# Auto-generated by install/gazebo.sh - source this block to set up Gazebo environment
# This block is managed by install/gazebo.sh - do not edit manually
# To update this block, edit and re-run install/gazebo.sh

if [[ -f "/opt/ros/${ROS_DISTRO}/setup.bash" ]]; then
  source "/opt/ros/${ROS_DISTRO}/setup.bash"
fi

if [[ -f "${GZ_WS_DIR}/install/setup.bash" ]]; then
  source "${GZ_WS_DIR}/install/setup.bash"
fi

if [[ -f "${ROS_GZ_WS_DIR}/install/setup.bash" ]]; then
  source "${ROS_GZ_WS_DIR}/install/setup.bash"
fi
EOF

  chmod +x "${env_file}"

  echo "==> Wrote environment helper:"
  echo "    ${env_file}"
}

# ------------------------------------------------------------------------------
# Shared dependency installation
# ------------------------------------------------------------------------------
install_common_deps() {
  echo ""
  echo "==> Installing common dependencies"
  echo ""

  sudo apt-get update
  sudo apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    gnupg \
    lsb-release \
    ninja-build \
    pkg-config \
    python3-colcon-common-extensions \
    python3-pip \
    python3-rosdep \
    python3-vcstool \
    python3-venv \
    wget

  if [[ "${DO_UPGRADE}" == "true" ]]; then
    sudo apt-get -y upgrade
  fi
}

# ------------------------------------------------------------------------------
# GZ and ros_gz installation helpers
# ------------------------------------------------------------------------------
install_gz_binary() {
  echo ""
  echo "==> Installing Gazebo binary package"
  echo ""
  sudo apt-get install -y --no-install-recommends "gz-${GZ_VERSION}"
}

install_ros_gz_binary() {
  [[ "${INSTALL_ROS_GZ}" == "true" ]] || return 0

  local mode
  mode="$(get_effective_ros_gz_mode)"
  [[ "${mode}" == "binary" ]] || return 0

  echo ""
  echo "==> Installing ros_gz binary package"
  echo ""

  case "${ROS_DISTRO}:${GZ_VERSION}" in
    humble:fortress)
      sudo apt-get install -y --no-install-recommends "ros-${ROS_DISTRO}-ros-gz"
      ;;
    jazzy:harmonic)
      sudo apt-get install -y --no-install-recommends "ros-${ROS_DISTRO}-ros-gz"
      ;;
    humble:harmonic)
      echo "WARNING: ROS 2 Humble + Gazebo Harmonic is a non-default pairing."
      sudo apt-get install -y --no-install-recommends "ros-${ROS_DISTRO}-ros-gzharmonic"
      ;;
    *)
      die "Unsupported binary ros_gz pairing: ROS_DISTRO=${ROS_DISTRO}, GZ_VERSION=${GZ_VERSION}"
      ;;
  esac
}

build_gz_source() {
  echo ""
  echo "==> Building Gazebo from source"
  echo ""

  require_dir "${GZ_WS_DIR}"
  require_dir "${GZ_WS_DIR}/src"

  cd "${GZ_WS_DIR}"
  colcon build --merge-install \
    --cmake-args \
      -DBUILD_TESTING=OFF \
      -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
}

build_ros_gz_source() {
  [[ "${INSTALL_ROS_GZ}" == "true" ]] || return 0

  local mode
  mode="$(get_effective_ros_gz_mode)"
  [[ "${mode}" == "source" ]] || return 0

  echo ""
  echo "==> Building ros_gz from source"
  echo ""

  require_dir "${ROS_GZ_WS_DIR}"
  require_dir "${ROS_GZ_WS_DIR}/src"

  source_ros_setup_if_exists
  source_gz_overlay_if_exists

  cd "${ROS_GZ_WS_DIR}"
  colcon build --merge-install \
    --cmake-args \
      -DBUILD_TESTING=OFF \
      -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
}

# ------------------------------------------------------------------------------
# Source fetch
# ------------------------------------------------------------------------------
fetch_gz_source() {
  echo ""
  echo "==> Fetching Gazebo source tree with vcs import"
  echo "    destination: ${GZ_WS_DIR}/src"
  echo "    collection:  ${GZ_REPOS_YAML}"
  echo ""

  require_command vcs
  require_command curl

  local gz_src="${GZ_WS_DIR}/src"
  mkdir -p "${gz_src}"

  local repos_file="${gz_src}/repos.yaml"
  fetch_to_file "${GZ_REPOS_YAML}" "${repos_file}"

  cd "${gz_src}"
  vcs import < "${repos_file}"
}

fetch_ros_gz_source() {
  [[ "${INSTALL_ROS_GZ}" == "true" ]] || return 0

  local mode
  mode="$(get_effective_ros_gz_mode)"
  [[ "${mode}" == "source" ]] || return 0

  echo ""
  echo "==> Fetching ros_gz source"
  echo "    destination: ${ROS_GZ_WS_DIR}/src/ros_gz"
  echo "    repo:        ${ROS_GZ_REPO_URL}"
  echo "    branch:      ${ROS_GZ_REPO_BRANCH}"
  echo ""

  require_command git

  local ros_gz_src="${ROS_GZ_WS_DIR}/src"
  mkdir -p "${ros_gz_src}"

  if [[ -d "${ros_gz_src}/ros_gz/.git" ]]; then
    echo "==> ros_gz already exists. Skipping clone."
  else
    git clone -b "${ROS_GZ_REPO_BRANCH}" "${ROS_GZ_REPO_URL}" "${ros_gz_src}/ros_gz"
  fi
}

# ------------------------------------------------------------------------------
# Source dependency helpers
# ------------------------------------------------------------------------------
install_source_gz_deps() {
  echo ""
  echo "==> Installing Gazebo source-build dependencies"
  echo ""

  add_osrf_repo

  local gz_src="${GZ_WS_DIR}/src"
  require_dir "${gz_src}"

  cd "${gz_src}"

  echo "==> Installing apt packages declared by Gazebo source repositories"

  local pkg_files=""
  pkg_files="$(find . \( -iname "packages-$(lsb_release -cs).apt" -o -iname "packages.apt" \) | grep -v '/\.git/' || true)"

  if [[ -z "${pkg_files}" ]]; then
    echo "WARNING: No Gazebo package manifest files found under ${gz_src}"
    return 0
  fi

  local packages=""
  while IFS= read -r file; do
    [[ -n "${file}" ]] || continue
    while IFS= read -r pkg; do
      [[ -n "${pkg}" ]] || continue
      packages+="${pkg}"$'\n'
    done < "${file}"
  done <<< "${pkg_files}"

  packages="$(printf "%s" "${packages}" | sed '/^\s*$/d' | sed '/gz\|sdf/d' | sort -u || true)"

  if [[ -n "${packages}" ]]; then
    # shellcheck disable=SC2086
    sudo apt-get install -y ${packages//$'\n'/ }
  fi
}

install_ros_gz_source_deps() {
  [[ "${INSTALL_ROS_GZ}" == "true" ]] || return 0

  local mode
  mode="$(get_effective_ros_gz_mode)"
  [[ "${mode}" == "source" ]] || return 0

  echo ""
  echo "==> Installing ros_gz source-build dependencies"
  echo ""

  install_gz_rosdep_rules

  sudo rosdep init 2>/dev/null || true
  rosdep update

  local ros_gz_src="${ROS_GZ_WS_DIR}/src"
  require_dir "${ros_gz_src}"

  source_ros_setup_if_exists
  source_gz_overlay_if_exists

  cd "${ROS_GZ_WS_DIR}"
  rosdep install -r --from-paths src -i -y --rosdistro "${ROS_DISTRO}" || true
}

# ------------------------------------------------------------------------------
# Binary mode phases
# ------------------------------------------------------------------------------
run_binary_fetch_phase() {
  echo ""
  echo "==> Binary installation : fetch"
  echo ""

  # Binary Gazebo itself does not need source fetch.
  # However, binary Gazebo + source ros_gz is supported, so fetch ros_gz here if requested.
  fetch_ros_gz_source
}

run_binary_deps_phase() {
  echo ""
  echo "==> Binary installation : deps"
  echo ""

  install_common_deps
  add_osrf_repo

  # Needed only when ros_gz is built from source on top of binary Gazebo.
  install_ros_gz_source_deps
}

run_binary_env_phase() {
  echo ""
  echo "==> Binary installation : env"
  echo ""

  echo "Gazebo binary will be available as:"
  echo "  /usr/bin/gz"

  if [[ "${INSTALL_ROS_GZ}" == "true" ]]; then
    local mode
    mode="$(get_effective_ros_gz_mode)"
    if [[ "${mode}" == "source" ]]; then
      echo ""
      echo "After building ros_gz from source:"
      echo "  source ${ROS_GZ_WS_DIR}/install/setup.bash"
    else
      echo ""
      echo "ros_gz binary environment (installed in ROS packages):"
      echo "  source /opt/ros/${ROS_DISTRO}/setup.bash"
    fi
  fi

  write_env_script
}

run_binary_build_phase() {
  echo ""
  echo "==> Binary installation : build"
  echo ""

  install_gz_binary

  if [[ "${INSTALL_ROS_GZ}" == "true" ]]; then
    local mode
    mode="$(get_effective_ros_gz_mode)"

    case "${mode}" in
      binary)
        install_ros_gz_binary
        ;;
      source)
        build_ros_gz_source
        ;;
      *)
        die "Invalid effective ros_gz mode: ${mode}"
        ;;
    esac
  fi
}

main_binary_installation() {
  local phase="${PHASE}"

  echo ""
  echo "==> Binary installation"
  echo "    PHASE=${phase}"
  echo "    ROS_GZ_MODE=$(get_effective_ros_gz_mode)"
  echo ""

  case "${phase}" in
    fetch)
      run_binary_fetch_phase
      ;;
    deps)
      run_binary_deps_phase
      ;;
    env)
      run_binary_env_phase
      ;;
    build)
      run_binary_build_phase
      ;;
    all)
      run_binary_fetch_phase
      run_binary_deps_phase
      run_binary_env_phase
      run_binary_build_phase
      ;;
    *)
      die "Unsupported phase for binary installation: ${phase}"
      ;;
  esac
}

# ------------------------------------------------------------------------------
# Source mode phases
# ------------------------------------------------------------------------------
run_source_fetch_phase() {
  echo ""
  echo "==> Source installation : fetch"
  echo ""

  fetch_gz_source
  fetch_ros_gz_source
}

run_source_deps_phase() {
  echo ""
  echo "==> Source installation : deps"
  echo ""

  install_common_deps
  install_source_gz_deps
  install_ros_gz_source_deps
}

run_source_env_phase() {
  echo ""
  echo "==> Source installation : env"
  echo ""

  echo "After building Gazebo from source:"
  echo "  source ${GZ_WS_DIR}/install/setup.bash"

  if [[ "${INSTALL_ROS_GZ}" == "true" ]]; then
    local mode
    mode="$(get_effective_ros_gz_mode)"
    if [[ "${mode}" == "source" ]]; then
      echo ""
      echo "After building ros_gz from source:"
      echo "  source ${ROS_GZ_WS_DIR}/install/setup.bash"
    else
      echo ""
      echo "ros_gz binary environment (installed in ROS packages):"
      echo "  source /opt/ros/${ROS_DISTRO}/setup.bash"
    fi
  fi

  write_env_script
}

run_source_build_phase() {
  echo ""
  echo "==> Source installation : build"
  echo ""

  build_gz_source

  if [[ "${INSTALL_ROS_GZ}" == "true" ]]; then
    local mode
    mode="$(get_effective_ros_gz_mode)"

    case "${mode}" in
      source)
        build_ros_gz_source
        ;;
      binary)
        install_ros_gz_binary
        ;;
      *)
        die "Invalid effective ros_gz mode: ${mode}"
        ;;
    esac
  fi
}

main_source_installation() {
  local phase="${PHASE}"

  echo ""
  echo "==> Source installation"
  echo "    PHASE=${phase}"
  echo "    ROS_GZ_MODE=$(get_effective_ros_gz_mode)"
  echo ""

  case "${phase}" in
    fetch)
      run_source_fetch_phase
      ;;
    deps)
      run_source_deps_phase
      ;;
    env)
      run_source_env_phase
      ;;
    build)
      run_source_build_phase
      ;;
    all)
      run_source_fetch_phase
      run_source_deps_phase
      run_source_env_phase
      run_source_build_phase
      ;;
    *)
      die "Unsupported phase for source installation: ${phase}"
      ;;
  esac
}

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
main() {
  case "${INSTALL_MODE}" in
    binary)
      main_binary_installation
      ;;
    source)
      main_source_installation
      ;;
    *)
      die "Unhandled INSTALL_MODE: ${INSTALL_MODE}"
      ;;
  esac

  echo ""
  echo "==> gazebo.sh completed successfully"
  echo "    INSTALL_MODE=${INSTALL_MODE}"
  echo "    PHASE=${PHASE}"
  echo "    ROS_GZ_MODE=$(get_effective_ros_gz_mode)"
  echo ""
}

main "$@"