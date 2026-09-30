#!/usr/bin/env bash
# autopilot.sh — install autopilot dependencies, fetch auxiliary sources,
# build helper components, and generate environment snippets for PX4 and/or
# ArduPilot workflows.
#
# Modes:
#   --phase deps   : install system dependencies only
#   --phase fetch  : verify local source trees and fetch auxiliary repositories
#   --phase build  : build/install already-fetched sources only
#   --phase env    : generate environment helper snippets only
#   --phase all    : run deps + fetch + build + env (default)
#
# Design goals:
#   - Keep Docker image build and runtime setup separable
#   - Support mounted-source development workflows
#   - Support submodule-based repository layouts for PX4 and ArduPilot
#
# Notes:
#   - This script assumes ROS 2 is already installed at /opt/ros/<distro>.
#   - PX4 and ArduPilot are assumed to already exist locally, typically as
#     submodules under ap/px4 and ap/ardupilot.
#   - PX4 firmware itself is not cloned or built here.
#   - ArduPilot firmware itself is not cloned or built here.
#   - px4_msgs is fetched into a ROS 2 workspace because it is often needed
#     independently from the PX4 firmware tree.
#   - Micro XRCE-DDS Agent may be built either:
#       (A) inside a ROS 2 workspace with colcon
#       (B) from source and installed into /usr/local
#   - ArduPilot Gazebo plugin may build against either:
#       (A) a Gazebo source overlay
#       (B) system-installed Gazebo dev packages

# --------------------------
# Defaults
# --------------------------
DEBUG="false"
PHASE="all"  # deps | fetch | build | env | all

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$THIS_DIR/.." && pwd)"

# Packages to be installed by this script
WITH_PX4="true"
WITH_PX4_MSGS="true"
WITH_ARDUPILOT="false"

# PX4
PX4_REF="v1.16.2-moonshot"
PX4_DIR="${PROJECT_ROOT}/ap/px4"

# ArduPilot
ARDUPILOT_REF="Copter-4.6.2-moonshot"
ARDUPILOT_DIR="${PROJECT_ROOT}/ap/ardupilot"

# px4_msgs (ROS 2 message package)
PX4_MSGS_REF="v1.16.2"
PX4_MSGS_DIR="${PROJECT_ROOT}/ros2/px4_msgs_ws"

# Micro XRCE-DDS Agent
DDS_MODE="ros2"   # ros2 | source
ROS_DISTRO="humble"
ROS2_WS_DIR="${PROJECT_ROOT}/ros2/px4_ros_uxrce_dds_ws"
DDS_AGENT_REF=""  # auto-pick by ROS_DISTRO if empty
DDS_AGENT_DIR="${PROJECT_ROOT}/tools/Micro-XRCE-DDS-Agent"

# TODO: PX4 Gazebo models
PX4_SITL_MODELS_DIR="${PROJECT_ROOT}/gz/PX4_gazebo_models"

# ArduPilot Gazebo plugin + models
GZ_VERSION="harmonic"
ARDUPILOT_GZ_DIR="${PROJECT_ROOT}/gz/ardupilot_gazebo"
ARDUPILOT_SITL_MODELS_DIR="${PROJECT_ROOT}/gz/SITL_Models"
ARDUPILOT_GZ_BUILD_TYPE="RelWithDebInfo"

# Optional: explicitly point to a Gazebo overlay setup.bash.
# If empty, auto-detect:
#   ${PROJECT_ROOT}/gz/${GZ_VERSION}_ws/install/setup.bash
GZ_OVERLAY_SETUP=""

help() {
  cat <<EOF
Usage:
  bash install/autopilot.sh [options]

Options:
  -h, --help                 Show this help and exit
  --debug                    Enable command tracing (set -x)

  --phase PHASE              What to run: deps | fetch | build | env | all
                             deps  = install system dependencies only
                             fetch = verify local source trees and fetch
                                     auxiliary source repositories only
                             build = build/install already-fetched source only
                             env   = generate environment helper files only
                             all   = run everything in order (default)

  --project-root PATH        Project root (default: ${PROJECT_ROOT})

  # Select autopilots
  --no-px4                   Disable PX4 flow
  --no-px4-msgs              Disable px4_msgs setup
  --with-ardupilot           Enable ArduPilot flow

  # PX4 controls
  --px4-ref REF              PX4 ref/tag/branch for reference only
                             (default: ${PX4_REF})
  --px4-path PATH            Where PX4 repo should live (default: ${PX4_DIR})

  # PX4 ROS messages controls
  --px4-msgs-ref REF         px4_msgs ref/tag/branch (default: ${PX4_MSGS_REF})
  --px4-msgs-path PATH       px4_msgs directory
                             (default: ${PX4_MSGS_DIR})

  # Micro XRCE-DDS Agent controls
  --dds-mode MODE            Agent install mode: ros2 | source
                             (default: ${DDS_MODE})
  --ros-distro DISTRO        ROS 2 distro (default: ${ROS_DISTRO})
  --ros2-ws PATH             ROS 2 workspace for agent build
                             (default: ${ROS2_WS_DIR})
  --dds-ref REF              Micro-XRCE-DDS-Agent ref/tag
                             (default: auto by ROS distro)
  --dds-path PATH            DDS Agent source dir for source mode
                             (default: ${DDS_AGENT_DIR})

  # ArduPilot controls
  --ardupilot-ref REF        ArduPilot ref/tag/branch for reference only
                             (default: ${ARDUPILOT_REF})
  --ardupilot-path PATH      Where ArduPilot repo should live
                             (default: ${ARDUPILOT_DIR})

  # ArduPilot Gazebo plugin controls
  --gz-version VER           Gazebo version name used for deps mapping
                             (default: ${GZ_VERSION})
  --gz-overlay-setup PATH    Optional setup.bash for Gazebo source overlay
  --ardupilot-gz-path PATH   Where ardupilot_gazebo should live
                             (default: ${ARDUPILOT_GZ_DIR})
  --sitl-models-path PATH    Where SITL_Models should live
                             (default: ${ARDUPILOT_SITL_MODELS_DIR})
  --ardupilot-gz-build-type  CMake build type for ardupilot_gazebo
                             (default: ${ARDUPILOT_GZ_BUILD_TYPE})

Examples:
  # Install system packages only
  bash install/autopilot.sh --phase deps

  # Verify PX4 submodule/source and fetch px4_msgs + DDS source
  bash install/autopilot.sh --phase fetch

  # Enable ArduPilot flow as well
  bash install/autopilot.sh --phase fetch --with-ardupilot

  # Build fetched helper components
  bash install/autopilot.sh --phase build --with-ardupilot

  # Generate environment snippet
  bash install/autopilot.sh --phase env --with-ardupilot

  # Everything
  bash install/autopilot.sh --phase all --with-ardupilot
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

    --phase)
      [[ $# -ge 2 ]] || die "--phase requires a value (deps|fetch|build|env|all)"
      PHASE="$2"
      shift 2
      ;;

    --project-root)
      [[ $# -ge 2 ]] || die "--project-root requires a path"
      PROJECT_ROOT="$2"
      shift 2
      PX4_DIR="${PROJECT_ROOT}/ap/px4"
      ARDUPILOT_DIR="${PROJECT_ROOT}/ap/ardupilot"
      PX4_MSGS_DIR="${PROJECT_ROOT}/ros2/px4_msgs_ws"
      ROS2_WS_DIR="${PROJECT_ROOT}/ros2/px4_ros_uxrce_dds_ws"
      DDS_AGENT_DIR="${PROJECT_ROOT}/tools/Micro-XRCE-DDS-Agent"
      PX4_SITL_MODELS_DIR="${PROJECT_ROOT}/gz/PX4_gazebo_models"
      ARDUPILOT_GZ_DIR="${PROJECT_ROOT}/gz/ardupilot_gazebo"
      ARDUPILOT_SITL_MODELS_DIR="${PROJECT_ROOT}/gz/SITL_Models"
      ;;

    # package installation selection
    --no-px4) WITH_PX4="false"; shift ;;
    --no-px4-msgs) WITH_PX4_MSGS="false"; shift ;;
    --with-ardupilot) WITH_ARDUPILOT="true"; shift ;;

    # px4 controls
    --px4-ref) [[ $# -ge 2 ]] || die "--px4-ref requires a value"; PX4_REF="$2"; shift 2 ;;
    --px4-path) [[ $# -ge 2 ]] || die "--px4-path requires a path"; PX4_DIR="$2"; shift 2 ;;

    # ardupilot controls
    --ardupilot-ref) [[ $# -ge 2 ]] || die "--ardupilot-ref requires a value"; ARDUPILOT_REF="$2"; shift 2 ;;
    --ardupilot-path) [[ $# -ge 2 ]] || die "--ardupilot-path requires a path"; ARDUPILOT_DIR="$2"; shift 2 ;;

    # px4_msgs controls
    --px4-msgs-ref) [[ $# -ge 2 ]] || die "--px4-msgs-ref requires a value"; PX4_MSGS_REF="$2"; shift 2 ;;
    --px4-msgs-path) [[ $# -ge 2 ]] || die "--px4-msgs-path requires a path"; PX4_MSGS_DIR="$2"; shift 2 ;;

    # dds / ros2 ws controls
    --dds-mode) [[ $# -ge 2 ]] || die "--dds-mode requires a value"; DDS_MODE="$2"; shift 2 ;;
    --ros-distro) [[ $# -ge 2 ]] || die "--ros-distro requires a value"; ROS_DISTRO="$2"; shift 2 ;;
    --ros2-ws) [[ $# -ge 2 ]] || die "--ros2-ws requires a path"; ROS2_WS_DIR="$2"; shift 2 ;;
    --dds-ref) [[ $# -ge 2 ]] || die "--dds-ref requires a value"; DDS_AGENT_REF="$2"; shift 2 ;;
    --dds-path) [[ $# -ge 2 ]] || die "--dds-path requires a path"; DDS_AGENT_DIR="$2"; shift 2 ;;

    # ardupilot gazebo plugin controls
    --gz-version) [[ $# -ge 2 ]] || die "--gz-version requires a value"; GZ_VERSION="$2"; shift 2 ;;
    --gz-overlay-setup) [[ $# -ge 2 ]] || die "--gz-overlay-setup requires a path"; GZ_OVERLAY_SETUP="$2"; shift 2 ;;
    --ardupilot-gz-path) [[ $# -ge 2 ]] || die "--ardupilot-gz-path requires a path"; ARDUPILOT_GZ_DIR="$2"; shift 2 ;;
    --sitl-models-path) [[ $# -ge 2 ]] || die "--sitl-models-path requires a path"; ARDUPILOT_SITL_MODELS_DIR="$2"; shift 2 ;;
    --ardupilot-gz-build-type) [[ $# -ge 2 ]] || die "--ardupilot-gz-build-type requires a value"; ARDUPILOT_GZ_BUILD_TYPE="$2"; shift 2 ;;

    *) die "Unknown option: $1 (use --help)" ;;
  esac
done

# --------------------------
# Validate args
# --------------------------
case "${PHASE}" in
  deps|fetch|build|env|all) ;;
  *) die "Invalid --phase: ${PHASE} (expected deps|fetch|build|env|all)" ;;
esac

case "${DDS_MODE}" in
  ros2|source) ;;
  *) die "Invalid --dds-mode: ${DDS_MODE} (expected ros2|source)" ;;
esac

# --------------------------
# Strict mode
# --------------------------
set -Eeuo pipefail
trap 'echo "[autopilot.sh] ERROR line=$LINENO cmd=$BASH_COMMAND" >&2' ERR
if [[ "${DEBUG}" == "true" ]]; then
  set -x
fi

# --------------------------
# Helpers
# --------------------------
pick_dds_ref() {
  if [[ "${DDS_MODE}" == "source" ]]; then
    if [[ -n "${DDS_AGENT_REF}" && "${DDS_AGENT_REF}" != "v2.4.3" ]]; then
      echo "WARNING: dds-mode=source forces DDS_AGENT_REF=v2.4.3 (ignoring --dds-ref ${DDS_AGENT_REF})." >&2
    fi
    DDS_AGENT_REF="v2.4.3"
    return 0
  fi

  if [[ -n "${DDS_AGENT_REF}" ]]; then
    return 0
  fi

  case "${ROS_DISTRO}" in
    humble|foxy) DDS_AGENT_REF="v2.4.2" ;;
    jazzy) DDS_AGENT_REF="v2.4.3" ;;
    *)
      DDS_AGENT_REF="v2.4.2"
      echo "WARNING: Unknown ROS_DISTRO='${ROS_DISTRO}'. Defaulting DDS_AGENT_REF=${DDS_AGENT_REF}." >&2
      ;;
  esac
}

ensure_git_repo_or_clone() {
  local repo_url="$1"
  local repo_ref="$2"
  local repo_dir="$3"

  mkdir -p "$(dirname "${repo_dir}")"

  if [[ -d "${repo_dir}/.git" ]]; then
    git -C "${repo_dir}" fetch --all --tags || true
    git -C "${repo_dir}" checkout "${repo_ref}"
    git -C "${repo_dir}" pull --ff-only || true
  else
    git clone -b "${repo_ref}" "${repo_url}" "${repo_dir}"
  fi
}

ensure_submodule_present() {
  local repo_dir="$1"
  local repo_name="$2"

  if [[ ! -d "${repo_dir}" ]]; then
    die "${repo_name} directory not found: ${repo_dir}
Expected it to exist as a submodule or pre-populated source tree."
  fi

  if [[ -z "$(ls -A "${repo_dir}" 2>/dev/null)" ]]; then
    die "${repo_name} directory is empty: ${repo_dir}
Initialize submodules first, e.g.:
  git submodule update --init --recursive"
  fi
}

detect_or_set_gz_overlay() {
  if [[ -n "${GZ_OVERLAY_SETUP}" ]]; then
    [[ -f "${GZ_OVERLAY_SETUP}" ]] || die "--gz-overlay-setup not found: ${GZ_OVERLAY_SETUP}"
    return 0
  fi

  local candidate="${PROJECT_ROOT}/gz/${GZ_VERSION}_ws/install/setup.bash"
  if [[ -f "${candidate}" ]]; then
    GZ_OVERLAY_SETUP="${candidate}"
  else
    GZ_OVERLAY_SETUP=""
  fi
}

print_config() {
  echo "PROJECT_ROOT=${PROJECT_ROOT}"
  echo "PHASE=${PHASE}"
  echo "WITH_PX4=${WITH_PX4}"
  echo "WITH_PX4_MSGS=${WITH_PX4_MSGS}"
  echo "WITH_ARDUPILOT=${WITH_ARDUPILOT}"
  echo "DDS_MODE=${DDS_MODE}"
  echo "ROS_DISTRO=${ROS_DISTRO}"
  echo "ROS2_WS_DIR=${ROS2_WS_DIR}"
  echo "DDS_AGENT_REF=${DDS_AGENT_REF:-<auto>}"
  echo "DDS_AGENT_DIR=${DDS_AGENT_DIR}"
  echo "PX4_REF=${PX4_REF}"
  echo "PX4_DIR=${PX4_DIR}"
  echo "PX4_MSGS_REF=${PX4_MSGS_REF}"
  echo "PX4_MSGS_DIR=${PX4_MSGS_DIR}"
  echo "ARDUPILOT_REF=${ARDUPILOT_REF}"
  echo "ARDUPILOT_DIR=${ARDUPILOT_DIR}"
  echo "ARDUPILOT_GZ_DIR=${ARDUPILOT_GZ_DIR}"
  echo "ARDUPILOT_SITL_MODELS_DIR=${ARDUPILOT_SITL_MODELS_DIR}"
  echo "GZ_VERSION=${GZ_VERSION}"
}

# --------------------------
# PX4: deps / fetch / build / env
# --------------------------
px4_deps() {
  echo ""
  echo "==> Installing PX4 dependencies (PX4_REF=${PX4_REF})"
  echo "    (using official Tools/setup/ubuntu.sh --no-sim-tools)"
  echo ""

  local tmpdir="/tmp/px4_setup"
  mkdir -p "${tmpdir}"

  local ubuntu_sh_url="https://raw.githubusercontent.com/mhcho1994/PX4-Autopilot/${PX4_REF}/Tools/setup/ubuntu.sh"
  local req_txt_url="https://raw.githubusercontent.com/mhcho1994/PX4-Autopilot/${PX4_REF}/Tools/setup/requirements.txt"

  wget -q "${ubuntu_sh_url}" -O "${tmpdir}/ubuntu.sh"
  wget -q "${req_txt_url}" -O "${tmpdir}/requirements.txt"
  chmod +x "${tmpdir}/ubuntu.sh"

  bash "${tmpdir}/ubuntu.sh" --no-sim-tools
}

px4_fetch() {
  echo ""
  echo "==> Checking PX4 submodule/source directory"
  echo ""

  ensure_submodule_present "${PX4_DIR}" "PX4-Autopilot"

  echo "PX4 source directory is present: ${PX4_DIR}"
}

px4_build() {
  echo ""
  echo "==> PX4 build stage"
  echo ""
  echo "No PX4 firmware build is performed by this script."
  echo "This script only manages support dependencies around PX4."
}

px4_env() {
  echo ""
  echo "==> PX4 env stage"
  echo ""
  echo "No PX4-specific env snippet generated."
}

# --------------------------
# ArduPilot: deps / fetch / build / env
# --------------------------
ardupilot_deps() {
  echo ""
  echo "==> Installing ArduPilot dependencies (ARDUPILOT_REF=${ARDUPILOT_REF})"
  echo "    (using Tools/environment_install/install-prereqs-ubuntu.sh -y)"
  echo ""

  export DO_AP_STM_ENV=0
  export SKIP_AP_COMPLETION_ENV=1
  export SKIP_AP_GIT_CHECK=1

  local tmpdir="/tmp/ardupilot_setup_repo"
  rm -rf "${tmpdir}"
  mkdir -p "${tmpdir}"

  git clone --depth 1 --filter=blob:none --sparse --branch "${ARDUPILOT_REF}" \
    https://github.com/mhcho1994/ardupilot.git "${tmpdir}"

  pushd "${tmpdir}" >/dev/null
  git sparse-checkout set Tools/environment_install Tools/completion
  bash Tools/environment_install/install-prereqs-ubuntu.sh -y
  popd >/dev/null

  rm -rf "${tmpdir}"
}

ardupilot_fetch() {
  echo ""
  echo "==> Checking ArduPilot submodule/source directory"
  echo ""

  ensure_submodule_present "${ARDUPILOT_DIR}" "ArduPilot"

  echo "ArduPilot source directory is present: ${ARDUPILOT_DIR}"
}

ardupilot_build() {
  echo ""
  echo "==> ArduPilot build stage"
  echo ""
  echo "No ArduPilot firmware build is performed by this script."
}

ardupilot_env() {
  echo ""
  echo "==> ArduPilot env stage"
  echo ""
  # Add user-local bin (for MAVProxy, pip tools, etc.) to ArduPilot env file
  cat <<'EOF' >> ~/.ardupilot_env
if [ -d "$HOME/.local/bin" ] ; then
    PATH="$HOME/.local/bin:$PATH"
fi
EOF

  echo 'Added "$HOME/.local/bin" to ~/.ardupilot_env'
}

# --------------------------
# px4_gazebo_model: fetch /  env
# --------------------------
px4_gz_models_fetch() {
  echo ""
  echo "==> Checking PX4 Gazebo SITL Models submodule/source directory"
  echo ""

  ensure_submodule_present "${PX4_SITL_MODELS_DIR}" "PX4-gazebo-models"

  echo "PX4 Gazebo SITL Models source directory is present: ${PX4_SITL_MODELS_DIR}"
}

px4_gz_models_env() {
  local env_dir="${PROJECT_ROOT}/gz/env"
  local env_file="${env_dir}/px4_gz_env.sh"
  local fire_gz_dir="${PROJECT_ROOT}/gz/FIRE_moonshot_gazebo"

  mkdir -p "${env_dir}"

  cat > "${env_file}" <<EOF
#!/usr/bin/env bash
# Auto-generated by install/autopilot.sh
# Source this file to expose PX4 Gazebo resource paths.

export GZ_VERSION=${GZ_VERSION}

# Required when Gazebo is started separately from PX4.
export GZ_SIM_SERVER_CONFIG_PATH="${PX4_DIR}/src/modules/simulation/gz_bridge/server.config"

# Prepend without duplicating entries already present (avoids growth when
# this file is re-sourced, e.g. in nested interactive shells).
_gz_path_prepend() {
  local -n _var="\$1"
  local path="\$2"
  case ":\${_var}:" in
    *":\${path}:"*) ;;
    *) _var="\${path}:\${_var}" ;;
  esac
}

if [ -d "${PX4_SITL_MODELS_DIR}" ]; then
  _gz_path_prepend GZ_SIM_RESOURCE_PATH "${PX4_SITL_MODELS_DIR}/models"
  _gz_path_prepend GZ_SIM_RESOURCE_PATH "${PX4_SITL_MODELS_DIR}/worlds"
fi

if [ -d "${PX4_DIR}/Tools/simulation/gz" ]; then
  _gz_path_prepend GZ_SIM_RESOURCE_PATH "${PX4_DIR}/Tools/simulation/gz/models"
  _gz_path_prepend GZ_SIM_RESOURCE_PATH "${PX4_DIR}/Tools/simulation/gz/worlds"
fi

if [ -d "${fire_gz_dir}" ]; then
  _gz_path_prepend GZ_SIM_RESOURCE_PATH "${fire_gz_dir}/models"
  _gz_path_prepend GZ_SIM_RESOURCE_PATH "${fire_gz_dir}/worlds"
fi

_gz_path_prepend GZ_SIM_SYSTEM_PLUGIN_PATH "${PX4_DIR}/build/px4_sitl_default/src/modules/simulation/gz_plugins"
if [ -d "${PROJECT_ROOT}/build/fire_gz_plugins/lib" ]; then
  _gz_path_prepend GZ_SIM_SYSTEM_PLUGIN_PATH "${PROJECT_ROOT}/build/fire_gz_plugins/lib"
fi

export GZ_SIM_RESOURCE_PATH
export GZ_SIM_SYSTEM_PLUGIN_PATH
unset -f _gz_path_prepend
EOF

  echo ""
  echo "==> Wrote env snippet:"
  echo "    ${env_file}"
  echo ""
}

# --------------------------
# px4_msgs: fetch / build / env
# --------------------------
px4_msgs_fetch() {
  if [[ "${WITH_PX4_MSGS}" != "true" ]]; then
    echo ""
    echo "==> px4_msgs disabled (--no-px4-msgs)"
    echo ""
    return 0
  fi

  echo ""
  echo "==> Fetching px4_msgs source"
  echo "    ref=${PX4_MSGS_REF}"
  echo "    dir=${PX4_MSGS_DIR}/src/px4_msgs"
  echo ""

  ensure_git_repo_or_clone \
    "https://github.com/PX4/px4_msgs.git" \
    "${PX4_MSGS_REF}" \
    "${PX4_MSGS_DIR}/src/px4_msgs"
}

px4_msgs_build() {

  if [[ "${WITH_PX4_MSGS}" != "true" ]]; then
    echo ""
    echo "==> px4_msgs disabled (--no-px4-msgs)"
    echo ""
    return 0
  fi

  echo ""
  echo "==> Building px4_msgs in ROS 2 workspace"
  echo ""

  local ros_setup="/opt/ros/${ROS_DISTRO}/setup.bash"
  [[ -f "${ros_setup}" ]] || die "ROS 2 setup not found: ${ros_setup} (install ROS 2 ${ROS_DISTRO} first)"

  sudo rosdep init 2>/dev/null || true
  rosdep update

  local src_dir="${PX4_MSGS_DIR}/src"
  [[ -d "${src_dir}/px4_msgs" ]] || die "px4_msgs source not found. Run --phase fetch first."

  set +u
  source "${ros_setup}"
  set -u

  cd "${PX4_MSGS_DIR}"
  rosdep install -r --from-paths src -i -y --rosdistro "${ROS_DISTRO}" || true
  colcon build
}

px4_msgs_env() {
  echo ""
  echo "==> px4_msgs env stage"
  echo ""

  echo "px4_msgs built in ROS 2 workspace:"
  echo "  source /opt/ros/${ROS_DISTRO}/setup.bash"
  echo "  source ${PX4_MSGS_DIR}/install/local_setup.bash"
}

# --------------------------
# DDS Agent: deps / fetch / build / env
# --------------------------
dds_deps() {
  pick_dds_ref

  echo ""
  echo "==> Installing Micro XRCE-DDS Agent build dependencies"
  echo "    DDS_MODE=${DDS_MODE}"
  echo "    ROS_DISTRO=${ROS_DISTRO}"
  echo "    DDS_AGENT_REF=${DDS_AGENT_REF}"
  echo ""

  sudo apt-get -y update

  if [[ "${DDS_MODE}" == "ros2" ]]; then
    sudo apt-get -y --no-install-recommends install \
      git \
      build-essential \
      cmake \
      python3-colcon-common-extensions \
      python3-rosdep \
      python3-vcstool
  else
    sudo apt-get -y --no-install-recommends install \
      git \
      cmake \
      build-essential
  fi
}

dds_fetch() {
  pick_dds_ref

  echo ""
  echo "==> Fetching Micro XRCE-DDS Agent source"
  echo "    DDS_MODE=${DDS_MODE}"
  echo ""

  if [[ "${DDS_MODE}" == "ros2" ]]; then
    local src_dir="${ROS2_WS_DIR}/src"
    mkdir -p "${src_dir}"
    ensure_git_repo_or_clone \
      "https://github.com/eProsima/Micro-XRCE-DDS-Agent.git" \
      "${DDS_AGENT_REF}" \
      "${src_dir}/Micro-XRCE-DDS-Agent"
  else
    ensure_git_repo_or_clone \
      "https://github.com/eProsima/Micro-XRCE-DDS-Agent.git" \
      "${DDS_AGENT_REF}" \
      "${DDS_AGENT_DIR}"
  fi
}

dds_build_ros2_ws() {
  echo ""
  echo "==> Building Micro XRCE-DDS Agent in ROS 2 workspace"
  echo ""

  local ros_setup="/opt/ros/${ROS_DISTRO}/setup.bash"
  [[ -f "${ros_setup}" ]] || die "ROS 2 setup not found: ${ros_setup} (install ROS 2 ${ROS_DISTRO} first)"

  sudo rosdep init 2>/dev/null || true
  rosdep update

  local src_dir="${ROS2_WS_DIR}/src"
  [[ -d "${src_dir}/Micro-XRCE-DDS-Agent" ]] || die "DDS source not found. Run --phase fetch first."

  set +u
  source "${ros_setup}"
  set -u

  cd "${ROS2_WS_DIR}"
  rosdep install -r --from-paths src -i -y --rosdistro "${ROS_DISTRO}" || true
  colcon build
}

dds_build_source() {
  echo ""
  echo "==> Building/installing Micro XRCE-DDS Agent to /usr/local"
  echo ""

  [[ -d "${DDS_AGENT_DIR}" ]] || die "DDS source not found at ${DDS_AGENT_DIR}. Run --phase fetch first."

  pushd "${DDS_AGENT_DIR}" >/dev/null
  mkdir -p build
  cd build
  cmake ..
  make -j"$(nproc)"
  sudo make install
  sudo ldconfig
  popd >/dev/null
}

dds_build() {
  pick_dds_ref

  case "${DDS_MODE}" in
    ros2) dds_build_ros2_ws ;;
    source) dds_build_source ;;
    *) die "Internal error: unknown DDS_MODE=${DDS_MODE}" ;;
  esac
}

dds_env() {
  echo ""
  echo "==> DDS env stage"
  echo ""

  if [[ "${DDS_MODE}" == "ros2" ]]; then
    echo "DDS agent built in ROS 2 workspace:"
    echo "  source /opt/ros/${ROS_DISTRO}/setup.bash"
    echo "  source ${ROS2_WS_DIR}/install/local_setup.bash"
    echo "  MicroXRCEAgent udp4 -p 8888"
  else
    echo "DDS agent installed into /usr/local."
  fi
}

# --------------------------
# ArduPilot Gazebo plugin: deps / fetch / build / env
# --------------------------
ardupilot_gz_plugin_deps() {
  echo ""
  echo "==> Installing ArduPilot Gazebo plugin build dependencies only"
  echo "    GZ_VERSION=${GZ_VERSION}"
  echo ""

  local gz_sim_dev_pkg=""
  case "${GZ_VERSION}" in
    harmonic) gz_sim_dev_pkg="libgz-sim8-dev" ;;
    garden) gz_sim_dev_pkg="libgz-sim7-dev" ;;
    ionic)
      gz_sim_dev_pkg="libgz-sim8-dev"
      echo "WARNING: GZ_VERSION=ionic detected; defaulting dev pkg to ${gz_sim_dev_pkg}."
      ;;
    *) die "Unsupported GZ_VERSION for ArduPilot plugin deps: ${GZ_VERSION}" ;;
  esac

  sudo apt-get -y update
  sudo DEBIAN_FRONTEND=noninteractive apt-get -y --no-install-recommends install \
    git \
    cmake \
    ninja-build \
    pkg-config \
    build-essential \
    "${gz_sim_dev_pkg}" \
    rapidjson-dev \
    libopencv-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-libav \
    gstreamer1.0-gl
}

ardupilot_gz_plugin_fetch() {
  echo ""
  echo "==> Checking ArduPilot Gazebo plugin submodule/source directory"
  echo ""

  ensure_submodule_present "${ARDUPILOT_GZ_DIR}" "ardupilot_gazebo"

  echo "ArduPilot Gazebo plugin source directory is present: ${ARDUPILOT_GZ_DIR}"
}

sitl_models_fetch() {
  echo ""
  echo "==> Checking ArduPilot Gazebo SITL Models submodule/source directory"
  echo ""

  ensure_submodule_present "${ARDUPILOT_SITL_MODELS_DIR}" "SITL_Models"

  echo "ArduPilot Gazebo SITL Models source directory is present: ${ARDUPILOT_SITL_MODELS_DIR}"
}

ardupilot_gz_plugin_build() {
  echo ""
  echo "==> Building ArduPilot Gazebo plugin"
  echo ""

  [[ -d "${ARDUPILOT_GZ_DIR}" ]] || die "ardupilot_gazebo source not found. Run --phase fetch first."

  mkdir -p "${ARDUPILOT_GZ_DIR}/build"
  pushd "${ARDUPILOT_GZ_DIR}/build" >/dev/null

  detect_or_set_gz_overlay
  if [[ -n "${GZ_OVERLAY_SETUP}" ]]; then
    # shellcheck disable=SC1090
    source "${GZ_OVERLAY_SETUP}"
    echo "==> Using Gazebo overlay for build: ${GZ_OVERLAY_SETUP}"
  else
    echo "==> No Gazebo overlay detected; building against system Gazebo."
  fi

  cmake .. -DCMAKE_BUILD_TYPE="${ARDUPILOT_GZ_BUILD_TYPE}"
  cmake --build . -j"$(nproc)"
  popd >/dev/null

  echo ""
  echo "==> ArduPilot Gazebo plugin built at:"
  echo "    ${ARDUPILOT_GZ_DIR}/build"
  echo ""
}

fire_gz_plugins_build() {
  local source_dir="${PROJECT_ROOT}/gz/FIRE_moonshot_gazebo/plugins/plugins"
  local build_dir="${PROJECT_ROOT}/build/fire_gz_plugins"

  echo ""
  echo "==> Building FIRE Gazebo plugins"
  echo ""

  [[ -f "${source_dir}/CMakeLists.txt" ]] || die "FIRE Gazebo plugin source not found: ${source_dir}"

  detect_or_set_gz_overlay
  if [[ -n "${GZ_OVERLAY_SETUP}" ]]; then
    # shellcheck disable=SC1090
    source "${GZ_OVERLAY_SETUP}"
  fi

  cmake -S "${source_dir}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE="${ARDUPILOT_GZ_BUILD_TYPE}"
  cmake --build "${build_dir}" -j"$(nproc)"

  echo ""
  echo "==> FIRE Gazebo plugins built at: ${build_dir}/lib"
  echo ""
}

ardupilot_gz_plugin_env() {
  local env_dir="${PROJECT_ROOT}/gz/env"
  local env_file="${env_dir}/ardupilot_gz_env.sh"
  local fire_gz_dir="${PROJECT_ROOT}/gz/FIRE_moonshot_gazebo"
  local fire_gz_plugin_lib="${PROJECT_ROOT}/build/fire_gz_plugins/lib"

  mkdir -p "${env_dir}"

  cat > "${env_file}" <<EOF
#!/usr/bin/env bash
# Auto-generated by install/autopilot.sh
# Source this file to expose ArduPilot Gazebo plugin and resource paths.

export GZ_VERSION=${GZ_VERSION}

# Prepend without duplicating entries already present (avoids growth when
# this file is re-sourced, e.g. in nested interactive shells).
_gz_path_prepend() {
  local -n _var="\$1"
  local path="\$2"
  case ":\${_var}:" in
    *":\${path}:"*) ;;
    *) _var="\${path}:\${_var}" ;;
  esac
}

if [ -d "${ARDUPILOT_GZ_DIR}" ]; then
  _gz_path_prepend GZ_SIM_SYSTEM_PLUGIN_PATH "${ARDUPILOT_GZ_DIR}/build"
  _gz_path_prepend GZ_SIM_RESOURCE_PATH "${ARDUPILOT_GZ_DIR}/models"
  _gz_path_prepend GZ_SIM_RESOURCE_PATH "${ARDUPILOT_GZ_DIR}/worlds"
fi

if [ -d "${ARDUPILOT_SITL_MODELS_DIR}/Gazebo" ]; then
  _gz_path_prepend GZ_SIM_RESOURCE_PATH "${ARDUPILOT_SITL_MODELS_DIR}/Gazebo/models"
  _gz_path_prepend GZ_SIM_RESOURCE_PATH "${ARDUPILOT_SITL_MODELS_DIR}/Gazebo/worlds"
fi

# Prefer FIRE models and worlds over the upstream resource directories.
if [ -d "${fire_gz_dir}" ]; then
  _gz_path_prepend GZ_SIM_RESOURCE_PATH "${fire_gz_dir}/models"
  _gz_path_prepend GZ_SIM_RESOURCE_PATH "${fire_gz_dir}/worlds"
fi

if [ -d "${fire_gz_plugin_lib}" ]; then
  _gz_path_prepend GZ_SIM_SYSTEM_PLUGIN_PATH "${fire_gz_plugin_lib}"
fi

export GZ_SIM_SYSTEM_PLUGIN_PATH
export GZ_SIM_RESOURCE_PATH
unset -f _gz_path_prepend
EOF

  echo ""
  echo "==> Wrote env snippet:"
  echo "    ${env_file}"
  echo ""
}

# --------------------------
# Main
# --------------------------
print_config

if [[ "${WITH_PX4}" == "true" ]]; then
  case "${PHASE}" in
    deps)
      px4_deps
      dds_deps
      ;;
    fetch)
      px4_fetch
      px4_gz_models_fetch
      px4_msgs_fetch
      dds_fetch
      ;;
    build)
      px4_build
      dds_build
      px4_msgs_build
      ;;
    env)
      px4_env
      px4_gz_models_env
      dds_env
      px4_msgs_env
      ;;
    all)
      px4_deps
      dds_deps
      px4_fetch
      px4_msgs_fetch
      dds_fetch
      px4_build
      dds_build
      px4_msgs_build
      px4_env
      px4_gz_models_env
      dds_env
      px4_msgs_env
      ;;
  esac
fi

if [[ "${WITH_PX4}" != "true" && "${WITH_PX4_MSGS}" == "true" ]]; then
  case "${PHASE}" in
    fetch|all)
      px4_msgs_fetch
      ;;
  esac
fi

if [[ "${WITH_ARDUPILOT}" == "true" ]]; then
  case "${PHASE}" in
    deps)
      ardupilot_deps
      ardupilot_gz_plugin_deps
      ;;
    fetch)
      ardupilot_fetch
      ardupilot_gz_plugin_fetch
      sitl_models_fetch
      ;;
    build)
      ardupilot_build
      ardupilot_gz_plugin_build
      fire_gz_plugins_build
      ;;
    env)
      ardupilot_env
      ardupilot_gz_plugin_env
      ;;
    all)
      ardupilot_deps
      ardupilot_gz_plugin_deps
      ardupilot_fetch
      ardupilot_gz_plugin_fetch
      sitl_models_fetch
      ardupilot_build
      ardupilot_gz_plugin_build
      fire_gz_plugins_build
      ardupilot_env
      ardupilot_gz_plugin_env
      ;;
  esac
fi

echo ""
echo "Autopilot installation DONE. PHASE=${PHASE}"