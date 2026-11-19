#!/usr/bin/env bash
# extra.sh — install extra dev/tools + GeographicLib datasets + QGroundControl prereqs
#
# Phases:
#   --phase deps   : install system/package dependencies only
#   --phase fetch  : download external artifacts only
#   --phase env    : run runtime/user environment setup only
#   --phase all    : run deps + env + fetch (default)
#
# Intended workflow:
#   - Before docker build: fetch
#   - During docker image build: deps
#   - After entering container / runtime host setup: env
#
# Features:
#  - help
#  - debug enables `set -x`
#  - installs common utilities + Python deps
#  - installs MAVROS GeographicLib datasets
#  - installs QGroundControl prerequisites
#  - optionally downloads QGroundControl AppImage
#  - ensures NumPy version is compatible with MAVProxy (numpy<2)
#
# Notes:
#  - We intentionally do NOT run `apt-get upgrade` by default to preserve reproducibility.
#    Use --upgrade if you really want it.
#  - QGC may require logout/login after adding user to dialout group.
#  - In Docker image build, user/group changes are generally less useful than in runtime containers.
#  - MAVProxy itself is assumed to be installed elsewhere (e.g. autopilot.sh for ArduPilot deps).
#    This script only checks/fixes NumPy compatibility for MAVProxy-related usage.

# --------------------------
# Defaults
# --------------------------
DEBUG="false"                                 # --debug
PHASE="all"                                   # deps | fetch | env | all
DO_UPGRADE="false"                            # --upgrade
INSTALL_QGC="true"                            # --no-qgc

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${THIS_DIR}/.." && pwd)"

QGC_DIR="${PROJECT_ROOT}/tools/QGC"
QGC_URL="https://d176tv9ibo4jno.cloudfront.net/latest/QGroundControl-x86_64.AppImage"

# MAVProxy / NumPy compatibility controls
INSTALL_MAVPROXY_NUMPY_FIX="true"             # --no-mavproxy-numpy-fix
MAVPROXY_INSTALL_SCOPE="user"                 # user | system
MAVPROXY_NUMPY_SPEC="numpy<2"                 # pin for MAVProxy compatibility

help() {
  cat <<EOF
Usage:
  bash extra.sh [options]

Options:
  -h, --help            Show this help and exit (no command tracing)
  --debug               Enable command tracing (set -x)

  --phase PHASE         What to run: deps | fetch | env | all
                        (default: ${PHASE})
                        deps  = install apt/pip/system dependencies only
                        fetch = download external artifacts only
                        env   = runtime/user environment setup only
                        all   = run deps + env + fetch

  --upgrade             Run apt-get upgrade (default: off)
  --no-qgc              Skip QGroundControl prerequisites + AppImage download

  --project-root PATH   Project root (default: ${PROJECT_ROOT})
  --qgc-dir PATH        Where to put QGC AppImage (default: ${QGC_DIR})
  --qgc-url URL         Direct URL to QGroundControl AppImage
                        (default: ${QGC_URL})

  --no-mavproxy-numpy-fix
                        Skip NumPy compatibility check/fix for MAVProxy
  --mavproxy-scope S    NumPy install scope for MAVProxy compatibility: user | system
                        (default: ${MAVPROXY_INSTALL_SCOPE})
  --mavproxy-numpy SPEC NumPy version spec for MAVProxy compatibility
                        (default: ${MAVPROXY_NUMPY_SPEC})

Examples:
  # Dockerfile: dependencies only
  bash extra.sh --phase deps

  # Download QGC before docker build
  bash extra.sh --phase fetch

  # Runtime in container: user/group setup only
  bash extra.sh --phase env

  # Full install
  bash extra.sh --phase all

  # Explicit project root
  bash extra.sh --project-root /home/user/ws/flightstack_sim
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

# Prevent sourcing
(return 0 2>/dev/null) && {
  echo "Do not source this script. Run: bash $0" >&2
  return 1
}

# --------------------------
# Parse args (before set -x)
# --------------------------
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
    --phase)
      [[ $# -ge 2 ]] || die "--phase requires a value (deps|fetch|env|all)"
      PHASE="$2"
      shift 2
      ;;
    --upgrade)
      DO_UPGRADE="true"
      shift
      ;;
    --no-qgc)
      INSTALL_QGC="false"
      shift
      ;;
    --project-root)
      [[ $# -ge 2 ]] || die "--project-root requires a path"
      PROJECT_ROOT="$2"
      QGC_DIR="${PROJECT_ROOT}/tools/QGC"
      shift 2
      ;;
    --qgc-dir)
      [[ $# -ge 2 ]] || die "--qgc-dir requires a path"
      QGC_DIR="$2"
      shift 2
      ;;
    --qgc-url)
      [[ $# -ge 2 ]] || die "--qgc-url requires a URL"
      QGC_URL="$2"
      shift 2
      ;;
    --no-mavproxy-numpy-fix)
      INSTALL_MAVPROXY_NUMPY_FIX="false"
      shift
      ;;
    --mavproxy-scope)
      [[ $# -ge 2 ]] || die "--mavproxy-scope requires a value (user|system)"
      MAVPROXY_INSTALL_SCOPE="$2"
      shift 2
      ;;
    --mavproxy-numpy)
      [[ $# -ge 2 ]] || die "--mavproxy-numpy requires a version spec"
      MAVPROXY_NUMPY_SPEC="$2"
      shift 2
      ;;
    *)
      die "Unknown option: $1 (use --help)"
      ;;
  esac
done

# --------------------------
# Validate
# --------------------------
case "${PHASE}" in
  deps|fetch|env|all) ;;
  *)
    die "Invalid --phase: ${PHASE} (expected: deps|fetch|env|all)"
    ;;
esac

if [[ "${MAVPROXY_INSTALL_SCOPE}" != "user" && "${MAVPROXY_INSTALL_SCOPE}" != "system" ]]; then
  die "Invalid --mavproxy-scope: ${MAVPROXY_INSTALL_SCOPE} (expected: user|system)"
fi

# --------------------------
# Strict mode
# --------------------------
set -Eeuo pipefail
trap 'echo "[extra.sh] ERROR line=$LINENO cmd=$BASH_COMMAND" >&2' ERR

if [[ "${DEBUG}" == "true" ]]; then
  set -x
fi

# --------------------------
# Helpers
# --------------------------
apt_update() {
  sudo apt-get -y update
  if [[ "${DO_UPGRADE}" == "true" ]]; then
    sudo DEBIAN_FRONTEND=noninteractive apt-get -y upgrade
  else
    echo "==> Skipping apt-get upgrade (use --upgrade to enable)"
  fi
}

python_has_numpy() {
  python3 - <<'PY' >/dev/null 2>&1
import numpy  # noqa: F401
PY
}

get_numpy_version() {
  python3 - <<'PY'
import numpy
print(numpy.__version__)
PY
}

numpy_major_version() {
  python3 - <<'PY'
import numpy
v = numpy.__version__.split(".")[0]
try:
    print(int(v))
except Exception:
    print(-1)
PY
}

detect_ros_distro() {
  if [[ -n "${ROS_DISTRO:-}" ]]; then
    echo "${ROS_DISTRO}"
    return
  fi

  local ros_dirs=(/opt/ros/*)
  if [[ -d "${ros_dirs[0]}" ]]; then
    basename "${ros_dirs[0]}"
  else
    echo "humble"  # fallback
  fi
}

# --------------------------
# Phase: deps
# --------------------------
# deps: common packages + tools + Python deps
install_common_packages() {
  echo ""
  echo "==> Installing extra tools/dependencies (apt)"
  echo ""

  apt_update

  sudo DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    htop \
    iproute2 \
    lcov \
    mesa-utils \
    openbox \
    python3-jinja2 \
    python3-numpy \
    python3-pip \
    python3-xdg \
    python3-xmltodict \
    qt5dxcb-plugin \
    screen \
    terminator \
    vim \
    libasio-dev \
    qtbase5-dev \
    qtbase5-dev-tools \
    qtchooser \
    qt5-qmake \
    libqt5opengl5-dev \
    libqt5svg5-dev \
    qml-module-qtquick2 \
    qml-module-qtquick-controls \
    qml-module-qtquick-controls2 \
    qml-module-qtquick-layouts \
    qml-module-qtgraphicaleffects \
    ros-$(detect_ros_distro)-rosbag2 \
    ros-$(detect_ros_distro)-rosbag2-storage-mcap

  echo ""
  echo "==> Installing extra Python packages (pip)"
  echo ""
  python3 -m pip install --upgrade pip
  python3 -m pip install pykwalify
}

# deps: GeographicLib datasets
install_geographiclib_datasets() {
  echo ""
  echo "==> Installing MAVROS GeographicLib datasets"
  echo ""

  local tmp="/tmp/install_geographiclib_datasets.sh"
  wget -q "https://raw.githubusercontent.com/mavlink/mavros/master/mavros/scripts/install_geographiclib_datasets.sh" -O "${tmp}"
  sudo bash "${tmp}"
  rm -f "${tmp}"
}

# deps: QGC system prerequisites only
install_qgc_prereqs() {
  if [[ "${INSTALL_QGC}" != "true" ]]; then
    echo "==> Skipping QGroundControl prerequisites (--no-qgc)"
    return 0
  fi

  echo ""
  echo "==> Installing QGroundControl system prerequisites"
  echo ""

  sudo apt-get remove -y modemmanager || true

  apt_update
  sudo DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-libav \
    gstreamer1.0-gl \
    libfuse2 \
    libxcb-xinerama0 \
    libxkbcommon-x11-0 \
    libxcb-cursor-dev
}

# deps: ensure NumPy compatibility for MAVProxy
ensure_numpy_for_mavproxy() {
  if [[ "${INSTALL_MAVPROXY_NUMPY_FIX}" != "true" ]]; then
    echo "==> Skipping MAVProxy NumPy compatibility fix (--no-mavproxy-numpy-fix)"
    return 0
  fi

  echo ""
  echo "==> Checking NumPy compatibility for MAVProxy"
  echo "    scope=${MAVPROXY_INSTALL_SCOPE}"
  echo "    required spec=${MAVPROXY_NUMPY_SPEC}"
  echo ""

  local need_fix="false"

  if python_has_numpy; then
    local current_ver
    current_ver="$(get_numpy_version)"
    local current_major
    current_major="$(numpy_major_version)"

    echo "Detected NumPy version: ${current_ver}"

    if [[ "${current_major}" -ge 2 ]]; then
      echo "NumPy major version is >= 2, which may break MAVProxy-related modules."
      need_fix="true"
    else
      echo "NumPy version is already compatible with MAVProxy."
    fi
  else
    echo "NumPy is not currently importable from python3."
    need_fix="true"
  fi

  if [[ "${need_fix}" != "true" ]]; then
    return 0
  fi

  echo ""
  echo "==> Installing compatible NumPy for MAVProxy"
  echo ""

  if [[ "${MAVPROXY_INSTALL_SCOPE}" == "system" ]]; then
    sudo python3 -m pip install --upgrade pip setuptools wheel --break-system-packages
    sudo python3 -m pip install --upgrade "${MAVPROXY_NUMPY_SPEC}" --break-system-packages
  else
    python3 -m pip install --upgrade --user pip setuptools wheel
    python3 -m pip install --upgrade --user "${MAVPROXY_NUMPY_SPEC}"
  fi
}

verify_numpy_for_mavproxy() {
  if [[ "${INSTALL_MAVPROXY_NUMPY_FIX}" != "true" ]]; then
    return 0
  fi

  echo ""
  echo "==> Verifying NumPy compatibility for MAVProxy"
  echo ""

  python3 - <<'PY'
import sys
print("Python executable:", sys.executable)
print("Python version   :", sys.version.split()[0])

import numpy
print("NumPy version    :", numpy.__version__)

major = int(numpy.__version__.split(".")[0])
assert major < 2, f"NumPy major version must be < 2 for MAVProxy compatibility, got {numpy.__version__}"

try:
    import MAVProxy  # noqa: F401
    print("MAVProxy import  : OK")
except Exception as e:
    print(f"MAVProxy import  : NOT VERIFIED ({e})")
PY

  echo ""
  echo "NumPy compatibility check finished."
  echo ""
}


# --------------------------
# Phase: fetch
# --------------------------
# fetch: download QGC AppImage
download_qgc_appimage() {
  if [[ "${INSTALL_QGC}" != "true" ]]; then
    echo "==> Skipping QGroundControl AppImage download (--no-qgc)"
    return 0
  fi

  mkdir -p "${QGC_DIR}"

  local out="${QGC_DIR}/QGroundControl-x86_64.AppImage"

  if [[ -f "${out}" ]]; then
    echo ""
    echo "==> QGroundControl AppImage already exists"
    echo "    ${out}"
    echo "    Skipping download."
    echo ""
    return 0
  fi

  echo ""
  echo "==> Downloading QGroundControl AppImage"
  echo "    URL: ${QGC_URL}"
  echo "    OUT: ${out}"
  echo ""

  wget -O "${out}" "${QGC_URL}"
  chmod +x "${out}"

  echo ""
  echo "QGroundControl downloaded to:"
  echo "  ${out}"
  echo ""
  echo "To run:"
  echo "  ${out}"
  echo ""
}

# --------------------------
# Phase: env
# --------------------------
# env: QGroundControl user access setup (dialout group)
setup_qgc_user_access() {
  if [[ "${INSTALL_QGC}" != "true" ]]; then
    echo "==> Skipping QGroundControl user setup (--no-qgc)"
    return 0
  fi

  echo ""
  echo "==> Configuring QGroundControl user access"
  echo ""

  if [[ -n "${USER:-}" ]]; then
    sudo usermod -a -G dialout "${USER}" || true
    echo "Added user '${USER}' to dialout group (best-effort)."
  else
    echo "WARNING: USER is not set; skipping dialout group update."
  fi

  echo ""
  echo "IMPORTANT: You may need logout/login (or a new shell/session) for dialout group changes to take effect."
  echo ""
}

# env: extract AppImage so FUSE is not required
extract_qgc_appimage() {

  if [[ "${INSTALL_QGC}" != "true" ]]; then
    return 0
  fi

  local appimg="${QGC_DIR}/QGroundControl-x86_64.AppImage"
  local extracted="${QGC_DIR}/squashfs-root"

  if [[ ! -f "${appimg}" ]]; then
    echo "QGC AppImage not found:"
    echo "  ${appimg}"
    echo "Run --phase fetch first."
    return 1
  fi

  if [[ -d "${extracted}" ]]; then
    echo "==> QGC AppImage already extracted"
    return 0
  fi

  echo ""
  echo "==> Extracting QGroundControl AppImage (FUSE-free mode)"
  echo ""

  (
    cd "${QGC_DIR}"
    "${appimg}" --appimage-extract
  )

  echo ""
  echo "Run QGC with:"
  echo "  ${extracted}/AppRun"
  echo ""
}

# --------------------------
# Main
# --------------------------
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "PHASE=${PHASE}"
echo "DO_UPGRADE=${DO_UPGRADE}"
echo "INSTALL_QGC=${INSTALL_QGC}"
echo "QGC_DIR=${QGC_DIR}"
echo "QGC_URL=${QGC_URL}"
echo "INSTALL_MAVPROXY_NUMPY_FIX=${INSTALL_MAVPROXY_NUMPY_FIX}"
echo "MAVPROXY_INSTALL_SCOPE=${MAVPROXY_INSTALL_SCOPE}"
echo "MAVPROXY_NUMPY_SPEC=${MAVPROXY_NUMPY_SPEC}"

case "${PHASE}" in
  deps)
    install_common_packages
    install_geographiclib_datasets
    install_qgc_prereqs
    ensure_numpy_for_mavproxy
    verify_numpy_for_mavproxy
    ;;
  fetch)
    download_qgc_appimage
    ;;
  env)
    setup_qgc_user_access
    extract_qgc_appimage
    ;;
  all)
    install_common_packages
    install_geographiclib_datasets
    install_qgc_prereqs
    ensure_numpy_for_mavproxy
    verify_numpy_for_mavproxy
    setup_qgc_user_access
    download_qgc_appimage
    extract_qgc_appimage
    ;;
esac

echo ""
echo "Extra packages installation DONE. PHASE=${PHASE}"