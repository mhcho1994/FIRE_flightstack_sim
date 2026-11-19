#!/bin/bash
# base.sh — install fundamental build/dev packages for PX4/ArduPilot/ROS2/Gazebo
#
# Features:
#  - help
#  - debug enables `set -x`

# --------------------------
# Defaults
# --------------------------
DEBUG="false"       # --debug
DO_UPGRADE="false"  # --upgrade (off by default to preserve reproducibility)

help() {
  cat <<'EOF'
Usage:
  bash base.sh [options]

Options:
  -h, --help     Show this help and exit (no command tracing)
  --debug        Enable command tracing (set -x)

Examples:
  bash base.sh
  bash base.sh --debug
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

# Prevent sourcing (this script should be executed, not sourced)
(return 0 2>/dev/null) && { echo "Do not source this script. Run: bash $0" >&2; return 1; }

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
    *)
      die "Unknown option: $1 (use --help)"
      ;;
  esac
done

# --------------------------
# Strict mode (after help)
# --------------------------
set -Ee
trap 'echo "[base.sh] ERROR line=$LINENO cmd=$BASH_COMMAND" >&2' ERR
set -e
if [[ "${DEBUG}" == "true" ]]; then
  set -x
fi

# --------------------------
# APT update / optional upgrade
# --------------------------
sudo apt-get -y update

# Warning: Do NOT run `apt upgrade` here to if you want to strongly preserve Docker reproducibility.
# Running upgrade makes the image non-deterministic over time.
if [[ "${DO_UPGRADE}" == "true" ]]; then
  sudo apt-get -y upgrade
fi

# ------------------------------------------------------------------------------
# Fundamental packages required for:
#   - C/C++ compilation (PX4, ArduPilot, Gazebo plugins)
#   - ROS 2 workspace builds
#   - MAVLink-related tools (via pip/ros deps later)
#   - General development inside the container
# ------------------------------------------------------------------------------
# Packages to install (use an array to avoid line-continuation whitespace bugs)
PKGS=(
  bash-completion # Shell completion for better CLI usability
  build-essential # gcc, g++, make (core compilation toolchain)
  ca-certificates # Required for HTTPS downloads
  ccache # Speeds up repeated C/C++ builds
  cmake # Required by PX4, Gazebo, many C++ projects
  curl # HTTP requests (repo keys, etc.)
  git # Clone PX4, ArduPilot, plugins
  gnupg # Required for repository key management
  lsb-release # Detect Ubuntu codename (jammy, etc.)
  ninja-build # Faster build backend (used by meson/cmake)
  pkg-config # Library discovery during compilation
  python3-pip # Python package manager
  python3-setuptools # Python build utilities
  python3-venv # Python virtual environments
  python3-wheel # Python wheel builds
  software-properties-common # add-apt-repository
  unzip
  wget
  zip
)
sudo apt-get -y --no-install-recommends install "${PKGS[@]}"

# Upgrade pip inside container to avoid compatibility issues
python3 -m pip install --upgrade pip

# --------------------------
# Notes / next steps
# --------------------------
echo "Base Packages Installation DONE."