#!/usr/bin/env bash
# clean.sh — cleanup helper for Docker builds + local dev machines
#
# Features:
#  - Clean --help output
#  - Optional --debug enables `set -x`
#  - Safe defaults for local machines (does NOT wipe /tmp unless requested)
#  - Optional /tmp cleanup using extglob keep-pattern logic
#
# Notes:
#  - Removing /var/lib/apt/lists/* is great for Docker layers; on local machines it is safe,
#    but it forces `apt-get update` to re-download package lists next time.
#  - /tmp cleanup is OFF by default for local safety.

# --------------------------
# Defaults
# --------------------------
DEBUG="false"
DO_TMP="false"
KEEP_PATTERNS=()  # extglob patterns to KEEP in /tmp when --clean-tmp is enabled

help() {
  cat <<EOF
Usage:
  bash clean.sh [options]

Options:
  -h, --help         Show this help
  --debug            Enable command tracing (set -x)
  --clean-tmp        Also clean /tmp (OFF by default; safer for local machines)
  --keep PATTERN     When cleaning /tmp, keep files matching this extglob pattern.
                     Can be provided multiple times.
                     Example: --keep "clean.sh" --keep "*.sock"

Examples:
  # Safe default: apt cache + /var/lib/apt/lists cleanup only
  bash clean.sh

  # Include /tmp cleanup, keep this script and *.sock
  bash clean.sh --clean-tmp --keep "clean.sh" --keep "*.sock"

  # Debug
  bash clean.sh --debug --clean-tmp
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
    --clean-tmp) DO_TMP="true"; shift ;;
    --keep)
      [[ $# -ge 2 ]] || die "--keep requires a pattern (e.g., '*.sock')"
      KEEP_PATTERNS+=("$2")
      shift 2
      ;;
    *) die "Unknown option: $1 (use --help)" ;;
  esac
done

# --------------------------
# Strict mode
# --------------------------
set -Ee
trap 'echo "[clean.sh] ERROR line=$LINENO cmd=$BASH_COMMAND" >&2' ERR
[[ "${DEBUG}" == "true" ]] && set -x

echo "Cleaning APT cache..."
sudo apt-get clean
sudo apt-get autoremove --purge -y

# Docker-friendly size reduction; safe on local too (but forces next apt update to download lists again)
echo "Removing APT package lists..."
sudo rm -rf /var/lib/apt/lists/*

# --------------------------
# Optional /tmp cleanup
# --------------------------
if [[ "${DO_TMP}" == "true" ]]; then
  echo "Cleaning /tmp (best-effort)..."

  script_base="$(basename "$0")"
  if [[ ${#KEEP_PATTERNS[@]} -eq 0 ]]; then
    KEEP_PATTERNS=("${script_base}")
  fi

  # Always keep common runtime dirs/sockets that are often bind-mounted/readonly
  # (safe for Docker + local; avoids breaking GUI/X11 and some runtimes)
  KEEP_PATTERNS+=(
    ".X11-unix"
    ".dbus"
    "systemd-private-*"
  )

  [[ -d /tmp ]] || die "/tmp does not exist or is not a directory"

  # Build keep_joined="a|b|c"
  keep_joined=""
  for pat in "${KEEP_PATTERNS[@]}"; do
    pat="${pat#"${pat%%[![:space:]]*}"}"
    pat="${pat%"${pat##*[![:space:]]}"}"
    [[ -n "${pat}" ]] || continue
    if [[ -z "${keep_joined}" ]]; then
      keep_joined="${pat}"
    else
      keep_joined="${keep_joined}|${pat}"
    fi
  done

  # Enable extglob to compute targets
  shopt -s extglob nullglob dotglob
  exclude_pat="!(@(${keep_joined}))"
  targets=(/tmp/${exclude_pat})
  shopt -u extglob nullglob dotglob

  if [[ ${#targets[@]} -gt 0 ]]; then
    # best-effort remove: ignore permission/readonly failures
    sudo rm -rf -- "${targets[@]}" 2>/dev/null || true
  fi

  echo "Finished cleaning /tmp (kept: ${KEEP_PATTERNS[*]})."
else
  echo "Skipping /tmp cleanup (enable with --clean-tmp)."
fi

echo "Cleanup DONE."