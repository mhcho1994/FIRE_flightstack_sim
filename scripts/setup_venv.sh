#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# setup_venv.sh
#     This script sets up a Python virtual environment (venv) for trajectory 
#     segmentation and classification experiments within the FLIGHTSTACK_SIM project.
#
#     It performs the following steps:
#        1. Creates a virtual environment (if it does not already exist)
#        2. Activates the virtual environment
#        3. Upgrades pip
#        4. Installs required Python packages for:
#            - Log parsing (pyulog, pybinlog)
#            - Configuration handling (pyyaml)
#            - Mavlink handling (pymavlink)
#            - Data processing (pandas, numpy)
#            - Visualization (matplotlib)
#            - Classical ML (scikit-learn)
#            - Deep learning (torch, torchvision, torchaudio)
#
#     Note:
#       - This script is executed from the root directory of FLIGHTSTACK_SIM
#
#     Example:
#       bash setup_venv.sh
# -----------------------------------------------------------------------------

echo "=== Starting automatic setup virtual environment for segmentation/classification experiments ==="

# ------------------------------------------------------------------------------
# Step 1: Create virtual environment if it does not exist
# ------------------------------------------------------------------------------
if [ ! -d ".venv" ]; then
    echo ">> Creating virtual environment (venv)..."
    python3 -m venv .venv
else
    echo ">> Virtual environment (./venv) already exists. Skipping creation."
fi

# ------------------------------------------------------------------------------
# Step 2: Activate the virtual environment
# ------------------------------------------------------------------------------
echo ">> Activating virtual environment..."
source .venv/bin/activate

# ------------------------------------------------------------------------------
# Step 3: Install required Python packages
#   - First upgrade pip to the latest version
#   - Then install dependencies for ML, logging, and simulation analysis
# ------------------------------------------------------------------------------
echo ">> Installing required packages..."

# Upgrade pip to avoid compatibility issues
pip install --upgrade pip

# Log parsing, scenario parsing and mavlink handling
pip install pyyaml pyulog pybinlog pymavlink

# Data processing, visulization
pip install pandas numpy matplotlib

# Classical ML and deep learning
pip install scikit-learn torch torchvision torchaudio

# Finish setup
echo ""
echo "Virtual environment setup DONE."
echo "To activate the virtual environment, run: source ./venv/bin/activate"
echo "To enable from auto-activation in vscode, change \"python.terminal.activateEnvironment\": true in settings.json"