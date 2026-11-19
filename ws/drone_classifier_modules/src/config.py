from pathlib import Path

import numpy as np

# ==========================================
# 1. Directory & Path Settings
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / "cache"

# ==========================================
# 2. Dataset Settings
# ==========================================
SITL_FOLDER = "sitl_logs"
REAL_FLIGHT_FOLDERS = ["260527_flight_logs_1", "260527_flight_logs_2"]

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class FwConfig:
    name: str
    class_label: int
    sub_paths: List[str]
    ext: str
    color: str

def get_fw_configs(is_sitl: bool) -> List[FwConfig]:
    """Returns the firmware configurations based on whether the data is SITL or Real Flight."""
    if is_sitl:
        return [
            FwConfig('px4', 0, ['raw', ''], '.ulg', 'tab:green'),
            FwConfig('ardu', 1, ['raw/logs', 'logs'], '.bin', 'tab:orange')
        ]
    else:
        return [
            FwConfig('px4', 0, ['processed'], '.csv', 'tab:green'),
            FwConfig('ardu', 1, ['processed'], '.csv', 'tab:orange'),
            FwConfig('cogni', 2, ['processed'], '.csv', 'tab:purple')
        ]

def find_fw_dir(run_dir: Path, fw_name: str, sub_paths: List[str]) -> Optional[Path]:
    """Searches for the valid firmware directory by iterating through possible sub_paths."""
    for sub_path in sub_paths:
        candidate_dir = run_dir / f"{fw_name}_logs" / sub_path
        if candidate_dir.exists():
            return candidate_dir
    return None

# ==========================================
# 3. Feature Extraction Settings
# ==========================================
TARGET_FEATURES = ['XY-Accel', 'XY-Jerk', 'Curvature']

@dataclass
class FeatureDef:
    id: str
    plot_label: str

FEATURE_DEFINITIONS = [
    FeatureDef('Altitude', 'Altitude (m)'),
    FeatureDef('Heading',  'Heading (rad)'),
    FeatureDef('VZ',       'Z-Axis Velocity (m/s)'),
    FeatureDef('XY-Speed', 'XY-Plane Speed (m/s)'),
    FeatureDef('AZ',       'Z-Axis Acceleration (m/s²)'),
    FeatureDef('XY-Accel', 'XY-Plane Accel Norm (m/s²)'),
    FeatureDef('JZ',       'Z-Axis Jerk (m/s³)'),
    FeatureDef('XY-Jerk',  'XY-Plane Jerk Norm (m/s³)'),
    FeatureDef('Curvature','Curvature (1/m)'),
    FeatureDef('YawRate',  'Yaw rate (rad/s)'),
    FeatureDef('SlipRate', 'Slip Rate')
]

FEATURE_MAP = {feat.id: idx for idx, feat in enumerate(FEATURE_DEFINITIONS)}
WAVELET_NAME = 'db4'
WAVELET_LEVEL = 3

# ==========================================
# 4. HMM (Flight Segmenter) Settings
# ==========================================
HMM_PI = np.array([0.3, 0.3, 0.1, 0.1, 0.1, 0.1])
HMM_A = np.array([
    # Hov,  Tkf,  Lnd,  Str,   LT,   RT
    [0.70, 0.10, 0.10, 0.10, 0.00, 0.00], # Hover
    [0.10, 0.70, 0.00, 0.20, 0.00, 0.00], # Take-off
    [0.10, 0.00, 0.80, 0.10, 0.00, 0.00], # Landing
    [0.15, 0.00, 0.05, 0.30, 0.25, 0.25], # Straight
    [0.00, 0.00, 0.00, 0.20, 0.70, 0.10], # Left_Turn
    [0.00, 0.00, 0.00, 0.20, 0.10, 0.70]  # Right_Turn
])

# ==========================================
# 5. Model Training Hyperparameters
# ==========================================
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.001
