"""Built-in demo data selection for the telemetry MVP."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEMO_RULE_PATH = PROJECT_ROOT / "examples" / "rules.yaml"
DEMO_DATASET_KIND = "last_avg"
DEMO_DATASET_PATH = PROJECT_ROOT / "dataset" / DEMO_DATASET_KIND / "all_features_10_db_2022_12_29_165551_last_avg.csv"

# 只对 last_avg 这一类均值遥测特征做 MVP 演示。时间戳、label、diagnosis
# 不进入关联规则和相关性分析，避免产生没有业务意义的恒定项规则。
TELEMETRY_ANALYSIS_COLUMNS = [
    "1_battery_remaining_avg",
    "1_current_battery_avg",
    "1_voltage_battery_avg",
    "147_battery_remaining_avg",
    "147_current_battery_avg",
    "147_current_consumed_avg",
    "168_speed_avg",
    "182_altitude_avg",
    "241_vibration_x_avg",
    "241_vibration_y_avg",
    "241_vibration_z_avg",
    "29_press_abs_avg",
    "29_press_diff_avg",
    "29_temperature_avg",
    "33_alt_avg",
    "33_vx_avg",
    "33_vy_avg",
    "33_vz_avg",
]
