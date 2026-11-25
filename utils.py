"""
utils.py – Common utility functions for Smart Industrial Energy AI
"""

import os
import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# ─────────────────────────── SKAB Column Renaming ───────────────────────────

RENAME_MAP_RULES = {
    "anomaly": "anomaly",
    "changepoint": "changepoint",
    "datetime": "datetime",
}


def rename_skab_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply canonical SKAB column renames to a DataFrame."""
    rename_map = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl == "anomaly":
            rename_map[c] = "anomaly"
        elif cl == "changepoint":
            rename_map[c] = "changepoint"
        elif cl == "datetime":
            rename_map[c] = "datetime"
        elif "flow" in cl or "raterms" in cl:
            rename_map[c] = "FlowRate"
        elif cl == "current":
            rename_map[c] = "Current"
        elif cl == "voltage":
            rename_map[c] = "Voltage"
        elif cl == "pressure":
            rename_map[c] = "Pressure"
        elif cl == "temperature":
            rename_map[c] = "Temperature"
        elif cl == "thermocouple":
            rename_map[c] = "Thermocouple"
        elif "accelerometer1" in cl:
            rename_map[c] = "Vibration1"
        elif "accelerometer2" in cl:
            rename_map[c] = "Vibration2"
    return df.rename(columns=rename_map)


# ─────────────────────── Feature Engineering (must match training) ───────────

BASE_SENSOR_FEATURES = [
    "Vibration1", "Vibration2", "Current", "Pressure",
    "Temperature", "Thermocouple", "Voltage", "FlowRate",
]


def engineer_features(df: pd.DataFrame, base_features=None) -> pd.DataFrame:
    """
    Add diff1, deviation, and rolling_std features for each base sensor column.
    Groups by experiment_id if present, otherwise treats whole df as one group.
    Must exactly mirror the training feature engineering.
    """
    if base_features is None:
        base_features = [c for c in BASE_SENSOR_FEATURES if c in df.columns]

    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    for c in base_features:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "experiment_id" in df.columns:
        df = df.sort_values(["experiment_id"])
        grouped_col = "experiment_id"
    else:
        df["_dummy_group"] = "all"
        grouped_col = "_dummy_group"

    for col in base_features:
        grouped = df.groupby(grouped_col, sort=False)[col]
        df[f"{col}_diff1"] = grouped.diff().fillna(0)
        rolling_mean = grouped.transform(
            lambda s: s.rolling(window=5, min_periods=1).mean()
        )
        rolling_std = grouped.transform(
            lambda s: s.rolling(window=5, min_periods=2).std()
        ).fillna(0)
        df[f"{col}_deviation"] = df[col] - rolling_mean
        df[f"{col}_rolling_std"] = rolling_std

    if "_dummy_group" in df.columns:
        df = df.drop(columns=["_dummy_group"])

    return df


def build_feature_list(base_features=None):
    """Return the full ordered list of model features matching training."""
    if base_features is None:
        base_features = BASE_SENSOR_FEATURES
    return base_features + [
        f"{c}_{suffix}"
        for c in base_features
        for suffix in ("diff1", "deviation", "rolling_std")
    ]


# ─────────────────────────── JSON / Validation ──────────────────────────────

def safe_json_loads(text: str):
    """Try to parse JSON; return None on failure."""
    try:
        # Strip markdown code fences if present
        text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
        return json.loads(text)
    except Exception:
        return None


def validate_report_json(report: dict) -> tuple:
    """
    Validate that the generated report has required fields.
    Returns (is_valid: bool, missing_fields: list).
    """
    required = [
        "machine_id", "health_status", "sensor_readings",
        "anomalies_detected", "ensemble_score", "risk_severity",
        "recommended_actions",
    ]
    missing = [f for f in required if f not in report]
    return (len(missing) == 0, missing)


# ─────────────────────────── Sensor Display Names ───────────────────────────

SENSOR_DISPLAY = {
    "Vibration1": "Vibration 1 (Accel1 RMS)",
    "Vibration2": "Vibration 2 (Accel2 RMS)",
    "Current": "Motor Current (A)",
    "Pressure": "System Pressure (bar)",
    "Temperature": "Fluid Temperature (°C)",
    "Thermocouple": "Thermocouple (°C)",
    "Voltage": "Supply Voltage (V)",
    "FlowRate": "Volume Flow Rate (L/min)",
}


def sensor_label(col: str) -> str:
    return SENSOR_DISPLAY.get(col, col)


# ─────────────────────────── Status / Health ────────────────────────────────

def health_label(ensemble_score: float, threshold: float) -> str:
    if ensemble_score < threshold - 0.15:
        return "NORMAL"
    elif ensemble_score < threshold:
        return "WARNING"
    else:
        return "ANOMALY"


def health_color(label: str) -> str:
    return {"NORMAL": "#00ff88", "WARNING": "#ffaa00", "ANOMALY": "#ff4444"}.get(label, "#cccccc")


# ─────────────────────────── Misc ───────────────────────────────────────────

def format_timestamp(ts=None) -> str:
    if ts is None:
        ts = datetime.now()
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def project_root() -> Path:
    return Path(__file__).parent
