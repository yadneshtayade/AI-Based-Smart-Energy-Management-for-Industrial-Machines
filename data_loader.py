"""
data_loader.py – Load and cache SKAB dataset for the web UI.
Downloads from GitHub if not already cached locally.
"""

import io
import logging
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

DATA_CACHE_DIR = Path(__file__).parent / "data" / "skab"
SKAB_REPO = "https://github.com/waico/SKAB.git"


# ─────────────────────────── SKAB Loader ────────────────────────────────────

def read_skab_csv(path) -> pd.DataFrame:
    """Read a single SKAB CSV (auto-detect semicolon vs comma separator)."""
    try:
        df = pd.read_csv(path, sep=";", engine="python")
        if df.shape[1] <= 2:
            df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _apply_rename(df: pd.DataFrame) -> pd.DataFrame:
    """Apply canonical SKAB column renames."""
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


@st.cache_data(show_spinner=False)
def load_skab_dataset() -> Optional[pd.DataFrame]:
    """
    Load all SKAB CSV experiments into one DataFrame.
    Tries (in order):
      1. Local data/skab/ directory
      2. Git clone from GitHub (if git is available)
      3. Returns None with an error message
    """
    # 1. Try local cache
    local_csvs = list(DATA_CACHE_DIR.rglob("*.csv")) if DATA_CACHE_DIR.exists() else []
    if local_csvs:
        return _load_from_csv_list(local_csvs)

    # 2. Try git clone
    clone_target = Path(__file__).parent / "data" / "SKAB_repo"
    if not clone_target.exists():
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", SKAB_REPO, str(clone_target)],
                capture_output=True, timeout=60,
            )
        except Exception as e:
            logger.warning("Git clone failed: %s", e)

    if clone_target.exists():
        repo_csvs = list((clone_target / "data").rglob("*.csv"))
        if repo_csvs:
            return _load_from_csv_list(repo_csvs)

    return None


def _load_from_csv_list(csv_paths) -> Optional[pd.DataFrame]:
    frames = []
    for path in csv_paths:
        try:
            part = read_skab_csv(path)
            lower_cols = {c.lower() for c in part.columns}
            if "anomaly" not in lower_cols:
                continue
            part = _apply_rename(part)
            part["anomaly"] = pd.to_numeric(part["anomaly"], errors="coerce")
            part = part[part["anomaly"].isin([0, 1])].copy()
            part["anomaly"] = part["anomaly"].astype(int)
            part["experiment_id"] = Path(path).stem
            part["source_file"] = str(path)
            frames.append(part)
        except Exception as e:
            logger.warning("Skipped %s: %s", path, e)

    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True, sort=False)
    return df


def load_uploaded_csv(file_buffer) -> Optional[pd.DataFrame]:
    """Load a user-uploaded CSV file."""
    try:
        # Try semicolon first
        df = pd.read_csv(file_buffer, sep=";", engine="python")
        if df.shape[1] <= 2:
            file_buffer.seek(0)
            df = pd.read_csv(file_buffer, sep=None, engine="python")
    except Exception:
        file_buffer.seek(0)
        df = pd.read_csv(file_buffer, sep=None, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    df = _apply_rename(df)
    return df


def get_dataset_stats(df: pd.DataFrame) -> dict:
    """Compute summary statistics for the dataset."""
    total = len(df)
    normal = int((df["anomaly"] == 0).sum()) if "anomaly" in df.columns else 0
    anomalies = int((df["anomaly"] == 1).sum()) if "anomaly" in df.columns else 0
    anomaly_pct = (anomalies / total * 100) if total > 0 else 0.0

    experiments = df["experiment_id"].nunique() if "experiment_id" in df.columns else 1

    base_sensors = [
        "Vibration1", "Vibration2", "Current", "Pressure",
        "Temperature", "Thermocouple", "Voltage", "FlowRate",
    ]
    available_sensors = [s for s in base_sensors if s in df.columns]

    return {
        "total_records": total,
        "normal_records": normal,
        "anomaly_records": anomalies,
        "anomaly_percentage": round(anomaly_pct, 2),
        "num_experiments": experiments,
        "available_sensors": available_sensors,
    }


def sample_records(df: pd.DataFrame, n: int = 50, balanced: bool = True) -> pd.DataFrame:
    """Sample n records, optionally balanced between normal and anomaly."""
    if "anomaly" not in df.columns or not balanced:
        return df.sample(min(n, len(df))).reset_index(drop=True)

    half = n // 2
    normal_df = df[df["anomaly"] == 0]
    anomaly_df = df[df["anomaly"] == 1]

    n_normal = min(half, len(normal_df))
    n_anomaly = min(n - n_normal, len(anomaly_df))

    sampled = pd.concat([
        normal_df.sample(n_normal),
        anomaly_df.sample(n_anomaly),
    ], ignore_index=True).sample(frac=1).reset_index(drop=True)

    return sampled
