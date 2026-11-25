"""
model_loader.py – Load trained models and preprocessing objects from the
exported model bundle (skab_model_bundle.pkl or individual .pkl files).

Expected bundle keys (from the Colab notebook):
  - imputer          : sklearn SimpleImputer
  - scaler           : sklearn StandardScaler
  - iso_forest       : trained IsolationForest
  - knn_model        : trained NearestNeighbors (KNN)
  - pca_model        : trained PCA
  - autoencoder      : PyTorch Autoencoder state dict OR nn.Module
  - feature_names    : list[str]  – ordered model features (32 features)
  - base_features    : list[str]  – 8 base sensor columns
  - calibration_thresholds : dict  – 99th-pct normal-train scores per model
  - decision_thresholds    : dict  – validation-tuned thresholds per model
"""

import os
import pickle
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import streamlit as st

logger = logging.getLogger(__name__)

# ── Where to look for the bundle ────────────────────────────────────────────
MODEL_DIR = Path(__file__).parent / "models"
BUNDLE_PATH = MODEL_DIR / "skab_model_bundle.pkl"

# ── Fallback: individual files ───────────────────────────────────────────────
INDIVIDUAL_FILES = {
    "imputer":               MODEL_DIR / "imputer.pkl",
    "scaler":                MODEL_DIR / "scaler.pkl",
    "iso_forest":            MODEL_DIR / "iso_forest.pkl",
    "knn_model":             MODEL_DIR / "knn_model.pkl",
    "pca_model":             MODEL_DIR / "pca_model.pkl",
    "autoencoder":           MODEL_DIR / "autoencoder.pkl",
    "feature_names":         MODEL_DIR / "feature_names.pkl",
    "base_features":         MODEL_DIR / "base_features.pkl",
    "calibration_thresholds":MODEL_DIR / "calibration_thresholds.pkl",
    "decision_thresholds":   MODEL_DIR / "decision_thresholds.pkl",
}

# ── Default fallback values (used when bundle is missing) ──────────────────
DEFAULT_BASE_FEATURES = [
    "Vibration1", "Vibration2", "Current", "Pressure",
    "Temperature", "Thermocouple", "Voltage", "FlowRate",
]

DEFAULT_FEATURE_NAMES = DEFAULT_BASE_FEATURES + [
    f"{c}_{suffix}"
    for c in DEFAULT_BASE_FEATURES
    for suffix in ("diff1", "deviation", "rolling_std")
]

DEFAULT_CALIBRATION = {
    "IsolationForest": 0.0671,
    "KNN":             5.3867,
    "PCA":             0.3934,
    "Autoencoder":     0.8712,
}

DEFAULT_DECISION_THRESHOLDS = {
    "IsolationForest":    -0.502943,
    "KNN":                 0.616420,
    "PCA":                 0.324694,
    "Autoencoder":         0.380919,
    "EqualWeightEnsemble": 0.337992,
}


# ── Loader ───────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model_bundle() -> Dict[str, Any]:
    """
    Load the model bundle.  Returns a dict with all required keys.
    If no bundle is found, returns a dict with None models and defaults
    for metadata – the UI will display a clear warning.
    """
    bundle: Dict[str, Any] = {
        "imputer": None,
        "scaler": None,
        "iso_forest": None,
        "knn_model": None,
        "pca_model": None,
        "autoencoder": None,
        "feature_names": DEFAULT_FEATURE_NAMES,
        "base_features": DEFAULT_BASE_FEATURES,
        "calibration_thresholds": DEFAULT_CALIBRATION,
        "decision_thresholds": DEFAULT_DECISION_THRESHOLDS,
        "loaded": False,
        "source": "none",
    }

    # 1. Try the single bundle file first
    if BUNDLE_PATH.exists():
        try:
            with open(BUNDLE_PATH, "rb") as f:
                data = pickle.load(f)
            bundle.update(data)
            bundle["loaded"] = True
            bundle["source"] = str(BUNDLE_PATH)
            logger.info("Model bundle loaded from %s", BUNDLE_PATH)
            return bundle
        except Exception as e:
            logger.warning("Failed to load bundle %s: %s", BUNDLE_PATH, e)

    # 2. Try individual files
    all_found = True
    for key, path in INDIVIDUAL_FILES.items():
        if path.exists():
            try:
                with open(path, "rb") as f:
                    bundle[key] = pickle.load(f)
            except Exception as e:
                logger.warning("Failed to load %s: %s", path, e)
                all_found = False
        else:
            all_found = False

    if any(bundle[k] is not None for k in ["iso_forest", "knn_model", "pca_model"]):
        bundle["loaded"] = True
        bundle["source"] = "individual files"
        logger.info("Models loaded from individual files in %s", MODEL_DIR)
    else:
        logger.warning("No model files found. Running in demo mode.")

    return bundle


def models_available(bundle: Dict[str, Any]) -> bool:
    return bundle.get("loaded", False)


def get_missing_files() -> list:
    """Return a list of model files that are missing."""
    missing = []
    if not BUNDLE_PATH.exists():
        missing.append(str(BUNDLE_PATH.name))
        for key, path in INDIVIDUAL_FILES.items():
            if not path.exists():
                missing.append(str(path.name))
    return missing
