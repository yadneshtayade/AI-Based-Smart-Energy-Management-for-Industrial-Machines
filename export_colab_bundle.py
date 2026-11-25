"""
export_colab_bundle.py – Helper script / Colab code snippet.

Run this snippet in your Google Colab notebook right after running Section 7/8
(Ensemble & Evaluation) to export all trained models and preprocessing artifacts
into a single portable bundle file: 'skab_model_bundle.pkl'.

Download 'skab_model_bundle.pkl' from Colab and place it in the 'models/' directory
of this Streamlit project.
"""

import pickle
from pathlib import Path

# In Google Colab, run this block in a cell:
"""
import pickle

bundle = {
    "imputer": imputer,
    "scaler": scaler,
    "iso_forest": iso_forest,
    "knn_model": knn_model,
    "pca_model": pca_model,
    "autoencoder": autoencoder.state_dict() if hasattr(autoencoder, "state_dict") else autoencoder,
    "feature_names": numeric_features,
    "base_features": base_features,
    "calibration_thresholds": calibration_thresholds,
    "decision_thresholds": decision_thresholds,
}

with open("skab_model_bundle.pkl", "wb") as f:
    pickle.dump(bundle, f)

print("Saved skab_model_bundle.pkl successfully!")
from google.colab import files
files.download("skab_model_bundle.pkl")
"""

def generate_local_stub_bundle():
    """Generates a starter stub bundle if the user wants to test offline before Colab export."""
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import NearestNeighbors
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    import numpy as np
    import torch
    import torch.nn as nn

    base_features = [
        "Vibration1", "Vibration2", "Current", "Pressure",
        "Temperature", "Thermocouple", "Voltage", "FlowRate"
    ]
    numeric_features = base_features + [
        f"{c}_{s}" for c in base_features for s in ("diff1", "deviation", "rolling_std")
    ]

    # Create dummy synthetic data to fit standard objects
    np.random.seed(42)
    dummy_data = np.random.randn(200, len(numeric_features))
    
    imputer = SimpleImputer(strategy="median").fit(dummy_data)
    scaler = StandardScaler().fit(dummy_data)
    X_scaled = scaler.transform(dummy_data)

    iso_forest = IsolationForest(n_estimators=50, random_state=42).fit(X_scaled)
    knn_model = NearestNeighbors(n_neighbors=5).fit(X_scaled)
    pca_model = PCA(n_components=10, random_state=42).fit(X_scaled)

    class StubAE(nn.Module):
        def __init__(self, d=32):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(d, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU())
            self.decoder = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, d))
        def forward(self, x):
            return self.decoder(self.encoder(x))

    autoencoder = StubAE(len(numeric_features))

    calibration_thresholds = {
        "IsolationForest": 0.0671,
        "KNN": 5.3867,
        "PCA": 0.3934,
        "Autoencoder": 0.8712
    }

    decision_thresholds = {
        "IsolationForest": -0.502943,
        "KNN": 0.616420,
        "PCA": 0.324694,
        "Autoencoder": 0.380919,
        "EqualWeightEnsemble": 0.337992
    }

    bundle = {
        "imputer": imputer,
        "scaler": scaler,
        "iso_forest": iso_forest,
        "knn_model": knn_model,
        "pca_model": pca_model,
        "autoencoder": autoencoder.state_dict(),
        "feature_names": numeric_features,
        "base_features": base_features,
        "calibration_thresholds": calibration_thresholds,
        "decision_thresholds": decision_thresholds,
    }

    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = models_dir / "skab_model_bundle.pkl"
    with open(bundle_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"Generated stub model bundle at: {bundle_path}")

if __name__ == "__main__":
    generate_local_stub_bundle()
