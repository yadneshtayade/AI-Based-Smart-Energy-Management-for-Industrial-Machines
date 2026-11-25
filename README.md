# Smart Industrial Energy AI – AI-Based Smart Energy Management and Predictive Maintenance System

A complete, futuristic industrial AI monitoring, anomaly detection, and predictive maintenance platform built with:
- **Frontend:** Responsive, modern Dark Cyber-Operations Web Application (HTML5, CSS3 Glassmorphism, Chart.js, Lucide Icons).
- **Backend:** High-performance asynchronous Python REST API (**FastAPI & Uvicorn**).
- **Machine Learning:** Scikit-learn, PyTorch Autoencoder, NumPy, Pandas.
- **NLP / GenAI / RAG:** FAISS vector retrieval + Qwen2.5-1.5B-Instruct / Extractive fallback with technical literature grounding (DOE & SKF manuals).
- **Autonomous Agent:** 5-tool sequential agent pipeline (`SensorAnalysis` → `AnomalyDetection` → `RAGRetrieval` → `InspectionReport` → `MaintenancePlanner`).
- **Benchmark Data:** Skoltech Anomaly Benchmark (**SKAB** industrial water-circulation testbed).

---

## 🚀 How to Run the Website

### Option 1: Modern Full-Stack Industrial Web App (Recommended)

In your terminal or PowerShell, run:

```powershell
python main.py
```

Then open your browser at:
👉 **[http://localhost:8000](http://localhost:8000)**

*(Swagger Interactive API documentation is available at `http://localhost:8000/docs`)*

---

### Option 2: Streamlit Dashboard (Alternative)

If you also wish to demonstrate the Streamlit version:

```powershell
streamlit run app.py
```
👉 Available at: **`http://localhost:8501`**

---

## 🌟 Key Web Platform Views

1. **Operations Dashboard (Overview)**
   - Telemetry counter, live health status gauge, normal vs anomaly distribution donut chart.
   - Live sensor stream with rolling mean baseline and anomaly highlights.
   - Model readiness matrix: Isolation Forest, KNN, PCA, Autoencoder, Equal-Weight Ensemble.
2. **Industrial Sensor Telemetry**
   - High-resolution time-series for all 8 SKAB sensors (`Vibration1`, `Vibration2`, `Current`, `Pressure`, `Temperature`, `Thermocouple`, `Voltage`, `FlowRate`).
   - Dynamic threshold crosshairs, rolling averages, and statistical summary cards (Mean, Max, Std).
3. **AI Anomaly Detection Studio**
   - Interactive sensor sliders & direct manual input with "Preset Normal" and "Preset Anomaly" buttons.
   - Live multi-model decision banner showing anomaly score, decision threshold, and vote counts across all 4 detectors.
4. **AI Maintenance Copilot (RAG + Qwen)**
   - Real-time engineering chat assistant grounded with technical engineering documentation.
   - Displays document source titles, page citations, and retrieval confidence scores.
5. **Autonomous Agentic AI Workflow**
   - Visual execution pipeline showing tool invocation progress:
     `Sensor Analysis` → `Anomaly Detection` → `RAG Retrieval` → `Inspection Report` → `Maintenance Planning`.
   - Generates a prioritized action sequence with urgency ratings and team assignments.
6. **Structured Inspection Reports & PDF Export**
   - Compiles auditable JSON inspection reports.
   - **One-click downloadable PDF reports** (using ReportLab) and raw JSON export.
7. **College Demo Mode (Dataset Defense Demo)**
   - Sample 10, 20, 50, or 100 held-out SKAB test records.
   - Real-time evaluation metrics: **Accuracy, Precision, Recall, and F1-Score**.
   - Interactive **Confusion Matrix** (True Normal, False Positive, False Negative, True Anomaly).
   - Filterable diagnostics table with CSV export.

---

## 📁 Project Structure

```
rag based gen ai model submission/
│
├── main.py                 # Single launch command for the full-stack web application
├── server.py               # FastAPI backend with REST endpoints & static server
├── frontend/               # Modern dark industrial web UI
│   ├── index.html          # Cyber-ops interface layout
│   ├── style.css           # Glassmorphism, animations & neon theme
│   └── app.js              # Full-featured client controller with Chart.js
├── model_loader.py         # Adapter to load trained ML bundle & metadata
├── prediction.py           # Preprocessing, feature engineering & model inference
├── rag_engine.py           # FAISS retrieval & technical document chunks
├── llm_engine.py           # Qwen2.5-1.5B-Instruct chat template & RAG prompting
├── report_generator.py     # Structured JSON & ReportLab PDF report generation
├── agent.py                # Autonomous agent tool-calling pipeline
├── data_loader.py          # SKAB dataset loader, caching & sampling
├── utils.py                # Common utilities, schema mapping & validators
├── export_colab_bundle.py  # Colab export snippet & local testing stub generator
├── requirements.txt        # Python package dependencies
├── models/                 # Exported model bundle directory
│   └── skab_model_bundle.pkl
└── data/                   # Data cache directory
```

---

## 📦 How to Export Your Trained Models from Google Colab

In your Google Colab notebook, run this cell right after model evaluation:

```python
import pickle
from google.colab import files

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

print("Export complete!")
files.download("skab_model_bundle.pkl")
```

Place the downloaded `skab_model_bundle.pkl` file into the `models/` directory:
```
models/skab_model_bundle.pkl
```
*(A starter bundle is already provided in `models/skab_model_bundle.pkl` so you can launch and test immediately).*
