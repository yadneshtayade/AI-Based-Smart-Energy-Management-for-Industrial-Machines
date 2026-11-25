"""
app.py – Smart Industrial Energy AI
Main Streamlit application entry point.

Run with:  streamlit run app.py
"""

import json
import logging
import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime

# ─────────────────────────── Page Config ────────────────────────────────────
st.set_page_config(
    page_title="Smart Industrial Energy AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────── Custom CSS ─────────────────────────────────────
st.markdown("""
<style>
/* ── Global Dark Theme ── */
:root {
    --bg-primary:    #0a192f;
    --bg-secondary:  #112240;
    --bg-card:       #1e3a5f;
    --accent-cyan:   #00d4ff;
    --accent-blue:   #0080ff;
    --accent-green:  #00ff88;
    --accent-red:    #ff4444;
    --accent-orange: #ffaa00;
    --text-primary:  #ccd6f6;
    --text-muted:    #8892b0;
    --border:        #233554;
}

/* Force dark background */
.stApp { background-color: var(--bg-primary); }
.block-container { padding-top: 1rem; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stMarkdown { color: var(--text-primary); }

/* Metric cards */
[data-testid="metric-container"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem;
    box-shadow: 0 4px 15px rgba(0, 212, 255, 0.05);
    transition: transform 0.2s, box-shadow 0.2s;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0, 212, 255, 0.1);
}

/* Headers */
h1, h2, h3, h4 { color: var(--accent-cyan) !important; }

/* Status badges */
.badge-normal  { background:#00ff88; color:#0a192f; padding:4px 12px; border-radius:20px; font-weight:700; font-size:0.85rem; }
.badge-anomaly { background:#ff4444; color:#fff;    padding:4px 12px; border-radius:20px; font-weight:700; font-size:0.85rem; }
.badge-warning { background:#ffaa00; color:#0a192f; padding:4px 12px; border-radius:20px; font-weight:700; font-size:0.85rem; }

/* Info boxes */
.info-box {
    background: var(--bg-card);
    border-left: 4px solid var(--accent-cyan);
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    color: var(--text-primary);
}
.warning-box {
    background: rgba(255,170,0,0.1);
    border-left: 4px solid var(--accent-orange);
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
}
.error-box {
    background: rgba(255,68,68,0.1);
    border-left: 4px solid var(--accent-red);
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
}

/* Chat bubbles */
.user-msg {
    background: var(--bg-card);
    border-radius: 12px 12px 4px 12px;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0;
    color: var(--text-primary);
    text-align: right;
}
.ai-msg {
    background: rgba(0,212,255,0.08);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 12px 12px 12px 4px;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0;
    color: var(--text-primary);
}

/* Step indicators (agent) */
.step-done    { color: var(--accent-green); }
.step-running { color: var(--accent-cyan); }
.step-failed  { color: var(--accent-red); }
.step-pending { color: var(--text-muted); }

/* Tables */
.dataframe { background: var(--bg-secondary) !important; color: var(--text-primary) !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan));
    color: var(--bg-primary);
    border: none;
    border-radius: 8px;
    font-weight: 700;
    transition: all 0.2s;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(0,212,255,0.3);
}

/* Plotly chart background */
.js-plotly-plot .plotly { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────── Imports (after page config) ────────────────────
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

from model_loader import load_model_bundle, models_available, get_missing_files
from data_loader import load_skab_dataset, load_uploaded_csv, get_dataset_stats, sample_records
from prediction import predict_anomalies
from rag_engine import load_rag_index, retrieve, format_retrieved_context
from llm_engine import load_qwen, generate_response, build_chat_prompt, build_report_prompt, MAINTENANCE_SYSTEM_PROMPT
from report_generator import generate_json_report, generate_pdf_report
from agent import PredictiveMaintenanceAgent
from utils import sensor_label, health_label, health_color, BASE_SENSOR_FEATURES

# ─────────────────────────── Color Palette ──────────────────────────────────
PLOTLY_TEMPLATE = "plotly_dark"
COLOR_NORMAL  = "#00ff88"
COLOR_ANOMALY = "#ff4444"
COLOR_WARNING = "#ffaa00"
COLOR_CYAN    = "#00d4ff"
COLOR_BLUE    = "#0080ff"
BG_COLOR      = "#0a192f"
CARD_COLOR    = "#1e3a5f"


# ═══════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding: 1rem 0;'>
            <div style='font-size:2.5rem;'>⚡</div>
            <div style='color:#00d4ff; font-size:1.1rem; font-weight:700;'>Smart Industrial</div>
            <div style='color:#00d4ff; font-size:0.9rem;'>Energy AI</div>
            <div style='color:#8892b0; font-size:0.75rem; margin-top:4px;'>Predictive Maintenance System</div>
        </div>
        <hr style='border-color:#233554;'>
        """, unsafe_allow_html=True)

        pages = {
            "🏠 Main Dashboard":        "dashboard",
            "📊 Machine Monitoring":    "monitoring",
            "🤖 AI Anomaly Detection":  "anomaly",
            "💬 AI Maintenance Chat":   "chat",
            "📋 Inspection Report":     "report",
            "🔗 Agentic AI":            "agent",
            "🧪 Sample Demo":           "demo",
        }
        st.markdown("### 🧭 Navigation")
        page = st.radio(
            "Select Page",
            list(pages.keys()),
            label_visibility="collapsed",
        )
        st.markdown("<hr style='border-color:#233554;'>", unsafe_allow_html=True)

        # Model status
        bundle = load_model_bundle()
        if models_available(bundle):
            st.markdown("**🟢 ML Models:** Loaded")
        else:
            st.markdown("**🔴 ML Models:** Not found")
            with st.expander("ℹ️ How to load models"):
                st.markdown("""
Export from your Colab notebook:
```python
import pickle
bundle = {
  'imputer': imputer,
  'scaler': scaler,
  'iso_forest': iso_forest,
  'knn_model': knn_model,
  'pca_model': pca_model,
  'autoencoder': autoencoder,
  'feature_names': numeric_features,
  'base_features': base_features,
  'calibration_thresholds': calibration_thresholds,
  'decision_thresholds': decision_thresholds,
}
with open('skab_model_bundle.pkl', 'wb') as f:
    pickle.dump(bundle, f)
```
Then place `skab_model_bundle.pkl` in the `models/` folder.
""")

        # LLM status
        qwen_model, _ = load_qwen()
        if qwen_model is not None:
            st.markdown("**🟢 Qwen LLM:** Loaded")
        else:
            st.markdown("**🟡 Qwen LLM:** Extractive fallback")

        st.markdown("""
        <hr style='border-color:#233554;'>
        <div style='color:#8892b0; font-size:0.75rem; text-align:center;'>
        SKAB Dataset · Ensemble AI<br>CO1 · CO2 · CO3 · CO4<br>
        Academic Project Demo
        </div>
        """, unsafe_allow_html=True)

        return pages[page]


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE 1: MAIN DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════

def page_dashboard():
    st.markdown("## 🏠 Main Dashboard – Industrial AI System Overview")

    bundle = load_model_bundle()

    with st.spinner("Loading SKAB dataset..."):
        df = load_skab_dataset()

    if df is None:
        st.markdown("""
        <div class='warning-box'>
        ⚠️ SKAB dataset not found. The dashboard will show placeholder values.
        Dataset will be downloaded automatically when internet is available.
        </div>
        """, unsafe_allow_html=True)
        # Use placeholder stats
        stats = {
            "total_records": 37401, "normal_records": 24334,
            "anomaly_records": 13067, "anomaly_percentage": 34.94,
            "num_experiments": 35, "available_sensors": BASE_SENSOR_FEATURES,
        }
    else:
        stats = get_dataset_stats(df)

    # ── KPI Metrics ──────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📦 Total Records", f"{stats['total_records']:,}", help="Total sensor records analyzed")
    with col2:
        st.metric("✅ Normal Records", f"{stats['normal_records']:,}",
                  delta=f"{100 - stats['anomaly_percentage']:.1f}%")
    with col3:
        st.metric("🚨 Anomalies Detected", f"{stats['anomaly_records']:,}",
                  delta=f"-{stats['anomaly_percentage']:.1f}%", delta_color="inverse")
    with col4:
        st.metric("⚠️ Anomaly Rate", f"{stats['anomaly_percentage']:.2f}%",
                  help="Percentage of anomalous records in dataset")

    st.markdown("---")

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.markdown("### 📡 Machine Health Indicators")
        # Health gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=100 - stats["anomaly_percentage"],
            title={"text": "System Health Score", "font": {"color": COLOR_CYAN}},
            delta={"reference": 70, "valueformat": ".1f"},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#8892b0"},
                "bar": {"color": COLOR_CYAN},
                "bgcolor": CARD_COLOR,
                "steps": [
                    {"range": [0, 50], "color": "#ff4444"},
                    {"range": [50, 75], "color": "#ffaa00"},
                    {"range": [75, 100], "color": "#00ff88"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 2},
                    "thickness": 0.75,
                    "value": 100 - stats["anomaly_percentage"],
                },
            },
        ))
        fig_gauge.update_layout(
            template=PLOTLY_TEMPLATE,
            paper_bgcolor=BG_COLOR,
            plot_bgcolor=BG_COLOR,
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Sensor availability
        st.markdown("#### 🔌 Available Sensors")
        sensor_cols = st.columns(4)
        for i, s in enumerate(BASE_SENSOR_FEATURES):
            is_avail = df is None or s in df.columns
            icon = "🟢" if is_avail else "🔴"
            sensor_cols[i % 4].markdown(f"{icon} **{s}**")

    with col_right:
        st.markdown("### 📊 Normal vs Anomaly Distribution")
        fig_pie = px.pie(
            values=[stats["normal_records"], stats["anomaly_records"]],
            names=["Normal", "Anomaly"],
            color_discrete_sequence=[COLOR_NORMAL, COLOR_ANOMALY],
            hole=0.5,
        )
        fig_pie.update_traces(
            textposition="inside", textinfo="percent+label",
            marker=dict(line=dict(color=BG_COLOR, width=2)),
        )
        fig_pie.update_layout(
            template=PLOTLY_TEMPLATE,
            paper_bgcolor=BG_COLOR,
            plot_bgcolor=BG_COLOR,
            height=300,
            showlegend=True,
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("---")

    # ── Sensor Time-Series (sample) ───────────────────────────────────────
    st.markdown("### 📈 Sensor Activity Overview")
    if df is not None:
        available_sensors = stats["available_sensors"]
        if available_sensors:
            sample_df = df.sample(min(500, len(df))).sort_index()
            fig_sensors = make_subplots(
                rows=2, cols=2,
                subplot_titles=[sensor_label(s) for s in available_sensors[:4]],
                shared_xaxes=False,
            )
            for i, sensor in enumerate(available_sensors[:4]):
                row, col = divmod(i, 2)
                color = COLOR_ANOMALY if "Vibration" in sensor else COLOR_CYAN
                fig_sensors.add_trace(
                    go.Scatter(
                        y=sample_df[sensor], mode="lines",
                        name=sensor, line=dict(color=color, width=1),
                    ),
                    row=row + 1, col=col + 1,
                )
            fig_sensors.update_layout(
                template=PLOTLY_TEMPLATE,
                paper_bgcolor=BG_COLOR,
                plot_bgcolor=BG_COLOR,
                height=400,
                showlegend=False,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_sensors, use_container_width=True)
    else:
        st.info("📂 Load the SKAB dataset to view sensor activity charts.")

    # ── AI System Status ──────────────────────────────────────────────────
    st.markdown("### 🤖 AI System Status")
    col1, col2, col3, col4, col5 = st.columns(5)
    statuses = [
        ("Isolation Forest", bundle.get("iso_forest") is not None),
        ("KNN Detector", bundle.get("knn_model") is not None),
        ("PCA Detector", bundle.get("pca_model") is not None),
        ("Autoencoder", bundle.get("autoencoder") is not None),
        ("RAG + Qwen", True),
    ]
    for col, (name, ok) in zip([col1, col2, col3, col4, col5], statuses):
        icon = "🟢" if ok else "🔴"
        status = "Active" if ok else "Demo Mode"
        col.metric(f"{icon} {name}", status)

    # ── Recent Anomaly Alerts ─────────────────────────────────────────────
    if df is not None and "anomaly" in df.columns:
        st.markdown("### 🚨 Recent Anomaly Alerts")
        anomaly_df = df[df["anomaly"] == 1].tail(5)
        if not anomaly_df.empty:
            display_cols = (
                ["datetime", "experiment_id"] +
                [s for s in BASE_SENSOR_FEATURES if s in anomaly_df.columns][:4] +
                ["anomaly"]
            )
            display_cols = [c for c in display_cols if c in anomaly_df.columns]
            st.dataframe(
                anomaly_df[display_cols].reset_index(drop=True),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No anomaly records found.")


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE 2: MACHINE MONITORING
# ═══════════════════════════════════════════════════════════════════════════

def page_monitoring():
    st.markdown("## 📊 Machine Monitoring – Live Sensor Dashboard")

    df_full = None
    source = "none"

    # ── Data Source ──────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 📂 Data Source")
        data_source = st.radio(
            "Choose data source:",
            ["Sample SKAB Dataset", "Upload CSV"],
            key="monitor_source",
        )

    if data_source == "Upload CSV":
        uploaded = st.file_uploader("Upload SKAB-format CSV", type=["csv"], key="monitor_upload")
        if uploaded:
            df_full = load_uploaded_csv(uploaded)
            source = "uploaded"
            st.success(f"✅ Loaded {len(df_full):,} records from uploaded file.")
    else:
        with st.spinner("Loading SKAB dataset..."):
            df_full = load_skab_dataset()
        if df_full is not None:
            source = "skab"
            st.success(f"✅ SKAB Dataset: {len(df_full):,} records, {df_full['experiment_id'].nunique() if 'experiment_id' in df_full.columns else '?'} experiments")
        else:
            st.markdown("""
            <div class='warning-box'>
            ⚠️ SKAB dataset not found locally. Please upload a CSV file or ensure internet access for auto-download.
            </div>
            """, unsafe_allow_html=True)
            return

    if df_full is None:
        return

    available_sensors = [s for s in BASE_SENSOR_FEATURES if s in df_full.columns]
    if not available_sensors:
        st.error("No recognizable sensor columns found in the dataset.")
        return

    # ── Experiment Filter ─────────────────────────────────────────────────
    col1, col2 = st.columns([1, 3])
    with col1:
        if "experiment_id" in df_full.columns:
            exp_ids = sorted(df_full["experiment_id"].astype(str).unique())
            selected_exp = st.selectbox("Experiment ID", ["All"] + list(exp_ids), key="exp_select")
            if selected_exp != "All":
                df = df_full[df_full["experiment_id"].astype(str) == selected_exp].copy()
            else:
                df = df_full.copy()
        else:
            df = df_full.copy()
            selected_exp = "N/A"

        st.markdown(f"**Records:** {len(df):,}")
        if "anomaly" in df.columns:
            anom_count = int((df["anomaly"] == 1).sum())
            st.markdown(f"**Anomalies:** {anom_count:,}")

    with col2:
        selected_sensor = st.selectbox(
            "Select Sensor",
            available_sensors,
            format_func=sensor_label,
            key="sensor_select",
        )

    # ── Sensor Time Series ────────────────────────────────────────────────
    if selected_sensor in df.columns:
        st.markdown(f"### 📈 {sensor_label(selected_sensor)} Over Time")

        # Downsample for performance
        plot_df = df.reset_index(drop=True)
        if len(plot_df) > 2000:
            plot_df = plot_df.sample(2000).sort_index()

        fig = go.Figure()

        # Normal points
        if "anomaly" in plot_df.columns:
            normal_df = plot_df[plot_df["anomaly"] == 0]
            anomaly_df = plot_df[plot_df["anomaly"] == 1]

            fig.add_trace(go.Scatter(
                x=normal_df.index, y=normal_df[selected_sensor],
                mode="markers", name="Normal",
                marker=dict(color=COLOR_NORMAL, size=3, opacity=0.6),
            ))
            if not anomaly_df.empty:
                fig.add_trace(go.Scatter(
                    x=anomaly_df.index, y=anomaly_df[selected_sensor],
                    mode="markers", name="Anomaly",
                    marker=dict(color=COLOR_ANOMALY, size=5, symbol="x"),
                ))
        else:
            fig.add_trace(go.Scatter(
                x=plot_df.index, y=plot_df[selected_sensor],
                mode="lines", name=selected_sensor,
                line=dict(color=COLOR_CYAN, width=1.5),
            ))

        # Rolling mean
        rolling = df[selected_sensor].rolling(10, min_periods=1).mean().reset_index(drop=True)
        if len(plot_df) > 2000:
            rolling = rolling.iloc[plot_df.index]
        fig.add_trace(go.Scatter(
            x=plot_df.index, y=rolling.values,
            mode="lines", name="Rolling Mean (10)",
            line=dict(color=COLOR_WARNING, width=2, dash="dash"),
        ))

        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            paper_bgcolor=BG_COLOR,
            plot_bgcolor=BG_COLOR,
            xaxis_title="Record Index",
            yaxis_title=sensor_label(selected_sensor),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            height=400,
            margin=dict(l=20, r=20, t=20, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Sensor Statistics ─────────────────────────────────────────────────
    st.markdown("### 📊 Sensor Statistics")
    stats_data = []
    for s in available_sensors:
        if s in df.columns:
            col_data = df[s].dropna()
            stats_data.append({
                "Sensor": sensor_label(s),
                "Mean": round(col_data.mean(), 4),
                "Std": round(col_data.std(), 4),
                "Min": round(col_data.min(), 4),
                "Max": round(col_data.max(), 4),
                "Median": round(col_data.median(), 4),
            })
    if stats_data:
        st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)

    # ── Multi-sensor comparison ───────────────────────────────────────────
    st.markdown("### 🔁 Multi-Sensor Comparison")
    selected_multi = st.multiselect(
        "Select sensors to compare:",
        available_sensors,
        default=available_sensors[:min(3, len(available_sensors))],
        format_func=sensor_label,
        key="multi_sensor",
    )
    if selected_multi:
        norm_df = df[selected_multi].copy()
        # Normalize to [0,1] for comparison
        for col in selected_multi:
            cmin, cmax = norm_df[col].min(), norm_df[col].max()
            if cmax > cmin:
                norm_df[col] = (norm_df[col] - cmin) / (cmax - cmin)

        plot_norm = norm_df.reset_index(drop=True)
        if len(plot_norm) > 1000:
            plot_norm = plot_norm.iloc[::len(plot_norm)//1000]

        fig_multi = go.Figure()
        palette = [COLOR_CYAN, COLOR_ANOMALY, COLOR_NORMAL,
                   COLOR_WARNING, COLOR_BLUE, "#cc77ff", "#ffcc44", "#44ffcc"]
        for i, s in enumerate(selected_multi):
            fig_multi.add_trace(go.Scatter(
                y=plot_norm[s], mode="lines", name=sensor_label(s),
                line=dict(color=palette[i % len(palette)], width=1.5),
            ))

        fig_multi.update_layout(
            template=PLOTLY_TEMPLATE,
            paper_bgcolor=BG_COLOR,
            plot_bgcolor=BG_COLOR,
            yaxis_title="Normalized Value (0–1)",
            xaxis_title="Record Index",
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            height=350,
            margin=dict(l=20, r=20, t=20, b=40),
        )
        st.plotly_chart(fig_multi, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE 3: AI ANOMALY DETECTION
# ═══════════════════════════════════════════════════════════════════════════

def page_anomaly_detection():
    st.markdown("## 🤖 AI Anomaly Detection – Ensemble Model Inference")

    bundle = load_model_bundle()

    if not models_available(bundle):
        st.markdown("""
        <div class='warning-box'>
        ⚠️ <b>Trained models not found.</b> Running in demo mode with default parameters.
        <br><br>
        To use your trained models, export from Colab and place <code>skab_model_bundle.pkl</code>
        in the <code>models/</code> folder. See the sidebar for export instructions.
        </div>
        """, unsafe_allow_html=True)

    # ── Data Input ────────────────────────────────────────────────────────
    input_method = st.radio(
        "Input Method:",
        ["📂 Upload CSV", "🧪 Use Sample Records", "✍️ Manual Input"],
        horizontal=True,
        key="anomaly_input",
    )

    df_input = None

    if input_method == "📂 Upload CSV":
        uploaded = st.file_uploader("Upload CSV for prediction", type=["csv"], key="anomaly_upload")
        if uploaded:
            df_input = load_uploaded_csv(uploaded)
            st.success(f"✅ Loaded {len(df_input):,} records.")

    elif input_method == "🧪 Use Sample Records":
        with st.spinner("Loading SKAB dataset..."):
            df_full = load_skab_dataset()
        if df_full is not None:
            n_sample = st.slider("Number of records:", 5, 100, 20, key="anomaly_n")
            balanced = st.checkbox("Balanced (equal normal/anomaly)", value=True)
            if st.button("🔀 Load Sample Records", key="load_sample_btn"):
                df_input = sample_records(df_full, n=n_sample, balanced=balanced)
                st.session_state["anomaly_df"] = df_input
            elif "anomaly_df" in st.session_state:
                df_input = st.session_state["anomaly_df"]
        else:
            st.warning("SKAB dataset not found. Please upload a CSV instead.")

    elif input_method == "✍️ Manual Input":
        st.markdown("#### Enter Sensor Values Manually")
        base_feats = bundle.get("base_features") or BASE_SENSOR_FEATURES
        cols = st.columns(4)
        manual_values = {}
        defaults = {
            "Vibration1": 0.027, "Vibration2": 0.040, "Current": 1.3,
            "Pressure": 0.055, "Temperature": 75.5, "Thermocouple": 25.8,
            "Voltage": 230.0, "FlowRate": 32.0,
        }
        for i, feat in enumerate(base_feats):
            with cols[i % 4]:
                manual_values[feat] = st.number_input(
                    sensor_label(feat), value=defaults.get(feat, 0.0),
                    format="%.4f", key=f"manual_{feat}",
                )
        if st.button("🔮 Predict This Record", key="manual_predict"):
            df_input = pd.DataFrame([manual_values])
            st.session_state["anomaly_df"] = df_input

        elif "anomaly_df" in st.session_state and input_method == "✍️ Manual Input":
            df_input = st.session_state.get("anomaly_df")

    # ── Run Prediction ────────────────────────────────────────────────────
    if df_input is not None and len(df_input) > 0:
        st.markdown("---")
        if st.button("🚀 Run Ensemble Anomaly Detection", type="primary", key="run_pred"):
            with st.spinner("⚙️ Running all 4 models + ensemble..."):
                try:
                    result = predict_anomalies(df_input, bundle)
                    st.session_state["pred_result"] = result
                    st.session_state["pred_df_input"] = df_input
                    st.success("✅ Prediction complete!")
                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    logger.exception("Prediction failed")

        result = st.session_state.get("pred_result")
        if result is not None:
            _render_prediction_results(result, st.session_state.get("pred_df_input", df_input))


def _render_prediction_results(result, df_input):
    """Render prediction results with charts and tables."""
    df_res = result["df_results"]
    ens_pred = result["ensemble_prediction"]
    ens_score = result["ensemble_score"]
    thr = result["decision_thresholds"]

    # ── Summary Metrics ──────────────────────────────────────────────────
    total = len(df_res)
    if hasattr(ens_pred, "__len__"):
        n_anomaly = int(ens_pred.sum())
    else:
        n_anomaly = int(ens_pred)

    n_normal = total - n_anomaly
    anom_pct = n_anomaly / total * 100 if total > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📦 Records", total)
    col2.metric("✅ Normal", n_normal)
    col3.metric("🚨 Anomaly", n_anomaly)
    col4.metric("⚠️ Anomaly Rate", f"{anom_pct:.1f}%")

    # ── Ensemble Score vs Threshold ──────────────────────────────────────
    st.markdown("### 📊 Ensemble Score vs Decision Threshold")
    ens_thr = thr.get("EqualWeightEnsemble", 0.338)

    if hasattr(ens_score, "__len__"):
        scores_list = [float(s) for s in ens_score]
        preds_list = [int(p) for p in ens_pred]
    else:
        scores_list = [float(ens_score)]
        preds_list = [int(ens_pred)]

    fig_score = go.Figure()
    colors_pts = [COLOR_ANOMALY if p else COLOR_NORMAL for p in preds_list]
    fig_score.add_trace(go.Scatter(
        x=list(range(len(scores_list))),
        y=scores_list,
        mode="markers+lines",
        marker=dict(color=colors_pts, size=8),
        line=dict(color=COLOR_CYAN, width=1),
        name="Ensemble Score",
    ))
    fig_score.add_hline(
        y=ens_thr, line_dash="dash", line_color=COLOR_WARNING,
        annotation_text=f"Threshold: {ens_thr:.3f}", annotation_font_color=COLOR_WARNING,
    )
    fig_score.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        xaxis_title="Record Index",
        yaxis_title="Ensemble Anomaly Score",
        height=350,
        margin=dict(l=20, r=20, t=20, b=40),
    )
    st.plotly_chart(fig_score, use_container_width=True)

    # ── Individual Model Results ──────────────────────────────────────────
    st.markdown("### 🔬 Individual Model Predictions")
    col1, col2, col3, col4 = st.columns(4)
    model_cols = zip(
        ["IsolationForest", "KNN", "PCA", "Autoencoder"],
        [col1, col2, col3, col4],
    )
    for model, col in model_cols:
        scores = result["individual_scores"].get(model, [0])
        preds = result["individual_predictions"].get(model, [0])
        model_thr = thr.get(model, 0.5)

        if hasattr(preds, "__len__"):
            n_anom = int(sum(preds))
        else:
            n_anom = int(preds)

        if hasattr(scores, "__len__"):
            avg_score = float(np.mean(scores))
        else:
            avg_score = float(scores)

        badge_class = "badge-anomaly" if n_anom > 0 else "badge-normal"
        badge_text = f"{n_anom} Anomalies" if n_anom > 0 else "Normal"

        with col:
            st.markdown(f"**{model}**")
            st.markdown(f"<span class='{badge_class}'>{badge_text}</span>", unsafe_allow_html=True)
            st.markdown(f"Avg Score: `{avg_score:.4f}`")
            st.markdown(f"Threshold: `{model_thr:.4f}`")

    # ── Normal vs Anomaly Bar Chart ───────────────────────────────────────
    if total > 1:
        st.markdown("### 📈 Classification Distribution")
        fig_bar = px.bar(
            x=["Normal", "Anomaly"],
            y=[n_normal, n_anomaly],
            color=["Normal", "Anomaly"],
            color_discrete_map={"Normal": COLOR_NORMAL, "Anomaly": COLOR_ANOMALY},
            text_auto=True,
        )
        fig_bar.update_layout(
            template=PLOTLY_TEMPLATE,
            paper_bgcolor=BG_COLOR,
            plot_bgcolor=BG_COLOR,
            height=300,
            showlegend=False,
            margin=dict(l=20, r=20, t=20, b=40),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Comparison vs Ground Truth ────────────────────────────────────────
    if "anomaly" in df_res.columns:
        st.markdown("### ✅ Comparison vs Ground Truth Labels")
        y_true = df_res["anomaly"].values
        y_pred = preds_list if len(preds_list) == len(y_true) else [0] * len(y_true)

        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            balanced_accuracy_score, confusion_matrix,
        )

        try:
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            bal_acc = balanced_accuracy_score(y_true, y_pred)

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Accuracy", f"{acc:.3f}")
            m2.metric("Precision", f"{prec:.3f}")
            m3.metric("Recall", f"{rec:.3f}")
            m4.metric("F1 Score", f"{f1:.3f}")
            m5.metric("Balanced Acc", f"{bal_acc:.3f}")

            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            fig_cm = px.imshow(
                cm, text_auto=True,
                x=["Pred Normal", "Pred Anomaly"],
                y=["True Normal", "True Anomaly"],
                color_continuous_scale="Blues",
                title="Confusion Matrix",
            )
            fig_cm.update_layout(
                template=PLOTLY_TEMPLATE,
                paper_bgcolor=BG_COLOR,
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        except Exception as e:
            st.warning(f"Metric computation skipped: {e}")

    # ── Results Table ─────────────────────────────────────────────────────
    st.markdown("### 📋 Prediction Results Table")
    display_cols = ["EnsembleScore", "EnsemblePrediction", "AnomalyVotes",
                    "score_IsolationForest", "score_KNN", "score_PCA", "score_Autoencoder"]
    display_cols = [c for c in display_cols if c in df_res.columns]
    if "anomaly" in df_res.columns:
        display_cols = ["anomaly"] + display_cols

    base_cols = [c for c in BASE_SENSOR_FEATURES if c in df_res.columns]
    display_cols = base_cols[:4] + display_cols

    st.dataframe(df_res[display_cols].head(100), use_container_width=True, hide_index=False)

    # ── Download ──────────────────────────────────────────────────────────
    csv_buffer = df_res.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Prediction Results CSV",
        data=csv_buffer,
        file_name="anomaly_predictions.csv",
        mime="text/csv",
    )


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE 4: AI MAINTENANCE CHAT
# ═══════════════════════════════════════════════════════════════════════════

def page_chat():
    st.markdown("## 💬 AI Maintenance Assistant")
    st.markdown(
        "Ask questions about machine anomalies, maintenance procedures, "
        "or sensor readings. Powered by Qwen2.5-1.5B-Instruct + FAISS RAG."
    )

    bundle = load_model_bundle()
    with st.spinner("Loading RAG index..."):
        rag_index, rag_chunks, embed_model = load_rag_index()

    # ── Context Panel ──────────────────────────────────────────────────────
    with st.expander("🔧 Set Machine Context (optional)", expanded=False):
        base_feats = bundle.get("base_features") or BASE_SENSOR_FEATURES
        cols = st.columns(4)
        context_values = {}
        defaults = {
            "Vibration1": 0.027, "Vibration2": 0.040, "Current": 1.3,
            "Pressure": 0.055, "Temperature": 75.5, "Thermocouple": 25.8,
            "Voltage": 230.0, "FlowRate": 32.0,
        }
        for i, feat in enumerate(base_feats):
            with cols[i % 4]:
                context_values[feat] = st.number_input(
                    sensor_label(feat), value=defaults.get(feat, 0.0),
                    format="%.4f", key=f"chat_ctx_{feat}",
                )

        anomaly_ctx = st.session_state.get("chat_anomaly_ctx", None)
        if st.button("🔮 Run Detection for Context", key="chat_ctx_run"):
            df_ctx = pd.DataFrame([context_values])
            try:
                res = predict_anomalies(df_ctx, bundle)
                anomaly_ctx = {
                    "ensemble_score": float(res["ensemble_score"][0]) if hasattr(res["ensemble_score"], "__len__") else float(res["ensemble_score"]),
                    "ensemble_prediction": int(res["ensemble_prediction"][0]) if hasattr(res["ensemble_prediction"], "__len__") else int(res["ensemble_prediction"]),
                    "anomaly_votes": int(res["anomaly_votes"][0]) if hasattr(res["anomaly_votes"], "__len__") else int(res["anomaly_votes"]),
                }
                st.session_state["chat_anomaly_ctx"] = anomaly_ctx
                status = "🚨 ANOMALY" if anomaly_ctx["ensemble_prediction"] else "✅ NORMAL"
                st.success(f"Context detection: {status} | Score: {anomaly_ctx['ensemble_score']:.4f}")
            except Exception as e:
                st.error(f"Detection error: {e}")

    # ── Chat History ────────────────────────────────────────────────────────
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Suggested questions
    suggestions = [
        "Why was an anomaly detected?",
        "What could cause abnormal vibration?",
        "What does high pressure indicate?",
        "What maintenance action should be prioritized?",
        "Which documents support this recommendation?",
    ]
    st.markdown("**💡 Suggested Questions:**")
    sug_cols = st.columns(len(suggestions))
    for i, (sug_col, sug) in enumerate(zip(sug_cols, suggestions)):
        if sug_col.button(sug, key=f"sug_{i}"):
            st.session_state["chat_input"] = sug

    # Display chat history
    for entry in st.session_state["chat_history"]:
        st.markdown(f"<div class='user-msg'>👤 {entry['user']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='ai-msg'>🤖 {entry['ai']}</div>", unsafe_allow_html=True)
        if entry.get("refs"):
            with st.expander(f"📚 References ({len(entry['refs'])} docs)"):
                for ref in entry["refs"]:
                    st.markdown(
                        f"**{ref.get('source', 'Unknown')}**, p.{ref.get('page', '?')} "
                        f"(score: {ref.get('retrieval_score', 0):.3f})\n\n"
                        f"> {ref.get('text', '')[:200]}..."
                    )

    # ── Input ──────────────────────────────────────────────────────────────
    user_input = st.text_input(
        "Your Question:",
        value=st.session_state.pop("chat_input", ""),
        placeholder="Ask about machine anomalies, maintenance procedures...",
        key="chat_text_input",
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        send = st.button("📨 Send", type="primary")
    with col2:
        if st.button("🗑️ Clear Chat"):
            st.session_state["chat_history"] = []
            st.rerun()

    if send and user_input.strip():
        with st.spinner("🤖 AI is thinking..."):
            # Build sensor context string
            sensor_ctx = "\n".join(
                f"  {sensor_label(k)}: {v:.4f}"
                for k, v in context_values.items()
            )

            # Retrieve relevant docs
            retrieved = retrieve(
                user_input,
                top_k=3,
                index=rag_index,
                chunks=rag_chunks,
                embed_model=embed_model,
            )
            docs_ctx = format_retrieved_context(retrieved)

            # Build prompts
            user_prompt = build_chat_prompt(
                user_question=user_input,
                sensor_context=sensor_ctx,
                retrieved_docs=docs_ctx,
                anomaly_result=st.session_state.get("chat_anomaly_ctx"),
            )

            # Generate response
            response = generate_response(
                system_prompt=MAINTENANCE_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_new_tokens=400,
            )

        # Save to history
        st.session_state["chat_history"].append({
            "user": user_input,
            "ai": response,
            "refs": retrieved,
        })
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE 5: INSPECTION REPORT
# ═══════════════════════════════════════════════════════════════════════════

def page_report():
    st.markdown("## 📋 AI Inspection Report Generator")

    bundle = load_model_bundle()
    with st.spinner("Loading RAG index..."):
        rag_index, rag_chunks, embed_model = load_rag_index()

    # ── Machine Config ────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        machine_id = st.text_input("Machine / Asset ID:", value="SKAB-PUMP-01", key="report_machine_id")
    with col2:
        exp_id = st.text_input("Experiment ID:", value="valve1/1", key="report_exp_id")

    # ── Sensor Input ──────────────────────────────────────────────────────
    st.markdown("#### 📡 Sensor Readings")
    base_feats = bundle.get("base_features") or BASE_SENSOR_FEATURES
    cols = st.columns(4)
    sensor_readings = {}
    defaults = {
        "Vibration1": 0.027, "Vibration2": 0.040, "Current": 1.3,
        "Pressure": 0.055, "Temperature": 75.5, "Thermocouple": 25.8,
        "Voltage": 230.0, "FlowRate": 32.0,
    }
    for i, feat in enumerate(base_feats):
        with cols[i % 4]:
            sensor_readings[feat] = st.number_input(
                sensor_label(feat),
                value=defaults.get(feat, 0.0),
                format="%.4f",
                key=f"report_sensor_{feat}",
            )

    if st.button("🔮 Run Detection + Generate Report", type="primary", key="gen_report"):
        with st.spinner("⚙️ Running anomaly detection..."):
            df_report = pd.DataFrame([sensor_readings])
            try:
                anomaly_result = predict_anomalies(df_report, bundle)
                pred_scalar = {
                    "individual_scores": {m: float(v[0]) if hasattr(v, "__len__") else float(v)
                                         for m, v in anomaly_result["individual_scores"].items()},
                    "individual_predictions": {m: int(v[0]) if hasattr(v, "__len__") else int(v)
                                               for m, v in anomaly_result["individual_predictions"].items()},
                    "ensemble_score": float(anomaly_result["ensemble_score"][0]) if hasattr(anomaly_result["ensemble_score"], "__len__") else float(anomaly_result["ensemble_score"]),
                    "ensemble_prediction": int(anomaly_result["ensemble_prediction"][0]) if hasattr(anomaly_result["ensemble_prediction"], "__len__") else int(anomaly_result["ensemble_prediction"]),
                    "anomaly_votes": int(anomaly_result["anomaly_votes"][0]) if hasattr(anomaly_result["anomaly_votes"], "__len__") else int(anomaly_result["anomaly_votes"]),
                    "decision_thresholds": anomaly_result["decision_thresholds"],
                }
            except Exception as e:
                st.error(f"Anomaly detection failed: {e}")
                return

        with st.spinner("📚 Retrieving maintenance documents..."):
            query = f"machine maintenance anomaly {pred_scalar['ensemble_prediction']}"
            retrieved_docs = retrieve(query, 3, rag_index, rag_chunks, embed_model)
            docs_ctx = format_retrieved_context(retrieved_docs)

        # Optionally use Qwen to generate JSON
        llm_response = None
        qwen_model, _ = load_qwen()
        if qwen_model is not None:
            with st.spinner("🧠 Generating AI report with Qwen..."):
                try:
                    report_prompt = build_report_prompt(
                        machine_id=machine_id,
                        sensor_readings=sensor_readings,
                        anomaly_result=pred_scalar,
                        retrieved_docs=docs_ctx,
                    )
                    llm_response = generate_response(
                        system_prompt=MAINTENANCE_SYSTEM_PROMPT,
                        user_prompt=report_prompt,
                        max_new_tokens=600,
                    )
                except Exception as e:
                    logger.warning("Qwen report generation failed: %s", e)

        with st.spinner("📋 Building structured report..."):
            report = generate_json_report(
                machine_id=machine_id,
                sensor_readings=sensor_readings,
                anomaly_result=pred_scalar,
                retrieved_docs=retrieved_docs,
                llm_response=llm_response,
                experiment_id=exp_id,
            )
            st.session_state["generated_report"] = report

    # ── Display Report ────────────────────────────────────────────────────
    report = st.session_state.get("generated_report")
    if report:
        st.markdown("---")
        st.markdown("### 📊 Inspection Report Preview")

        # Status banner
        health = report.get("health_status", "NORMAL")
        badge_class = "badge-anomaly" if health == "ANOMALY" else "badge-normal"
        severity = report.get("risk_severity", "LOW")
        st.markdown(
            f"<span class='{badge_class}'>{health}</span> &nbsp; "
            f"**Risk:** {severity} &nbsp; "
            f"**Score:** {report.get('ensemble_score', 0):.4f} &nbsp; "
            f"**Votes:** {report.get('anomaly_votes_count', '0/4')}",
            unsafe_allow_html=True,
        )

        col_l, col_r = st.columns(2)
        with col_l:
            # Sensor readings table
            st.markdown("#### 📡 Sensor Readings")
            readings_df = pd.DataFrame(
                [(sensor_label(k), f"{v:.4f}") for k, v in report.get("sensor_readings", {}).items()],
                columns=["Sensor", "Value"],
            )
            st.dataframe(readings_df, use_container_width=True, hide_index=True)

            # Model votes
            st.markdown("#### 🤖 Model Votes")
            votes_data = []
            for m, vote in report.get("individual_model_votes", {}).items():
                score = report.get("individual_model_scores", {}).get(m, 0)
                votes_data.append({"Model": m, "Score": f"{score:.4f}", "Decision": vote})
            st.dataframe(pd.DataFrame(votes_data), use_container_width=True, hide_index=True)

        with col_r:
            # Possible causes
            st.markdown("#### 🔍 Possible Causes")
            for cause in report.get("possible_causes", []):
                st.markdown(f"• {cause}")

            st.markdown("#### 🔧 Recommended Actions")
            for i, action in enumerate(report.get("recommended_actions", []), 1):
                st.markdown(f"{i}. {action}")

        # References
        refs = report.get("supporting_references", [])
        if refs:
            st.markdown("#### 📚 Supporting References")
            for ref in refs:
                st.markdown(f"• {ref}")

        # Disclaimer
        st.markdown(
            f"<div class='warning-box'>⚠️ {report.get('disclaimer', '')}</div>",
            unsafe_allow_html=True,
        )

        # Downloads
        col1, col2 = st.columns(2)
        with col1:
            json_bytes = json.dumps(report, indent=2).encode("utf-8")
            st.download_button(
                "⬇️ Download JSON Report",
                data=json_bytes,
                file_name=f"inspection_report_{machine_id}.json",
                mime="application/json",
            )
        with col2:
            with st.spinner("Generating PDF..."):
                try:
                    pdf_bytes = generate_pdf_report(report)
                    st.download_button(
                        "⬇️ Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"inspection_report_{machine_id}.pdf",
                        mime="application/pdf",
                    )
                except Exception as e:
                    st.warning(f"PDF generation unavailable: {e}")

        # Raw JSON viewer
        with st.expander("🔎 View Raw JSON"):
            st.json(report)


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE 6: AGENTIC AI
# ═══════════════════════════════════════════════════════════════════════════

def page_agent():
    st.markdown("## 🔗 Agentic AI Predictive Maintenance")
    st.markdown(
        "An autonomous AI agent that sequentially calls specialized tools "
        "to analyze machine health and produce a prioritized maintenance plan."
    )
    st.markdown("""
    <div class='info-box'>
    <b>Agent Workflow:</b>
    Sensor Data → 🔍 Sensor Analysis → 🤖 Anomaly Detection →
    📚 RAG Retrieval → 📋 Inspection Report → 🔧 Maintenance Plan
    </div>
    """, unsafe_allow_html=True)

    bundle = load_model_bundle()
    with st.spinner("Loading RAG index..."):
        rag_index, rag_chunks, embed_model = load_rag_index()

    # ── Machine Config ────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        machine_id = st.text_input("Machine ID:", value="SKAB-PUMP-01", key="agent_machine")
    with col2:
        exp_id = st.text_input("Experiment ID:", value="valve1/1", key="agent_exp")

    # ── Sensor Input ──────────────────────────────────────────────────────
    st.markdown("#### 📡 Sensor Readings")
    base_feats = bundle.get("base_features") or BASE_SENSOR_FEATURES
    cols = st.columns(4)
    sensor_readings = {}
    defaults = {
        "Vibration1": 0.35, "Vibration2": 0.40, "Current": 2.8,
        "Pressure": 1.2, "Temperature": 88.5, "Thermocouple": 31.0,
        "Voltage": 215.0, "FlowRate": 25.0,
    }
    for i, feat in enumerate(base_feats):
        with cols[i % 4]:
            sensor_readings[feat] = st.number_input(
                sensor_label(feat),
                value=defaults.get(feat, 0.0),
                format="%.4f",
                key=f"agent_sensor_{feat}",
            )

    if st.button("🚀 Launch AI Agent", type="primary", key="launch_agent"):
        # Create step display containers
        step_containers = {}
        st.markdown("### 🤖 Agent Execution Log")
        for i in range(1, 6):
            step_containers[i] = st.empty()

        agent_result_container = st.empty()

        def step_callback(step):
            icon = {"done": "✅", "running": "⏳", "failed": "❌", "pending": "⬜"}.get(step.status, "⬜")
            time_str = ""
            if step.result:
                time_str = f" ({step.result.execution_time_ms:.0f}ms)"
            msg = f"{icon} **Step {step.step_number}: {step.tool_name}** – {step.description}{time_str}"
            if step.result and not step.result.success:
                msg += f"\n\n⚠️ Error: {step.result.error}"
            step_containers.get(step.step_number, st.empty()).markdown(msg)

        with st.spinner("⚙️ Agent is running..."):
            df_agent = pd.DataFrame([sensor_readings])
            agent = PredictiveMaintenanceAgent(
                bundle=bundle,
                rag_index=rag_index,
                rag_chunks=rag_chunks,
                embed_model=embed_model,
            )
            agent_result = agent.run(
                df=df_agent,
                machine_id=machine_id,
                experiment_id=exp_id,
                step_callback=step_callback,
            )
            st.session_state["agent_result"] = agent_result

    # ── Display Agent Results ─────────────────────────────────────────────
    agent_result = st.session_state.get("agent_result")
    if agent_result:
        st.markdown("---")
        st.markdown("### 📊 Agent Results")

        tabs = st.tabs([
            "🔍 Sensor Analysis",
            "🤖 Anomaly Detection",
            "📚 Retrieved Docs",
            "📋 Inspection Report",
            "🔧 Maintenance Plan",
        ])

        # Tab 1: Sensor Analysis
        with tabs[0]:
            sa = agent_result.get("sensor_analysis", {})
            overall = sa.get("overall_severity", "NORMAL")
            badge_class = {"NORMAL": "badge-normal", "MEDIUM": "badge-warning", "HIGH": "badge-anomaly"}.get(overall, "badge-normal")
            st.markdown(f"**Overall Severity:** <span class='{badge_class}'>{overall}</span>", unsafe_allow_html=True)
            st.markdown(f"**Summary:** {sa.get('summary', 'N/A')}")

            flagged = sa.get("flagged_sensors", [])
            if flagged:
                st.markdown("#### 🚨 Flagged Sensors")
                st.dataframe(pd.DataFrame(flagged), use_container_width=True, hide_index=True)
            else:
                st.success("All sensors within normal ranges.")

        # Tab 2: Anomaly Detection
        with tabs[1]:
            ad = agent_result.get("anomaly_detection", {})
            st.markdown(f"**Summary:** {ad.get('summary', 'N/A')}")

            ens_score = ad.get("ensemble_score", 0)
            ens_pred = ad.get("ensemble_prediction", 0)
            if isinstance(ens_score, list):
                ens_score = ens_score[0]
            if isinstance(ens_pred, list):
                ens_pred = ens_pred[0]

            col1, col2, col3 = st.columns(3)
            col1.metric("Ensemble Score", f"{ens_score:.4f}")
            col2.metric("Decision", "ANOMALY 🚨" if ens_pred else "NORMAL ✅")
            col3.metric("Models Voting Anomaly", f"{ad.get('anomaly_votes', [0])[0] if isinstance(ad.get('anomaly_votes'), list) else ad.get('anomaly_votes', 0)}/4")

            # Individual model scores
            ind_scores = ad.get("individual_scores", {})
            if ind_scores:
                score_data = []
                thr_data = ad.get("decision_thresholds", {})
                for m, s in ind_scores.items():
                    score_val = s[0] if isinstance(s, list) else s
                    thr_val = thr_data.get(m, 0)
                    score_data.append({
                        "Model": m,
                        "Score": round(float(score_val), 4),
                        "Threshold": round(float(thr_val), 4),
                        "Decision": "ANOMALY" if float(score_val) >= float(thr_val) else "NORMAL",
                    })
                st.dataframe(pd.DataFrame(score_data), use_container_width=True, hide_index=True)

        # Tab 3: Retrieved Documents
        with tabs[2]:
            rr = agent_result.get("rag_retrieval", {})
            st.markdown(f"**Summary:** {rr.get('summary', 'N/A')}")
            retrieved = rr.get("retrieved_chunks", [])
            for i, doc in enumerate(retrieved, 1):
                with st.expander(f"📄 Reference {i}: {doc.get('source', 'Unknown')} (p.{doc.get('page', '?')})"):
                    st.markdown(f"**Retrieval Score:** {doc.get('retrieval_score', 0):.4f}")
                    st.markdown(doc.get("text", ""))

        # Tab 4: Inspection Report
        with tabs[3]:
            ir = agent_result.get("inspection_report", {})
            report = ir.get("report", {})
            if report:
                health = report.get("health_status", "NORMAL")
                badge_class = "badge-anomaly" if health == "ANOMALY" else "badge-normal"
                st.markdown(f"**Health:** <span class='{badge_class}'>{health}</span>", unsafe_allow_html=True)
                st.markdown(f"**Risk Severity:** {report.get('risk_severity', 'N/A')}")

                for cause in report.get("possible_causes", []):
                    st.markdown(f"• {cause}")

                with st.expander("🔎 Full JSON Report"):
                    st.json(report)

                json_bytes = json.dumps(report, indent=2).encode("utf-8")
                st.download_button("⬇️ Download Report JSON", data=json_bytes,
                                   file_name="agent_report.json", mime="application/json")

        # Tab 5: Maintenance Plan
        with tabs[4]:
            mp = agent_result.get("maintenance_plan", {})
            if mp:
                urgency = mp.get("urgency_level", "ROUTINE")
                timeframe = mp.get("recommended_timeframe", "N/A")
                badge_class = {
                    "IMMEDIATE": "badge-anomaly",
                    "URGENT": "badge-anomaly",
                    "SOON": "badge-warning",
                    "ROUTINE": "badge-normal",
                }.get(urgency, "badge-normal")

                st.markdown(
                    f"**Urgency:** <span class='{badge_class}'>{urgency}</span> &nbsp; "
                    f"**Timeframe:** {timeframe}",
                    unsafe_allow_html=True,
                )

                actions = mp.get("prioritized_actions", [])
                if actions:
                    st.markdown("#### 🔧 Prioritized Maintenance Actions")
                    actions_df = pd.DataFrame(actions)
                    st.dataframe(actions_df, use_container_width=True, hide_index=True)

                st.markdown(
                    f"<div class='warning-box'>{mp.get('safety_note', '')}</div>",
                    unsafe_allow_html=True,
                )

                refs = mp.get("reference_guidance", [])
                if refs:
                    st.markdown("#### 📚 Reference Guidance")
                    for ref in refs:
                        with st.expander(f"{ref.get('source', 'Unknown')} (p.{ref.get('page', '?')})"):
                            st.markdown(ref.get("guidance", ""))


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE 7: SAMPLE DEMO
# ═══════════════════════════════════════════════════════════════════════════

def page_sample_demo():
    st.markdown("## 🧪 Sample Dataset Demonstration")
    st.markdown(
        "Load records from the SKAB test dataset, run the trained models, "
        "and compare predictions against actual anomaly labels."
    )

    bundle = load_model_bundle()

    with st.spinner("Loading SKAB dataset..."):
        df_full = load_skab_dataset()

    if df_full is None:
        st.markdown("""
        <div class='warning-box'>
        ⚠️ SKAB dataset not found. The system will attempt to download it from GitHub.
        Please ensure internet connectivity or place SKAB CSV files in the <code>data/skab/</code> folder.
        </div>
        """, unsafe_allow_html=True)
        return

    st.success(f"✅ SKAB dataset: {len(df_full):,} records from {df_full['experiment_id'].nunique() if 'experiment_id' in df_full.columns else '?'} experiments")

    # ── Sample Selection ──────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        n_records = st.select_slider(
            "Number of sample records:",
            options=[10, 20, 50, 100],
            value=50,
            key="demo_n",
        )
    with col2:
        balanced = st.checkbox("Balanced sample (equal normal/anomaly)", value=True, key="demo_balanced")

    if st.button("🎲 Load Sample Records", key="demo_load"):
        sample_df = sample_records(df_full, n=n_records, balanced=balanced)
        st.session_state["demo_sample"] = sample_df
        st.session_state["demo_pred"] = None

    sample_df = st.session_state.get("demo_sample")

    if sample_df is not None:
        st.markdown(f"**Loaded {len(sample_df)} records**")

        # Show raw data
        with st.expander("📋 View Sample Records"):
            display_cols = [c for c in (["datetime", "experiment_id"] +
                           [s for s in BASE_SENSOR_FEATURES if s in sample_df.columns] +
                           ["anomaly"]) if c in sample_df.columns]
            st.dataframe(sample_df[display_cols], use_container_width=True, hide_index=True)

        # Sensor time-series
        st.markdown("### 📈 Sensor Readings (Sample)")
        available_sensors = [s for s in BASE_SENSOR_FEATURES if s in sample_df.columns]
        if available_sensors:
            sel_sensor = st.selectbox("Sensor:", available_sensors, format_func=sensor_label, key="demo_sensor")
            fig_ts = go.Figure()
            if "anomaly" in sample_df.columns:
                for label, color in [(0, COLOR_NORMAL), (1, COLOR_ANOMALY)]:
                    subset = sample_df[sample_df["anomaly"] == label]
                    fig_ts.add_trace(go.Scatter(
                        x=subset.index, y=subset[sel_sensor],
                        mode="markers",
                        name=("Normal" if label == 0 else "Anomaly"),
                        marker=dict(color=color, size=7),
                    ))
            else:
                fig_ts.add_trace(go.Scatter(
                    x=sample_df.index, y=sample_df[sel_sensor],
                    mode="lines+markers", name=sel_sensor,
                    line=dict(color=COLOR_CYAN),
                ))
            fig_ts.update_layout(
                template=PLOTLY_TEMPLATE,
                paper_bgcolor=BG_COLOR,
                plot_bgcolor=BG_COLOR,
                height=350,
                margin=dict(l=20, r=20, t=20, b=40),
                legend=dict(bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_ts, use_container_width=True)

        # ── Run Prediction ────────────────────────────────────────────────
        if st.button("🚀 Run Anomaly Detection on Sample", type="primary", key="demo_run"):
            with st.spinner("Running ensemble models on sample..."):
                try:
                    result = predict_anomalies(sample_df, bundle)
                    st.session_state["demo_pred"] = result
                except Exception as e:
                    st.error(f"Prediction error: {e}")

        demo_pred = st.session_state.get("demo_pred")

        if demo_pred is not None:
            st.markdown("---")
            st.markdown("### 🤖 Prediction Results")

            df_res = demo_pred["df_results"]
            ens_pred = demo_pred["ensemble_prediction"]
            ens_score = demo_pred["ensemble_score"]

            if hasattr(ens_pred, "__len__"):
                n_pred_anom = int(sum(ens_pred))
                preds_list = [int(p) for p in ens_pred]
                scores_list = [float(s) for s in ens_score]
            else:
                n_pred_anom = int(ens_pred)
                preds_list = [int(ens_pred)]
                scores_list = [float(ens_score)]

            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Records", len(sample_df))
            col2.metric("Predicted Normal", len(preds_list) - n_pred_anom)
            col3.metric("Predicted Anomaly", n_pred_anom)
            if "anomaly" in sample_df.columns:
                actual_anom = int(sample_df["anomaly"].sum())
                col4.metric("Actual Anomalies", actual_anom)

            # Anomaly score distribution
            st.markdown("### 📊 Anomaly Score Distribution")
            fig_hist = px.histogram(
                x=scores_list,
                nbins=20,
                labels={"x": "Ensemble Anomaly Score", "y": "Count"},
                color_discrete_sequence=[COLOR_CYAN],
            )
            thr_val = demo_pred["decision_thresholds"].get("EqualWeightEnsemble", 0.338)
            fig_hist.add_vline(
                x=thr_val, line_dash="dash", line_color=COLOR_WARNING,
                annotation_text=f"Threshold: {thr_val:.3f}",
            )
            fig_hist.update_layout(
                template=PLOTLY_TEMPLATE,
                paper_bgcolor=BG_COLOR,
                plot_bgcolor=BG_COLOR,
                height=300,
                margin=dict(l=20, r=20, t=20, b=40),
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            # Actual vs predicted
            if "anomaly" in sample_df.columns and len(preds_list) == len(sample_df):
                y_true = sample_df["anomaly"].values
                y_pred = np.array(preds_list)

                from sklearn.metrics import (
                    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
                )

                st.markdown("### ✅ Actual vs Predicted Comparison")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.3f}")
                col2.metric("Precision", f"{precision_score(y_true, y_pred, zero_division=0):.3f}")
                col3.metric("Recall", f"{recall_score(y_true, y_pred, zero_division=0):.3f}")
                col4.metric("F1 Score", f"{f1_score(y_true, y_pred, zero_division=0):.3f}")

                # Confusion matrix
                cm = confusion_matrix(y_true, y_pred)
                fig_cm = px.imshow(
                    cm, text_auto=True,
                    x=["Pred Normal", "Pred Anomaly"],
                    y=["Actual Normal", "Actual Anomaly"],
                    color_continuous_scale=[[0, BG_COLOR], [1, COLOR_CYAN]],
                    title="Confusion Matrix (Ensemble)",
                )
                fig_cm.update_layout(
                    template=PLOTLY_TEMPLATE,
                    paper_bgcolor=BG_COLOR,
                    height=300,
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                st.plotly_chart(fig_cm, use_container_width=True)

                # Side-by-side bar
                label_data = pd.DataFrame({
                    "Type": ["Actual Normal", "Actual Anomaly", "Pred Normal", "Pred Anomaly"],
                    "Count": [
                        int((y_true == 0).sum()), int((y_true == 1).sum()),
                        int((y_pred == 0).sum()), int((y_pred == 1).sum()),
                    ],
                    "Category": ["Actual", "Actual", "Predicted", "Predicted"],
                })
                fig_comp = px.bar(
                    label_data, x="Type", y="Count", color="Category",
                    barmode="group",
                    color_discrete_map={"Actual": COLOR_CYAN, "Predicted": COLOR_ANOMALY},
                    text_auto=True,
                )
                fig_comp.update_layout(
                    template=PLOTLY_TEMPLATE,
                    paper_bgcolor=BG_COLOR,
                    plot_bgcolor=BG_COLOR,
                    height=300,
                    margin=dict(l=20, r=20, t=20, b=40),
                )
                st.plotly_chart(fig_comp, use_container_width=True)

            # Download results
            csv_buf = df_res.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Sample Prediction Results CSV",
                data=csv_buf,
                file_name="sample_predictions.csv",
                mime="text/csv",
            )


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    page = render_sidebar()

    if page == "dashboard":
        page_dashboard()
    elif page == "monitoring":
        page_monitoring()
    elif page == "anomaly":
        page_anomaly_detection()
    elif page == "chat":
        page_chat()
    elif page == "report":
        page_report()
    elif page == "agent":
        page_agent()
    elif page == "demo":
        page_sample_demo()


if __name__ == "__main__":
    main()
