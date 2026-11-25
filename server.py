"""
server.py – FastAPI Backend Server for Smart Industrial Energy AI
Supports multi-machine selection, multi-model anomaly detection, root cause diagnostics,
autonomous agentic pipeline, and AI maintenance copilot chat.
"""

import io
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from model_loader import load_model_bundle, models_available
from prediction import predict_anomalies
from rag_engine import load_rag_index, retrieve, format_retrieved_context, BUILTIN_KNOWLEDGE
from llm_engine import generate_response, build_chat_prompt, MAINTENANCE_SYSTEM_PROMPT
from report_generator import generate_json_report, generate_pdf_report
from agent import PredictiveMaintenanceAgent
from data_loader import load_skab_dataset, get_dataset_stats, sample_records
from utils import sensor_label, BASE_SENSOR_FEATURES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Smart Industrial Energy AI",
    description="Multi-Machine Anomaly Detection, Diagnostics, Agentic AI, and RAG Copilot",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load In-Memory Resources ────────────────────────────────────────────────
bundle = load_model_bundle()
rag_index, rag_chunks, embed_model = load_rag_index()
dataset_df: Optional[pd.DataFrame] = None

def get_or_load_dataset():
    global dataset_df
    if dataset_df is None:
        dataset_df = load_skab_dataset()
    return dataset_df

# ── Predefined Machines for College Demonstration ───────────────────────────
PREDEFINED_MACHINES = {
    "M-101": {
        "id": "M-101",
        "name": "Machine A: Primary Circulation Pump #1",
        "type": "Centrifugal Water Pump",
        "default_readings": {
            "Vibration1": 0.0270, "Vibration2": 0.0396, "Current": 1.3012,
            "Pressure": 0.0547, "Temperature": 75.5445, "Thermocouple": 25.8408,
            "Voltage": 224.170, "FlowRate": 32.000
        }
    },
    "M-102": {
        "id": "M-102",
        "name": "Machine B: Main Drive Induction Motor",
        "type": "3-Phase Induction Motor (45 kW)",
        "default_readings": {
            "Vibration1": 0.0265, "Vibration2": 0.0382, "Current": 1.2850,
            "Pressure": 0.0540, "Temperature": 74.8000, "Thermocouple": 25.6000,
            "Voltage": 230.500, "FlowRate": 31.800
        }
    },
    "M-103": {
        "id": "M-103",
        "name": "Machine C: High-Pressure Feed Valve & Actuator",
        "type": "Pneumatic Control Valve",
        "default_readings": {
            "Vibration1": 0.0280, "Vibration2": 0.0410, "Current": 1.3200,
            "Pressure": 0.0620, "Temperature": 76.1000, "Thermocouple": 26.0000,
            "Voltage": 222.800, "FlowRate": 32.200
        }
    },
    "M-104": {
        "id": "M-104",
        "name": "Machine D: Auxiliary Cooling Compressor",
        "type": "Reciprocating Chiller Compressor",
        "default_readings": {
            "Vibration1": 0.0290, "Vibration2": 0.0425, "Current": 1.3500,
            "Pressure": 0.0580, "Temperature": 75.2000, "Thermocouple": 25.7000,
            "Voltage": 225.000, "FlowRate": 31.500
        }
    }
}

# ── Schemas ──────────────────────────────────────────────────────────────────
class MachineAnalysisRequest(BaseModel):
    machine_id: str
    sensor_readings: Dict[str, float]

class MultiMachineAnalysisRequest(BaseModel):
    machines: List[MachineAnalysisRequest]

class ChatRequest(BaseModel):
    message: str
    machine_id: str = "Machine A: Primary Circulation Pump #1"
    sensor_readings: Optional[Dict[str, float]] = None

class AgentRequest(BaseModel):
    machine_id: str = "M-101"
    experiment_id: str = "valve1/1"
    sensor_readings: Dict[str, float]

# ── Root Cause & Action Generator Function ──────────────────────────────────
def generate_root_cause_and_action(row_dict: Dict[str, float], is_anomaly: bool, votes: int, score: float):
    """
    Computes human-interpretable 'Why it is happening' and 'What to do and why'.
    """
    reasons = []
    actions = []
    flagged_sensors = []

    v1, v2 = row_dict.get("Vibration1", 0), row_dict.get("Vibration2", 0)
    curr = row_dict.get("Current", 0)
    press = row_dict.get("Pressure", 0)
    temp = row_dict.get("Temperature", 0)
    volt = row_dict.get("Voltage", 0)
    flow = row_dict.get("FlowRate", 0)

    if v1 > 0.08 or v2 > 0.09:
        flagged_sensors.append("Vibration")
        reasons.append(f"Accelerometers indicate high harmonic energy (V1: {v1:.4f}g, V2: {v2:.4f}g), typical of bearing ball spalling, dynamic shaft misalignment, or rotor eccentricity.")
        actions.append({
            "action": "Perform bearing vibration spectral analysis (FFT) and check shaft laser alignment.",
            "why": "Unchecked vibration leads to catastrophic mechanical fatigue and catastrophic casing crack within 48–72 operating hours.",
            "priority": "HIGH"
        })

    if curr > 2.2:
        flagged_sensors.append("Current")
        reasons.append(f"Motor current draw ({curr:.3f}A) is elevated above normal operational envelope (0.5–2.0A), pointing to high mechanical resistance, pump rotor binding, or stator winding degradation.")
        actions.append({
            "action": "Check motor winding insulation resistance (Megger test) and verify terminal box lug torque.",
            "why": "Overcurrent accelerates thermal winding breakdown, reducing motor insulation lifespan by 50% for every 10°C rise.",
            "priority": "HIGH"
        })

    if press > 0.5:
        flagged_sensors.append("Pressure")
        reasons.append(f"Discharge pressure ({press:.3f} bar) exceeds normal baseline, indicating downstream valve blockage, line restriction, or hydraulic throttling.")
        actions.append({
            "action": "Inspect downstream control valves, clean inline pipe strainers, and check valve limit travel.",
            "why": "Pressure spikes trigger pipe seal degradation and increase pump cavitation risk.",
            "priority": "MEDIUM"
        })

    if temp > 82.0:
        flagged_sensors.append("Temperature")
        reasons.append(f"Fluid temperature ({temp:.1f}°C) is above the 80°C threshold, suggesting cooling loop fouling, lubricant degradation, or excessive friction.")
        actions.append({
            "action": "Flush heat exchanger plates, verify cooling water flow rate, and inspect bearing oil clarity.",
            "why": "High temperature breaks down lubrication viscosity, resulting in metal-to-metal boundary friction.",
            "priority": "MEDIUM"
        })

    if flow < 25.0:
        flagged_sensors.append("FlowRate")
        reasons.append(f"Volume flow rate ({flow:.1f} L/min) has dropped below design baseline, indicating suction cavitation, pump wear ring erosion, or partial inlet valve closure.")
        actions.append({
            "action": "Inspect pump impeller clearance and check suction strainer for debris clogging.",
            "why": "Low flow can cause pump dry-running, destroying mechanical face seals.",
            "priority": "HIGH"
        })

    if not is_anomaly:
        why_text = "All primary sensor readings (Vibration, Current, Pressure, Temperature, Voltage, Flow Rate) are tracking safely within nominal baseline tolerances. Multi-model covariance is stable."
        what_to_do = [{
            "action": "Continue standard predictive monitoring schedule. Log current telemetry in maintenance record.",
            "why": "Normal baseline verified across Isolation Forest, KNN, PCA, and PyTorch Autoencoder.",
            "priority": "ROUTINE"
        }]
    else:
        if not reasons:
            reasons.append("The 4-model ensemble detected correlated statistical shifts across multivariate features that deviate from normal training distributions.")
            actions.append({
                "action": "Perform visual, thermal, and acoustic walk-around inspection.",
                "why": "Early-stage multi-parameter deviation detected prior to threshold exceedance.",
                "priority": "MEDIUM"
            })
        why_text = " ".join(reasons)
        what_to_do = actions

    return {
        "why_it_is_happening": why_text,
        "what_to_do_and_why": what_to_do,
        "flagged_sensors": flagged_sensors,
    }

# ── API Endpoints ────────────────────────────────────────────────────────────

@app.get("/api/machines")
def get_machines_list():
    """Returns available industrial machines with their default baseline sensor values."""
    return list(PREDEFINED_MACHINES.values())

@app.get("/api/status")
def get_status():
    return {
        "status": "online",
        "models_loaded": models_available(bundle),
        "total_models": 4,
        "ensemble": "EqualWeightEnsemble",
    }

@app.post("/api/diagnose-machine")
def diagnose_machine(req: MachineAnalysisRequest):
    """
    Evaluates one machine: runs all 4 models, combines them into the ensemble,
    and returns 'Why it is happening' and 'What to do and why'.
    """
    row_dict = req.sensor_readings
    df = pd.DataFrame([row_dict])
    res = predict_anomalies(df, bundle)

    ens_score = float(res["ensemble_score"][0])
    ens_pred = int(res["ensemble_prediction"][0])
    votes = int(res["anomaly_votes"][0])
    thr = res["decision_thresholds"]

    model_breakdown = {}
    for m in ["IsolationForest", "KNN", "PCA", "Autoencoder"]:
        sc = float(res["individual_scores"][m][0])
        pr = int(res["individual_predictions"][m][0])
        model_breakdown[m] = {
            "score": round(sc, 5),
            "threshold": round(float(thr.get(m, 0.5)), 5),
            "decision": "ANOMALY" if pr else "NORMAL",
            "vote": pr,
        }

    is_anomaly = bool(ens_pred)
    root_cause = generate_root_cause_and_action(row_dict, is_anomaly, votes, ens_score)

    machine_info = PREDEFINED_MACHINES.get(req.machine_id, {
        "id": req.machine_id,
        "name": req.machine_id,
        "type": "Industrial Machine"
    })

    return {
        "machine": machine_info,
        "sensor_readings": row_dict,
        "is_anomaly": is_anomaly,
        "status": "ANOMALY DETECTED" if is_anomaly else "NORMAL (HEALTHY)",
        "ensemble_score": round(ens_score, 5),
        "ensemble_threshold": round(float(thr.get("EqualWeightEnsemble", 0.338)), 5),
        "anomaly_votes": votes,
        "total_models": 4,
        "model_breakdown": model_breakdown,
        "why_it_is_happening": root_cause["why_it_is_happening"],
        "what_to_do_and_why": root_cause["what_to_do_and_why"],
        "flagged_sensors": root_cause["flagged_sensors"],
    }

@app.post("/api/diagnose-multi")
def diagnose_multi_machines(req: MultiMachineAnalysisRequest):
    """Analyzes multiple machines side-by-side for comparison."""
    results = []
    for m in req.machines:
        res = diagnose_machine(m)
        results.append(res)
    return {"count": len(results), "machines": results}

@app.post("/api/chat")
def chat_copilot(req: ChatRequest):
    """
    RAG-grounded maintenance copilot. Always responds instantly with technical insights.
    """
    sensor_readings = req.sensor_readings or {
        "Vibration1": 0.027, "Vibration2": 0.040, "Current": 1.3,
        "Pressure": 0.055, "Temperature": 75.5, "Thermocouple": 25.8,
        "Voltage": 230.0, "FlowRate": 32.0,
    }

    df_single = pd.DataFrame([sensor_readings])
    pred = predict_anomalies(df_single, bundle)
    anomaly_result = {
        "ensemble_score": float(pred["ensemble_score"][0]),
        "ensemble_prediction": int(pred["ensemble_prediction"][0]),
        "anomaly_votes": int(pred["anomaly_votes"][0]),
    }

    # Retrieve matching chunks
    retrieved = retrieve(req.message, top_k=3, index=rag_index, chunks=rag_chunks, embed_model=embed_model)
    docs_ctx = format_retrieved_context(retrieved)
    sensor_ctx = "\n".join(f"  {sensor_label(k)}: {v:.4f}" for k, v in sensor_readings.items())

    user_prompt = build_chat_prompt(
        user_question=req.message,
        sensor_context=sensor_ctx,
        retrieved_docs=docs_ctx,
        anomaly_result=anomaly_result,
    )

    response_text = generate_response(
        system_prompt=MAINTENANCE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        user_question=req.message,
        sensor_readings=sensor_readings,
        anomaly_result=anomaly_result,
        retrieved_chunks=retrieved,
    )

    citations = [
        {
            "source": r.get("source", "Technical Manual"),
            "page": r.get("page", "?"),
            "score": round(float(r.get("retrieval_score", 0.0)), 4),
            "snippet": r.get("text", "")[:180] + "...",
        }
        for r in retrieved
    ]

    return {
        "reply": response_text,
        "citations": citations,
        "machine_health": "ANOMALY" if anomaly_result["ensemble_prediction"] else "NORMAL",
        "ensemble_score": anomaly_result["ensemble_score"],
    }

@app.post("/api/agent")
def run_agent(req: AgentRequest):
    """Executes the 5-step autonomous agent workflow."""
    df_agent = pd.DataFrame([req.sensor_readings])
    agent = PredictiveMaintenanceAgent(
        bundle=bundle,
        rag_index=rag_index,
        rag_chunks=rag_chunks,
        embed_model=embed_model,
    )
    result = agent.run(
        df=df_agent,
        machine_id=req.machine_id,
        experiment_id=req.experiment_id,
    )

    steps_log = []
    for s in result["steps"]:
        steps_log.append({
            "step_number": s.step_number,
            "tool_name": s.tool_name,
            "description": s.description,
            "status": s.status,
            "execution_time_ms": s.result.execution_time_ms if s.result else 0.0,
            "error": s.result.error if s.result else None,
        })

    return {
        "success": result["success"],
        "steps": steps_log,
        "sensor_analysis": result.get("sensor_analysis", {}),
        "anomaly_detection": result.get("anomaly_detection", {}),
        "rag_retrieval": result.get("rag_retrieval", {}),
        "inspection_report": result.get("inspection_report", {}).get("report", {}),
        "maintenance_plan": result.get("maintenance_plan", {}),
    }

@app.post("/api/report")
def get_machine_report(req: MachineAnalysisRequest):
    """Generates structured JSON inspection report for a machine."""
    row_dict = req.sensor_readings
    df = pd.DataFrame([row_dict])
    pred = predict_anomalies(df, bundle)
    
    anomaly_result = {
        "ensemble_score": float(pred["ensemble_score"][0]),
        "ensemble_prediction": int(pred["ensemble_prediction"][0]),
        "anomaly_votes": int(pred["anomaly_votes"][0]),
        "individual_scores": {
            m: {
                "score": round(float(pred["individual_scores"][m][0]), 4),
                "threshold": round(float(pred["decision_thresholds"].get(m, 0.5)), 4),
                "prediction": int(pred["individual_predictions"][m][0]),
                "status": "ANOMALY" if int(pred["individual_predictions"][m][0]) else "NORMAL"
            }
            for m in ["IsolationForest", "KNN", "PCA", "Autoencoder"]
        }
    }
    
    # Retrieve technical knowledge
    query = "vibration motor bearing failure maintenance procedure" if anomaly_result["ensemble_prediction"] else "routine preventive maintenance inspection"
    retrieved = retrieve(query, top_k=3, index=rag_index, chunks=rag_chunks, embed_model=embed_model)
    
    report = generate_json_report(
        machine_id=req.machine_id,
        sensor_readings=row_dict,
        anomaly_result=anomaly_result,
        retrieved_docs=retrieved,
        experiment_id="college-defense-demo",
    )
    return report

@app.post("/api/download-pdf")
def download_pdf_report(req: MachineAnalysisRequest):
    """Generates and downloads a complete formatted PDF inspection report."""
    report_data = get_machine_report(req)
    pdf_bytes = generate_pdf_report(report_data)
    
    clean_id = req.machine_id.replace(" ", "_").replace(":", "_").replace("#", "")
    filename = f"Inspection_Report_{clean_id}.pdf"
    
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Access-Control-Expose-Headers": "Content-Disposition"
        }
    )

# ── Mount Frontend ───────────────────────────────────────────────────────────
FRONTEND_DIR = Path(__file__).parent / "frontend"
FRONTEND_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
