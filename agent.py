"""
agent.py – Agentic AI Predictive Maintenance with Tool Calling
Autonomous multi-tool workflow:
Tool 1: Sensor Analysis -> Tool 2: Anomaly Detection -> Tool 3: RAG Retrieval
-> Tool 4: Inspection Report -> Tool 5: Action Planning
"""

import time
import logging
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Callable

import pandas as pd
from prediction import predict_anomalies
from rag_engine import retrieve, format_retrieved_context
from report_generator import generate_json_report

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    tool_name: str
    success: bool
    output: Dict[str, Any]
    execution_time_ms: float
    error: Optional[str] = None


@dataclass
class AgentStep:
    step_number: int
    tool_name: str
    description: str
    result: Optional[ToolResult] = None
    status: str = "pending"


def sensor_analysis_tool(sensor_readings: Dict[str, float]) -> Dict[str, Any]:
    """Tool 1: Evaluates physical design boundaries and flags deviations."""
    limits = {
        "Vibration1": (0.015, 0.080, "g RMS", "Bearing / Rotor Dynamics"),
        "Vibration2": (0.015, 0.090, "g RMS", "Casing Structural Vibration"),
        "Current":    (0.50,  2.20,  "A",     "Motor Electrical Stator Load"),
        "Pressure":   (-0.50, 0.45,  "bar",   "Hydraulic Loop Pressure"),
        "Temperature":(65.0,  82.0,  "°C",    "Fluid / Heat Exchanger Temp"),
        "Thermocouple":(22.0, 31.0,  "°C",    "Ambient / Core Thermocouple"),
        "Voltage":    (210.0, 245.0, "V",     "AC Power Supply Quality"),
        "FlowRate":   (26.0,  110.0, "L/min", "Volumetric Flow Delivery"),
    }

    flagged = []
    normal = []
    for s, val in sensor_readings.items():
        if s in limits:
            lo, hi, unit, desc = limits[s]
            if val < lo or val > hi:
                dev = ((val - hi) / hi * 100) if val > hi else ((lo - val) / lo * 100)
                flagged.append({
                    "sensor": s,
                    "measured": round(val, 4),
                    "unit": unit,
                    "safe_range": f"{lo}–{hi} {unit}",
                    "subsystem": desc,
                    "deviation_pct": f"+{abs(dev):.1f}%" if val > hi else f"-{abs(dev):.1f}%",
                    "severity": "CRITICAL" if abs(dev) > 60 else "WARNING"
                })
            else:
                normal.append({"sensor": s, "measured": round(val, 4), "unit": unit})

    sev = "CRITICAL" if any(f["severity"] == "CRITICAL" for f in flagged) else ("WARNING" if flagged else "NOMINAL")

    return {
        "status": sev,
        "flagged_sensors": flagged,
        "nominal_sensors_count": len(normal),
        "abnormal_sensors_count": len(flagged),
        "total_analyzed": len(sensor_readings),
        "summary": f"Analyzed {len(sensor_readings)} channels. {len(flagged)} telemetry parameters exceeded safe design boundaries." if flagged else "All telemetry parameters are tracking safely within design boundaries."
    }


def anomaly_detection_tool(df: pd.DataFrame, bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Tool 2: Runs the trained 4-model ensemble."""
    res = predict_anomalies(df, bundle)

    ens_score = float(res["ensemble_score"][0])
    ens_pred = int(res["ensemble_prediction"][0])
    votes = int(res["anomaly_votes"][0])
    thr = res["decision_thresholds"]

    models_eval = {}
    for m in ["IsolationForest", "KNN", "PCA", "Autoencoder"]:
        sc = float(res["individual_scores"][m][0])
        pr = int(res["individual_predictions"][m][0])
        models_eval[m] = {
            "score": round(sc, 4),
            "threshold": round(float(thr.get(m, 0.5)), 4),
            "decision": "ANOMALY" if pr else "NORMAL",
        }

    return {
        "ensemble_decision": "ANOMALY" if ens_pred else "NORMAL",
        "ensemble_score": round(ens_score, 4),
        "decision_threshold": round(float(thr.get("EqualWeightEnsemble", 0.338)), 4),
        "anomaly_votes": f"{votes}/4 models",
        "models_breakdown": models_eval,
    }


def rag_retrieval_tool(query: str, index, chunks, embed_model) -> Dict[str, Any]:
    """Tool 3: Queries FAISS vector index of DOE/SKF guides."""
    retrieved = retrieve(query, top_k=3, index=index, chunks=chunks, embed_model=embed_model)
    return {
        "query": query,
        "docs_retrieved": [
            {
                "source": c.get("source", "Industrial Standard"),
                "page": c.get("page", "?"),
                "relevance_score": round(float(c.get("retrieval_score", 0.95)), 3),
                "excerpt": c.get("text", "")[:180] + "..."
            }
            for c in retrieved
        ]
    }


def inspection_report_tool(machine_id: str, readings: Dict[str, float], anom_res: Dict[str, Any], retrieved_docs: List[Dict], exp_id: str) -> Dict[str, Any]:
    """Tool 4: Generates structured inspection findings."""
    report = generate_json_report(
        machine_id=machine_id,
        sensor_readings=readings,
        anomaly_result=anom_res,
        retrieved_docs=retrieved_docs,
        experiment_id=exp_id
    )
    return {"report": report}


def maintenance_planning_tool(sensor_diag: Dict[str, Any], anom_res: Dict[str, Any], machine_id: str) -> Dict[str, Any]:
    """Tool 5: Synthesizes concrete prioritized action steps."""
    flagged = sensor_diag.get("flagged_sensors", [])
    is_anomaly = anom_res.get("ensemble_decision") == "ANOMALY"
    votes = anom_res.get("anomaly_votes", "0/4")

    actions = []
    p = 1

    if flagged:
        for f in flagged:
            s = f["sensor"]
            if "Vibration" in s:
                actions.append({
                    "priority": f"P{p}",
                    "action": "Immediate Bearing FFT Spectral Analysis & Laser Shaft Alignment Verification",
                    "reason": f"Triggered by {s} at {f['measured']} {f['unit']} ({f['deviation_pct']} over safe limit). Prevents mechanical raceway spalling and casing fracture.",
                    "responsible": "Mechanical Reliability Team",
                    "urgency": "URGENT (Within 8–12 Hours)"
                })
                p += 1
            elif "Current" in s:
                actions.append({
                    "priority": f"P{p}",
                    "action": "Motor Stator Insulation Megger Test & Drive Load Verification",
                    "reason": f"Triggered by Current draw at {f['measured']}A. High current accelerates winding insulation thermal breakdown.",
                    "responsible": "Electrical Maintenance Team",
                    "urgency": "HIGH (Within 12–24 Hours)"
                })
                p += 1
            elif "Pressure" in s:
                actions.append({
                    "priority": f"P{p}",
                    "action": "Downstream Valve Throttling Inspection & Line Strainer Purge",
                    "reason": f"System pressure at {f['measured']} bar exceeds design bounds, inducing pipe stress and pump cavitation.",
                    "responsible": "Process Operations Team",
                    "urgency": "MEDIUM (Within 24 Hours)"
                })
                p += 1
            elif "Temperature" in s:
                actions.append({
                    "priority": f"P{p}",
                    "action": "Cooling Loop Heat Exchanger Descaling & Lube Oil Viscosity Check",
                    "reason": f"Fluid temperature reached {f['measured']}°C, degrading lubricating oil film boundary layer.",
                    "responsible": "Mechanical Maintenance Team",
                    "urgency": "MEDIUM (Within 24 Hours)"
                })
                p += 1
            elif "FlowRate" in s:
                actions.append({
                    "priority": f"P{p}",
                    "action": "Pump Impeller Cavitation Check & Inlet Suction Strainer Cleaning",
                    "reason": f"Flow delivery at {f['measured']} L/min indicates suction starved cavitation.",
                    "responsible": "Process Engineering Team",
                    "urgency": "HIGH (Within 12 Hours)"
                })
                p += 1

    if not actions:
        actions.append({
            "priority": "P1",
            "action": "Routine Telemetry Logging & Baseline Trend Verification",
            "reason": "All 8 sensor channels are operating safely within nominal boundaries.",
            "responsible": "Shift Operations Supervisor",
            "urgency": "ROUTINE (Next Scheduled Preventive Window)"
        })

    urgency_level = "CRITICAL / URGENT" if (is_anomaly and len(flagged) >= 2) else ("ELEVATED WARNING" if is_anomaly else "NORMAL / ROUTINE")

    return {
        "machine_id": machine_id,
        "urgency_level": urgency_level,
        "recommended_timeframe": "Immediate (Within 4–12 Hours)" if is_anomaly else "Routine (Standard 30-Day PM Window)",
        "anomaly_votes": votes,
        "prioritized_actions": actions,
        "safety_note": "⚠️ Safety Directive: Artificial Intelligence guidance must be physically confirmed by qualified engineering staff prior to locking out machinery or altering process parameters."
    }


class PredictiveMaintenanceAgent:
    def __init__(self, bundle: Dict[str, Any], rag_index, rag_chunks, embed_model):
        self.bundle = bundle
        self.rag_index = rag_index
        self.rag_chunks = rag_chunks
        self.embed_model = embed_model
        self.steps: List[AgentStep] = []

    def run(self, df: pd.DataFrame, machine_id: str, experiment_id: str = "N/A") -> Dict[str, Any]:
        self.steps = []
        readings = {col: float(df[col].iloc[0]) for col in df.columns if col in [
            "Vibration1", "Vibration2", "Current", "Pressure",
            "Temperature", "Thermocouple", "Voltage", "FlowRate"
        ]}

        # Step 1: Sensor Analysis
        s1 = AgentStep(1, "SensorAnalysisTool", "Validates 8 sensor channels against baseline design limits")
        t0 = time.time()
        r1_out = sensor_analysis_tool(readings)
        s1.result = ToolResult("SensorAnalysisTool", True, r1_out, round((time.time() - t0)*1000, 1))
        s1.status = "done"
        self.steps.append(s1)

        # Step 2: Anomaly Detection
        s2 = AgentStep(2, "AnomalyDetectionTool", "Executes Isolation Forest, KNN, PCA, Autoencoder & Ensemble")
        t0 = time.time()
        r2_out = anomaly_detection_tool(df, self.bundle)
        s2.result = ToolResult("AnomalyDetectionTool", True, r2_out, round((time.time() - t0)*1000, 1))
        s2.status = "done"
        self.steps.append(s2)

        # Step 3: RAG Retrieval
        s3 = AgentStep(3, "RAGRetrievalTool", "Searches FAISS index of technical literature")
        t0 = time.time()
        flagged_names = [f["sensor"] for f in r1_out.get("flagged_sensors", [])]
        query = f"maintenance guidance for {' '.join(flagged_names)}" if flagged_names else "routine industrial motor pump maintenance guidelines"
        r3_out = rag_retrieval_tool(query, self.rag_index, self.rag_chunks, self.embed_model)
        s3.result = ToolResult("RAGRetrievalTool", True, r3_out, round((time.time() - t0)*1000, 1))
        s3.status = "done"
        self.steps.append(s3)

        # Step 4: Inspection Report
        s4 = AgentStep(4, "InspectionReportTool", "Compiles comprehensive JSON inspection report")
        t0 = time.time()
        # Structure compatibility dict
        anom_compact = {
            "individual_scores": r2_out["models_breakdown"],
            "individual_predictions": {m: 1 if v["decision"] == "ANOMALY" else 0 for m, v in r2_out["models_breakdown"].items()},
            "ensemble_score": r2_out["ensemble_score"],
            "ensemble_prediction": 1 if r2_out["ensemble_decision"] == "ANOMALY" else 0,
            "anomaly_votes": int(r2_out["anomaly_votes"].split("/")[0]),
            "decision_thresholds": {"EqualWeightEnsemble": r2_out["decision_threshold"]}
        }
        r4_out = inspection_report_tool(machine_id, readings, anom_compact, r3_out["docs_retrieved"], experiment_id)
        s4.result = ToolResult("InspectionReportTool", True, r4_out, round((time.time() - t0)*1000, 1))
        s4.status = "done"
        self.steps.append(s4)

        # Step 5: Maintenance Planning
        s5 = AgentStep(5, "MaintenancePlanningTool", "Synthesizes prioritized action matrix with urgency and team roles")
        t0 = time.time()
        r5_out = maintenance_planning_tool(r1_out, r2_out, machine_id)
        s5.result = ToolResult("MaintenancePlanningTool", True, r5_out, round((time.time() - t0)*1000, 1))
        s5.status = "done"
        self.steps.append(s5)

        return {
            "success": True,
            "steps": self.steps,
            "sensor_analysis": r1_out,
            "anomaly_detection": r2_out,
            "rag_retrieval": r3_out,
            "inspection_report": r4_out,
            "maintenance_plan": r5_out,
        }
