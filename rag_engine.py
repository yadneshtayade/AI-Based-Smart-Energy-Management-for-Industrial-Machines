"""
rag_engine.py – FAISS-based Retrieval-Augmented Generation engine.

Loads or builds a FAISS index from chunked technical documents.
Expects either:
  - models/faiss_index/  folder with index.faiss + chunks.json, OR
  - models/rag_chunks.pkl  containing {"chunks": [...], "embeddings": np.ndarray}
"""

import os
import json
import pickle
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent / "models"
FAISS_DIR = MODEL_DIR / "faiss_index"
RAG_PKL   = MODEL_DIR / "rag_chunks.pkl"

# ─────────────────────────── Built-in Knowledge Base ────────────────────────
# Fallback knowledge when no FAISS index is found.

BUILTIN_KNOWLEDGE = [
    {
        "id": 0,
        "source": "DOE Motor Systems Market Assessment",
        "page": 1,
        "text": (
            "Abnormal vibration in industrial motors often indicates bearing wear, "
            "misalignment, or rotor imbalance. Accelerometer RMS values exceeding "
            "twice the baseline suggest immediate inspection is warranted. Regular "
            "vibration monitoring can extend motor life by detecting issues early."
        ),
    },
    {
        "id": 1,
        "source": "SKF Bearing Maintenance Guide",
        "page": 3,
        "text": (
            "Bearing failure accounts for approximately 40% of motor failures. "
            "Early indicators include increased vibration amplitude, temperature rise, "
            "and unusual noise. The recommended maintenance interval is every 2000 "
            "operating hours for standard bearings. Lubrication should be checked "
            "every 500 hours in high-temperature environments."
        ),
    },
    {
        "id": 2,
        "source": "DOE Pumping Systems Best Practices",
        "page": 7,
        "text": (
            "High pressure differentials in pump systems can indicate blockages, "
            "valve malfunctions, or impeller wear. Pressure deviations greater than "
            "15% from nominal operating pressure should trigger a maintenance alert. "
            "Cavitation, identified by rapid pressure fluctuations and noise, causes "
            "accelerated impeller erosion."
        ),
    },
    {
        "id": 3,
        "source": "DOE Motor Systems Market Assessment",
        "page": 12,
        "text": (
            "Motor current anomalies can signal overloading, insulation degradation, "
            "or phase imbalance. Current exceeding 110% of rated value for more than "
            "60 seconds may cause thermal damage. Variable frequency drives (VFDs) "
            "can reduce energy consumption by 20–50% by matching motor speed to load."
        ),
    },
    {
        "id": 4,
        "source": "SKF Condition Monitoring Handbook",
        "page": 15,
        "text": (
            "Temperature monitoring is a primary indicator of machine health. "
            "Fluid temperature exceeding design limits by 10°C accelerates seal and "
            "bearing degradation. Thermocouple readings should be cross-referenced "
            "with fluid temperature sensors to identify heat exchanger inefficiencies. "
            "Trending temperature data is more valuable than single-point readings."
        ),
    },
    {
        "id": 5,
        "source": "SKAB Dataset Technical Report",
        "page": 1,
        "text": (
            "The SKAB (Skoltech Anomaly Benchmark) dataset was collected from an "
            "industrial water-circulation testbed. It includes measurements of motor "
            "current, supply voltage, fluid temperature (PT100 sensor), thermocouple "
            "temperature, two accelerometer RMS channels (vibration), system pressure, "
            "and volumetric flow rate. Anomalies were introduced via valve manipulations "
            "and mechanical interventions."
        ),
    },
    {
        "id": 6,
        "source": "DOE Compressed Air Systems Best Practices",
        "page": 5,
        "text": (
            "Voltage fluctuations in industrial systems can damage motor windings and "
            "reduce equipment lifespan. Supply voltage should remain within ±5% of "
            "rated voltage. Voltage sags below 95% of nominal during motor starts "
            "are normal but persistent low voltage indicates supply problems. "
            "Power quality monitoring is recommended for critical machinery."
        ),
    },
    {
        "id": 7,
        "source": "SKF Flow Measurement Guide",
        "page": 9,
        "text": (
            "Volumetric flow rate deviations in pumping systems indicate blockages, "
            "leaks, or pump degradation. A 10% reduction in flow rate from baseline "
            "warrants investigation. Causes include partial valve closure, clogged "
            "filters, air entrainment (cavitation), and wear ring erosion. Regular "
            "flow calibration ensures accurate monitoring."
        ),
    },
    {
        "id": 8,
        "source": "Industrial Maintenance Best Practices Guide",
        "page": 2,
        "text": (
            "Predictive maintenance programs typically achieve 25–30% cost savings "
            "over reactive maintenance. Key success factors include continuous sensor "
            "monitoring, anomaly detection algorithms, and structured maintenance "
            "planning. Machine learning anomaly detection (Isolation Forest, Autoencoder) "
            "can detect subtle patterns that threshold-based systems miss."
        ),
    },
    {
        "id": 9,
        "source": "DOE Motor Systems Market Assessment",
        "page": 20,
        "text": (
            "Recommended maintenance actions for detected anomalies: "
            "1. NORMAL – Continue standard monitoring schedule. "
            "2. VIBRATION ANOMALY – Inspect bearings, check alignment, re-balance rotor. "
            "3. THERMAL ANOMALY – Check cooling system, inspect lubricant, reduce load. "
            "4. CURRENT ANOMALY – Check electrical connections, inspect windings, verify load. "
            "5. PRESSURE ANOMALY – Inspect valves, check for blockages, verify pump condition. "
            "6. FLOW ANOMALY – Inspect filter, check for leaks, verify valve positions."
        ),
    },
]


# ─────────────────────────── Embedding Model ────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    """Load SentenceTransformer for embedding queries."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("SentenceTransformer loaded.")
        return model
    except Exception as e:
        logger.warning("SentenceTransformer unavailable: %s", e)
        return None


# ─────────────────────────── FAISS Index ────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_rag_index():
    """
    Load a FAISS index + chunks.
    Returns (index, chunks, embed_model) or (None, builtin_chunks, embed_model).
    """
    embed_model = load_embedding_model()
    chunks = BUILTIN_KNOWLEDGE  # fallback

    # Try FAISS directory
    if (FAISS_DIR / "index.faiss").exists() and (FAISS_DIR / "chunks.json").exists():
        try:
            import faiss
            index = faiss.read_index(str(FAISS_DIR / "index.faiss"))
            with open(FAISS_DIR / "chunks.json", "r", encoding="utf-8") as f:
                chunks = json.load(f)
            logger.info("FAISS index loaded from %s", FAISS_DIR)
            return index, chunks, embed_model
        except Exception as e:
            logger.warning("FAISS load failed: %s", e)

    # Try pickle
    if RAG_PKL.exists():
        try:
            import faiss
            with open(RAG_PKL, "rb") as f:
                data = pickle.load(f)
            embeddings = np.array(data["embeddings"], dtype=np.float32)
            faiss.normalize_L2(embeddings)
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
            chunks = data["chunks"]
            logger.info("FAISS index built from pickle.")
            return index, chunks, embed_model
        except Exception as e:
            logger.warning("RAG pickle load failed: %s", e)

    # Fallback: build in-memory index from BUILTIN_KNOWLEDGE
    if embed_model is not None:
        try:
            import faiss
            texts = [c["text"] for c in BUILTIN_KNOWLEDGE]
            embeddings = embed_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            embeddings = embeddings.astype(np.float32)
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
            logger.info("FAISS index built from built-in knowledge base.")
            return index, BUILTIN_KNOWLEDGE, embed_model
        except Exception as e:
            logger.warning("In-memory FAISS build failed: %s", e)

    return None, BUILTIN_KNOWLEDGE, embed_model


# ─────────────────────────── Retrieval ──────────────────────────────────────

def retrieve(
    query: str,
    top_k: int = 3,
    index=None,
    chunks: List[Dict] = None,
    embed_model=None,
) -> List[Dict]:
    """
    Retrieve top-k relevant chunks for a query.
    Falls back to keyword search over built-in knowledge if FAISS unavailable.
    """
    if chunks is None:
        chunks = BUILTIN_KNOWLEDGE

    if index is not None and embed_model is not None:
        try:
            import faiss
            q_emb = embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
            q_emb = q_emb.astype(np.float32)
            scores, idxs = index.search(q_emb, min(top_k, len(chunks)))
            results = []
            for score, idx in zip(scores[0], idxs[0]):
                if 0 <= idx < len(chunks):
                    chunk = dict(chunks[idx])
                    chunk["retrieval_score"] = float(score)
                    results.append(chunk)
            return results
        except Exception as e:
            logger.warning("FAISS search failed: %s", e)

    # Keyword fallback
    q_lower = query.lower()
    scored = []
    for chunk in chunks:
        text_lower = chunk["text"].lower()
        keywords = q_lower.split()
        score = sum(1 for kw in keywords if kw in text_lower)
        scored.append((score, chunk))
    scored.sort(key=lambda x: -x[0])
    results = []
    for score, chunk in scored[:top_k]:
        c = dict(chunk)
        c["retrieval_score"] = float(score)
        results.append(c)
    return results


def format_retrieved_context(chunks: List[Dict]) -> str:
    """Format retrieved chunks into a prompt-ready context string."""
    parts = []
    for i, c in enumerate(chunks, 1):
        source = c.get("source", "Unknown Source")
        page = c.get("page", "?")
        text = c.get("text", "")
        parts.append(f"[Reference {i}] {source}, p.{page}:\n{text}")
    return "\n\n".join(parts)
