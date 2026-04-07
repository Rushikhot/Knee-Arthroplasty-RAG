"""
config.py
---------
Central configuration for the Knee-Arthroplasty RAG system.
All path constants and model names are defined here so that
the rest of the code never contains hard-coded paths.
"""

import os

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Models ───────────────────────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
CROSS_ENCODER_NAME   = "BAAI/bge-reranker-v2-m3"
LLM_MODEL_NAME       = "google/medgemma-27b-it"

# ── Paths (override with env vars for local use) ─────────────────────────────
_BASE = os.environ.get(
    "RAG_BASE_DIR",
    os.path.join(os.path.dirname(__file__), "..")   # repo root by default
)

DOC_FOLDER          = os.environ.get("RAG_DOC_FOLDER",   os.path.join(_BASE, "data"))
CACHE_FOLDER        = os.environ.get("RAG_CACHE_FOLDER", os.path.join(_BASE, "cache"))
RESULTS_FOLDER      = os.environ.get("RAG_RESULTS_DIR",  os.path.join(_BASE, "results"))
EXPERIMENT_LOG_PATH = os.path.join(RESULTS_FOLDER, "experiments.xlsx")
TUNED_PARAMS_PATH   = os.path.join(RESULTS_FOLDER, "best_parameters.json")

# ── Google Drive folder ID (used only on Colab) ───────────────────────────────
GDRIVE_DATA_FOLDER_ID = "1C8_jMFaiIz6JS-YzeAuvqXu-600sQSGE"

# ── Document metadata ─────────────────────────────────────────────────────────
DOC_METADATA = {
    "2018 Insall and Scott Surgery of the Knee 6e.pdf": {
        "book_title": "Insall & Scott Surgery of the Knee",
        "author": "W. Norman Scott",
        "year": 2018,
    },
    "aaos guidlines .pdf": {
        "book_title": "Surgical Management of Osteoarthritis of the Knee",
        "author": "AAOS",
        "year": 2022,
    },
    "Campbell's Operative Orthopaedics 14th Edition.pdf": {
        "book_title": "Campbell's Operative Orthopaedics",
        "author": "Frederick M. Azar, James H. Beaty",
        "year": 2021,
    },
    "Master_Techniques_in_orthopaedic_surgery_knee_Arthroplasty.pdf": {
        "book_title": "Master Techniques in Orthopaedic Surgery: Knee Arthroplasty",
        "author": "Mark Pagnano, Arlen Hanssen",
        "year": 2019,
    },
    "Noyes_Knee_Disorders_Surgery_Rehabilitation.pdf": {
        "book_title": "Noyes' Knee Disorders: Surgery, Rehabilitation, Clinical Outcomes",
        "author": "Frank R. Noyes",
        "year": 2017,
    },
    "Partial knee arthroplasty;techniques for optimal outcomes.pdf": {
        "book_title": "Partial Knee Arthroplasty: Techniques for Optimal Outcomes",
        "author": "Keith R. Berend, Fred D. Cushner",
        "year": 2022,
    },
    "Total Knee Arthroplasty-Richard D Scott 2nd.pdf": {
        "book_title": "Total Knee Arthroplasty",
        "author": "Richard D. Scott",
        "year": 2015,
    },
    "Unicompartmental_Knee_Arthroplasty.pdf": {
        "book_title": "Unicompartmental Knee Arthroplasty",
        "author": "Tad L. Gerlinger",
        "year": 2020,
    },
}

# ── Default RAG hyperparameters ───────────────────────────────────────────────
DEFAULT_HYPERPARAMS = {
    "chunk_size":     800,
    "chunk_overlap":  150,
    "top_k":          7,
    "rerank_enabled": True,
    "temperature":    0.0,
    "embedding_model": EMBEDDING_MODEL_NAME,
    "llm_model":      LLM_MODEL_NAME,
}
