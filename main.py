"""
main.py
-------
Entry-point for the Knee-Arthroplasty RAG pipeline.

Workflow
--------
1. Mount Google Drive (Colab) or use local DATA_FOLDER
2. Download PDFs from Drive if needed
3. Load & chunk documents
4. Build FAISS index (or restore from cache)
5. Load MedGemma LLM
6. (Optional) Tune hyperparameters
7. Run RAG-vs-Vanilla ablation study
8. Save results to Excel

Environment variables (optional, override defaults in config.py)
----------------------------------------------------------------
  RAG_BASE_DIR     – repository root (default: repo root)
  RAG_DOC_FOLDER   – path to PDF folder
  RAG_CACHE_FOLDER – path to embedding cache
  RAG_RESULTS_DIR  – path to results folder
  HF_TOKEN         – Hugging Face access token (required for MedGemma)
"""

import os
import sys
import random
import json

import numpy as np
import torch

# ── Add repo root to path if running as a script ─────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from src.config import (
    RANDOM_SEED, DOC_FOLDER, CACHE_FOLDER, RESULTS_FOLDER,
    EXPERIMENT_LOG_PATH, TUNED_PARAMS_PATH,
    EMBEDDING_MODEL_NAME, LLM_MODEL_NAME,
    DOC_METADATA, DEFAULT_HYPERPARAMS,
    GDRIVE_DATA_FOLDER_ID,
)
from src.data_io import load_documents
from src.cache_manager import CacheManager
from src.indexer import OptimizedRAGIndexer
from src.evaluation import ExperimentLogger, ComprehensiveEvaluator
from src.test_queries import TEST_QUERIES, GROUND_TRUTHS


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
    print(f"  Seed set to {seed}")


# ── Google Drive helpers (Colab only) ─────────────────────────────────────────

def _is_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def mount_drive() -> None:
    """Mount Google Drive (only inside Colab)."""
    if not _is_colab():
        return
    from google.colab import drive
    if not os.path.ismount("/content/drive"):
        drive.mount("/content/drive")
        print("  Drive mounted.")
    else:
        print("  Drive already mounted.")


def download_pdfs_from_drive(dest_folder: str, folder_id: str) -> None:
    """Download PDFs from a shared Drive folder (Colab only)."""
    if not _is_colab():
        return

    existing = [f for f in os.listdir(dest_folder) if f.lower().endswith('.pdf')]
    if len(existing) >= 8:
        print(f"  PDFs already present ({len(existing)} files).")
        return

    print("  Downloading PDFs from Google Drive …")
    try:
        from google.colab import auth
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaIoBaseDownload
        import io
        from tqdm import tqdm

        auth.authenticate_user()
        svc = build("drive", "v3")

        resp = svc.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            spaces="drive",
            fields="files(id, name)",
            pageSize=100,
        ).execute()
        files = resp.get("files", [])

        for file in tqdm(files, desc="Downloading"):
            dest = os.path.join(dest_folder, file["name"])
            if os.path.exists(dest):
                continue
            request    = svc.files().get_media(fileId=file["id"])
            buf        = io.BytesIO()
            downloader = MediaIoBaseDownload(buf, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            with open(dest, "wb") as f:
                f.write(buf.getvalue())

        print("  Download complete.")
    except Exception as e:
        print(f"  [ERROR] Download failed: {e}")
        print(f"  Please manually copy PDFs to: {dest_folder}")


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(model_name: str = LLM_MODEL_NAME):
    """Load MedGemma tokenizer and model."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from huggingface_hub import login

    # Authenticate (token from env or interactive prompt)
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
    else:
        login()   # interactive prompt

    print(f"  Loading tokenizer: {model_name} …")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print("  Tokenizer ready.")

    print(f"  Loading model: {model_name} …")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    torch.set_grad_enabled(False)
    print("  Model ready.\n")
    return tokenizer, model


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 70)
    print("  Knee-Arthroplasty RAG Pipeline")
    print("=" * 70 + "\n")

    set_seed(RANDOM_SEED)

    # ── Paths ──────────────────────────────────────────────────────────────
    os.makedirs(DOC_FOLDER,    exist_ok=True)
    os.makedirs(CACHE_FOLDER,  exist_ok=True)
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    # ── Data acquisition ───────────────────────────────────────────────────
    mount_drive()
    download_pdfs_from_drive(DOC_FOLDER, GDRIVE_DATA_FOLDER_ID)

    # ── Document loading ───────────────────────────────────────────────────
    documents, doc_sources = load_documents(DOC_FOLDER, DOC_METADATA)
    if not documents:
        print("  WARNING: No documents loaded — will attempt cache-only index restore.")

    # ── Components ─────────────────────────────────────────────────────────
    cache_manager     = CacheManager(CACHE_FOLDER)
    experiment_logger = ExperimentLogger(EXPERIMENT_LOG_PATH, clear_existing=True)
    indexer           = OptimizedRAGIndexer(EMBEDDING_MODEL_NAME, cache_manager)

    # ── Index ──────────────────────────────────────────────────────────────
    chunk_size    = DEFAULT_HYPERPARAMS["chunk_size"]
    chunk_overlap = DEFAULT_HYPERPARAMS["chunk_overlap"]

    if documents:
        indexer.build_index(documents, doc_sources, chunk_size, chunk_overlap)
    else:
        ok = indexer.load_index_from_cache(chunk_size, chunk_overlap)
        if not ok:
            raise RuntimeError(
                "No documents and no cached embeddings found. "
                "Please provide PDFs in the data/ folder."
            )

    indexer.load_embedding_model()   # needed for retrieval

    # ── LLM ────────────────────────────────────────────────────────────────
    tokenizer, model = load_model(LLM_MODEL_NAME)

    # ── (Optional) load best params from previous tuning run ──────────────
    best_params = DEFAULT_HYPERPARAMS.copy()
    if os.path.exists(TUNED_PARAMS_PATH):
        with open(TUNED_PARAMS_PATH) as f:
            best_params.update(json.load(f))
        print(f"  Loaded tuned params from {TUNED_PARAMS_PATH}")

    # ── Evaluation ─────────────────────────────────────────────────────────
    print("\nRUNNING EVALUATION")
    evaluator = ComprehensiveEvaluator(experiment_logger)
    evaluator.run_ablation_study(
        indexer=indexer,
        model=model,
        tokenizer=tokenizer,
        queries=TEST_QUERIES,
        ground_truths=GROUND_TRUTHS,
        hyperparameters=best_params,
    )

    print("\nEVALUATION COMPLETE")
    print(f"  Results: {EXPERIMENT_LOG_PATH}")
    print(f"  Cache  : {CACHE_FOLDER}\n")


if __name__ == "__main__":
    main()
