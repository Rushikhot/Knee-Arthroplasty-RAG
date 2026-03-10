# -*- coding: utf-8 -*-
"""
main.py
Entry point for the Knee Arthroplasty RAG pipeline.

Usage:
    python main.py --mode eval                  # Run ablation study (RAG vs Vanilla)
    python main.py --mode tune                  # Run hyperparameter tuning
    python main.py --mode query -q "your question here"   # Single query
"""

import argparse
import json
import os
import warnings
from datetime import datetime

import torch

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────

DOC_FOLDER          = "data/"
CACHE_FOLDER        = "cache/"
EXPERIMENT_LOG_PATH = "experiments/results.xlsx"
BEST_PARAMS_PATH    = "configs/best_params.json"
DEFAULT_PARAMS_PATH = "configs/default_params.json"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
LLM_MODEL_NAME       = "google/medgemma-27b-it"

# ── Document metadata ────────────────────────────────────────────────────────

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

# ── Evaluation queries & ground truths ───────────────────────────────────────

TEST_QUERIES = [
    "After insertion of the trial components in a total knee replacement, the surgeon finds that he is unable to fully extend the knee and that the tibial tray lifts-off when the knee is flexed past 90 degrees. What intervention should be taken to achieve a knee that is balanced in flexion and extension?",
    "A 40-year-old man complains of increasing groin pain. Radiographs show femoral head avascular necrosis with subchondral lucency but without femoral head collapse. Which of the following medical treatments have been shown to decrease the risk of subsequent femoral head collapse?",
    "Enumerate the important operative steps in total knee arthroplasty for a 65-year-old patient with advanced osteoarthritis and fixed flexion deformity",
    "Enumerate the most important operative steps in total knee arthroplasty for a 65-year-old patient with advanced osteoarthritis and Valgus deformity.",
    "What percentage of patients with complete peroneal nerve palsy after total hip arthroplasty will never recover full strength?",
    "A 68-year-old patient presents 8 months after total knee arthroplasty with complaints of giving way while descending stairs and recurrent swelling. What is your differential diagnosis?",
    "What investigations are required to assess patellar maltracking after TKA?",
    "A 68-year-old male presents 8 months after undergoing Total knee arthroplasty. He complains of a painful 'catching' sensation in the knee while rising from a chair. He describes a distinct 'clunk' when extending his knee from a flexed position. What is the diagnosis. How to treat this condition?",
]

GROUND_TRUTHS = [
    """In a total knee replacement, inability to fully extend the knee indicates a tight extension gap, and tibial tray lift-off beyond 90° of flexion indicates a tight flexion gap; therefore, both flexion and extension gaps are tight. When both gaps are equally tight, the appropriate intervention is additional proximal tibial resection because the tibial cut affects both the flexion and extension gaps equally. By recutting the proximal tibia slightly, both gaps are increased uniformly, restoring full extension, eliminating tibial tray lift-off in flexion, and achieving a balanced knee throughout the range of motion.""",
    """A 40-year-old man with groin pain and radiographic evidence of femoral head avascular necrosis showing subchondral lucency without collapse represents early (pre-collapse) disease, in which the necrotic trabecular bone becomes structurally weak and prone to subchondral fracture and eventual collapse. Bisphosphonate therapy has been shown to decrease the risk of femoral head collapse in this stage by inhibiting osteoclast-mediated bone resorption, thereby preserving trabecular architecture and maintaining subchondral bone strength.""",
    """The five most important operative steps in total knee arthroplasty for a 65-year-old patient with advanced osteoarthritis and fixed flexion deformity depend on the severity of the deformity. For mild deformity, the key steps are excision of medial and posterior osteophytes and posteromedial soft tissue release. For moderate deformity, additional steps include posterior capsular release, decreasing the tibial slope, releasing the gastrocnemius origin, and pie-crusting of the MCL. For severe deformity, an extra distal femoral cut, medial epicondylar osteotomy, and constrained implants may be required.""",
    """In total knee arthroplasty for a valgus knee deformity, a lateral parapatellar approach can be used. The tibial resection is performed in the standard manner. For the femur, a 3° valgus distal femoral cut is taken. Soft tissue balancing follows the Ranawat inside-out release technique: lateral osteophytes, PCL, posterolateral capsule, iliotibial band, and popliteus. The epicondylar axis is used as the primary reference for femoral rotation. Lateral epicondylar osteotomy is an alternative when additional correction is required.""",
    """Approximately 40–50% of patients with complete peroneal nerve palsy after total hip arthroplasty will never recover full strength.""",
    """The differential diagnosis includes flexion instability, component malposition or malrotation, polyethylene wear or early mechanical loosening, extensor mechanism insufficiency, patellofemoral instability or maltracking, periprosthetic joint infection (chronic low-grade), and aseptic loosening of components.""",
    """Skyline (Merchant) view, CT scan (to assess femoral and tibial component rotation), and long-leg alignment films.""",
    """Diagnosis: Patellar clunk syndrome. Treatment: Initial management may include observation if symptoms are mild. Definitive treatment is arthroscopic or open debridement of the fibrous nodule at the superior pole of the patella/posterior quadriceps tendon that catches in the intercondylar box of the femoral component.""",
]

TUNING_QUERIES = [
    "69 year old male patient has developed osteoarthritis of knee. He has a fixed flexion deformity of 50 degrees. What important surgical steps should I follow while doing total knee replacement?",
    "What are the indications for unicompartmental knee arthroplasty?",
    "How do you manage bone loss in revision total knee arthroplasty?",
    "What are the contraindications for total knee replacement?",
    "Describe the surgical approach for minimally invasive total knee arthroplasty.",
]

TUNING_GROUND_TRUTHS = [
    """In a patient with severe fixed flexion deformity of 50 degrees: perform posterior capsular release, consider PCL release, remove posterior osteophytes, use thicker polyethylene or augment tibial component, check gap balancing, and consider constrained implant if instability persists.""",
    """Indications: isolated medial or lateral compartment OA, intact ACL, ROM > 90°, angular deformity < 15° correctable, age > 60, weight ideally < 82 kg. Inflammatory arthritis is a contraindication.""",
    """Classify defect (AORI), use cement for small defects, metal augments for moderate, structural allografts for large defects, trabecular metal cones/sleeves for metaphyseal defects, stems for stability, constrained implants if ligamentous instability.""",
    """Contraindications: active/latent knee sepsis, active infection elsewhere, extensor mechanism dysfunction, neuropathic arthropathy, severe vascular insufficiency, morbid obesity (relative), non-functional quadriceps, poor bone stock.""",
    """Minimally invasive TKA: smaller incision (10–14 cm), midvastus or subvastus approach, specialised retractors, no patellar eversion in some techniques, careful soft tissue handling, good patient selection (non-obese, good ROM).""",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_params() -> dict:
    """Load tuned params if available, otherwise fall back to defaults."""
    for path in (BEST_PARAMS_PATH, DEFAULT_PARAMS_PATH):
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            params = data.get("best_parameters", data)
            print(f"Loaded parameters from: {path}")
            return params

    # Hard-coded fallback
    return {
        "chunk_size": 500,
        "chunk_overlap": 100,
        "top_k": 5,
        "rerank_enabled": True,
        "temperature": 0.0,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "llm_model": LLM_MODEL_NAME,
    }


def setup_seed(seed: int = 42):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed: {seed}")


def load_model(model_name: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from huggingface_hub import login

    login()

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager"
    )
    model.eval()
    torch.set_grad_enabled(False)

    print("Model ready\n")
    return model, tokenizer


# ── Modes ─────────────────────────────────────────────────────────────────────

def run_eval(args):
    """Ablation study: RAG vs Vanilla across all TEST_QUERIES."""
    from src.utils import load_documents
    from src.indexer import CacheManager, OptimizedRAGIndexer
    from src.logger import ExperimentLogger
    from src.evaluator import ComprehensiveEvaluator

    setup_seed()
    params = load_params()

    print("\n── Loading documents ──")
    documents, doc_sources = load_documents(DOC_FOLDER, DOC_METADATA)
    if not documents:
        raise RuntimeError(f"No documents found in {DOC_FOLDER}. Add PDFs and retry.")

    print("\n── Building index ──")
    cache_manager  = CacheManager(CACHE_FOLDER)
    indexer        = OptimizedRAGIndexer(EMBEDDING_MODEL_NAME, cache_manager)
    indexer.build_index(
        documents=documents,
        doc_sources=doc_sources,
        chunk_size=params["chunk_size"],
        chunk_overlap=params["chunk_overlap"]
    )
    indexer.load_embedding_model()

    print("\n── Loading LLM ──")
    model, tokenizer = load_model(LLM_MODEL_NAME)

    print("\n── Running ablation study ──")
    os.makedirs(os.path.dirname(EXPERIMENT_LOG_PATH), exist_ok=True)
    logger    = ExperimentLogger(EXPERIMENT_LOG_PATH)
    evaluator = ComprehensiveEvaluator(logger)

    evaluator.run_ablation_study(
        indexer=indexer,
        model=model,
        tokenizer=tokenizer,
        queries=TEST_QUERIES,
        ground_truths=GROUND_TRUTHS,
        hyperparameters=params
    )

    print(f"\nResults saved → {EXPERIMENT_LOG_PATH}")


def run_tune(args):
    """Hyperparameter search over chunk size, overlap, top-k, and reranking."""
    from src.utils import load_documents
    from src.indexer import CacheManager, OptimizedRAGIndexer
    from src.evaluator import OptimizedParameterTuner

    setup_seed()

    print("\n── Loading documents ──")
    documents, doc_sources = load_documents(DOC_FOLDER, DOC_METADATA)
    if not documents:
        raise RuntimeError(f"No documents found in {DOC_FOLDER}.")

    print("\n── Loading LLM ──")
    model, tokenizer = load_model(LLM_MODEL_NAME)

    cache_manager = CacheManager(CACHE_FOLDER)
    indexer       = OptimizedRAGIndexer(EMBEDDING_MODEL_NAME, cache_manager)

    tuner      = OptimizedParameterTuner(indexer, seed=42)
    best_params = tuner.tune_parameters(
        documents=documents,
        doc_sources=doc_sources,
        test_queries=TUNING_QUERIES,
        ground_truths=TUNING_GROUND_TRUTHS,
        model=model,
        tokenizer=tokenizer,
        num_combinations=args.num_combinations,
        verbose=True
    )

    if best_params:
        best_params["temperature"]     = 0.0
        best_params["embedding_model"] = EMBEDDING_MODEL_NAME
        best_params["llm_model"]       = LLM_MODEL_NAME

        os.makedirs(os.path.dirname(BEST_PARAMS_PATH), exist_ok=True)
        with open(BEST_PARAMS_PATH, "w") as f:
            json.dump({
                "best_parameters": best_params,
                "quality_score": float(tuner.best_score),
                "metric": "BERTScore_F1",
                "timestamp": datetime.now().isoformat(),
                "num_combinations_tested": len(tuner.results),
                "all_results": tuner.results
            }, f, indent=2)

        print(f"\nBest params saved → {BEST_PARAMS_PATH}")


def run_query(args):
    """Answer a single question in RAG mode."""
    from src.utils import load_documents
    from src.indexer import CacheManager, OptimizedRAGIndexer
    from src.generator import rag_generate

    if not args.query:
        raise ValueError("Provide a query with -q / --query")

    setup_seed()
    params = load_params()

    print("\n── Loading documents ──")
    documents, doc_sources = load_documents(DOC_FOLDER, DOC_METADATA)
    if not documents:
        raise RuntimeError(f"No documents found in {DOC_FOLDER}.")

    print("\n── Building index ──")
    cache_manager = CacheManager(CACHE_FOLDER)
    indexer       = OptimizedRAGIndexer(EMBEDDING_MODEL_NAME, cache_manager)
    indexer.build_index(
        documents=documents,
        doc_sources=doc_sources,
        chunk_size=params["chunk_size"],
        chunk_overlap=params["chunk_overlap"]
    )
    indexer.load_embedding_model()

    print("\n── Loading LLM ──")
    model, tokenizer = load_model(LLM_MODEL_NAME)

    print(f"\nQuery: {args.query}\n")
    result = rag_generate(
        indexer=indexer,
        model=model,
        tokenizer=tokenizer,
        query=args.query,
        k=params["top_k"],
        use_reranking=params["rerank_enabled"],
        verbose=True
    )

    print("=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(result.get("answer", "No answer generated."))
    print("\nSOURCES")
    print("=" * 70)
    for citation in result.get("citations", []):
        print(f"  • {citation}")
    print(f"\nExecution time: {result.get('execution_time', 0):.2f}s")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Knee Arthroplasty RAG Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--mode", choices=["eval", "tune", "query"], required=True,
        help="Pipeline mode: eval | tune | query"
    )
    parser.add_argument(
        "-q", "--query", type=str, default=None,
        help="Question to answer (required for --mode query)"
    )
    parser.add_argument(
        "--num-combinations", type=int, default=15,
        help="Number of hyperparameter combinations to test (default: 15)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "eval":
        run_eval(args)
    elif args.mode == "tune":
        run_tune(args)
    elif args.mode == "query":
        run_query(args)


if __name__ == "__main__":
    main()
