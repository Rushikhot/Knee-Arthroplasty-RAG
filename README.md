# Knee Arthroplasty RAG

A Retrieval-Augmented Generation (RAG) system for clinical question answering in knee arthroplasty, developed as part of a collaboration between **IIT Guwahati** and **AIIMS**. The system retrieves evidence from a curated corpus of orthopaedic textbooks and generates grounded answers using **MedGemma-27B**.

---

## Overview

The pipeline ingests PDF textbooks, chunks and embeds them using **BAAI/bge-m3**, indexes them in **FAISS**, optionally reranks retrieved chunks with a **cross-encoder (bge-reranker-v2-m3)**, and feeds the context to MedGemma to generate answers. An ablation study compares RAG against vanilla LLM generation across ROUGE, BERTScore, and METEOR metrics.

```
PDF Documents
     │
     ▼
 Chunking (RecursiveCharacterTextSplitter)
     │
     ▼
 Embeddings (BAAI/bge-m3)  ──► FAISS Index
                                    │
                          Query Embedding
                                    │
                          FAISS Search (top-k × 2)
                                    │
                          Cross-Encoder Reranking
                                    │
                          Context Assembly
                                    │
                          MedGemma-27B Generation
                                    │
                          Answer + Citations
```

---

## Knowledge Base

| Textbook | Author | Year |
|---|---|---|
| Insall & Scott Surgery of the Knee (6e) | W. Norman Scott | 2018 |
| Surgical Management of Osteoarthritis of the Knee | AAOS | 2022 |
| Campbell's Operative Orthopaedics (14e) | Azar & Beaty | 2021 |
| Master Techniques: Knee Arthroplasty | Pagnano & Hanssen | 2019 |
| Noyes' Knee Disorders | Frank R. Noyes | 2017 |
| Partial Knee Arthroplasty | Berend & Cushner | 2022 |
| Total Knee Arthroplasty | Richard D. Scott | 2015 |
| Unicompartmental Knee Arthroplasty | Tad L. Gerlinger | 2020 |

> PDFs are not included in this repository. Place them in `data/` before running.

---

## Models

| Role | Model |
|---|---|
| LLM | `google/medgemma-27b-it` |
| Embeddings | `BAAI/bge-m3` |
| Reranker | `BAAI/bge-reranker-v2-m3` |

> MedGemma requires a Hugging Face account with access granted. Run `huggingface-cli login` before use.

---

## Repo Structure

```
knee-arthroplasty-rag/
│
├── README.md
├── .gitignore
├── requirements.txt
│
├── src/
│   ├── loader.py        # PDF ingestion, text cleaning, chunking, PageSource
│   ├── indexer.py       # CacheManager + OptimizedRAGIndexer (FAISS + cross-encoder)
│   ├── generator.py     # rag_generate(), vanilla_query(), prompt templates
│   ├── tuner.py         # OptimizedParameterTuner, hyperparameter search
│   └── evaluator.py     # ComprehensiveEvaluator, ExperimentLogger (Excel)
│
├── scripts/
│   ├── build_index.py   # Load PDFs → build & cache FAISS index
│   ├── run_tuning.py    # Hyperparameter search → saves best_params.json
│   └── run_evaluation.py # Ablation study (RAG vs Vanilla) → Excel results
│
├── configs/
│   └── config.yaml      # All paths, model names, default hyperparameters
│
├── notebooks/
│   └── rag_llm_final_iitg_aiims.ipynb  # Original Colab notebook
│
├── data/                # Place PDF textbooks here (gitignored)
├── cache/               # Auto-created: chunked embeddings (gitignored)
└── outputs/             # Auto-created: Excel logs, best_params.json (gitignored)
```

---

## Setup

```bash
git clone https://github.com/your-org/knee-arthroplasty-rag.git
cd knee-arthroplasty-rag
pip install -r requirements.txt
```

**Requirements summary:** `torch`, `transformers`, `sentence-transformers`, `faiss-cpu`, `langchain-text-splitters`, `pdfplumber`, `PyPDF2`, `rouge-score`, `bert-score`, `evaluate`, `nltk`, `pandas`, `openpyxl`, `tqdm`

---

## Usage

### 1. Build the index
```bash
python scripts/build_index.py --data_dir data/ --cache_dir cache/
```

### 2. Tune hyperparameters
```bash
python scripts/run_tuning.py --output outputs/best_params.json
```
Searches over chunk size, chunk overlap, top-k retrieval, and reranking on/off. Optimises for **ROUGE-L F1**.

### 3. Run evaluation
```bash
python scripts/run_evaluation.py --params outputs/best_params.json --output outputs/results.xlsx
```
Runs an ablation study comparing **RAG vs Vanilla** across all test queries and logs ROUGE-1/2/L, BERTScore, and METEOR to an Excel file.

---

## Hyperparameter Search Space

| Parameter | Values |
|---|---|
| `chunk_size` | 600, 800, 1000, 1200 |
| `chunk_overlap` | 100, 150, 200 |
| `top_k` | 5, 7, 10, 12 |
| `rerank_enabled` | True, False |

Default best found: `chunk_size=1000`, `chunk_overlap=100`, `top_k=12`, `rerank_enabled=True`

---

## Evaluation Metrics

- **ROUGE-1, ROUGE-2, ROUGE-L** — n-gram overlap with ground truth
- **BERTScore** — contextual semantic similarity
- **METEOR** — unigram overlap with synonym matching
- **Retrieval metrics** — FAISS cosine similarity and cross-encoder scores per query

Results are logged row-by-row to an Excel sheet (`outputs/results.xlsx`) for easy analysis.

---

## Notes

- The system was built and tested on **Google Colab** with an A100 GPU. MedGemma-27B requires ~40 GB VRAM; use `bfloat16` and `device_map="auto"`.
- Chunk embeddings and FAISS indices are cached to disk so subsequent runs skip re-embedding.
- The `</think>` tag stripping handles any residual chain-of-thought output from MedGemma.
