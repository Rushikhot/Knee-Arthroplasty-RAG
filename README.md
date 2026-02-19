# Knee-Arthroplasty-RAG
A Retrieval-Augmented Generation (RAG) pipeline for orthopedic clinical decision support, built on MedGemma-27B and evaluated against vanilla LLM inference.
## Overview

This system retrieves relevant passages from a curated corpus of knee arthroplasty textbooks and guidelines, then generates grounded answers using a large language model. It includes hyperparameter tuning, cross-encoder reranking, and comprehensive evaluation (ROUGE, BERTScore, METEOR).

---

## Architecture

```
PDF Corpus
    │
    ▼
Text Extraction + Cleaning
    │
    ▼
Recursive Chunking  ──── chunk_size, chunk_overlap
    │
    ▼
Sentence Transformer Embeddings  ── all-mpnet-base-v2
    │
    ▼
FAISS Index  ◄──────── Cached to disk
    │
    │  Query
    ▼
Dense Retrieval (Inner Product Search)
    │
    ▼
Cross-Encoder Reranking  ── ms-marco-MiniLM-L-6-v2
    │
    ▼
Prompt Construction + MedGemma-27B Generation
    │
    ▼
Answer + Citations
    │
    ▼
Evaluation: ROUGE / BERTScore / METEOR
```

---

## Document Corpus

| Book | Author | Year |
|------|--------|------|
| Insall & Scott Surgery of the Knee | W. Norman Scott | 2018 |
| AAOS Surgical Management Guidelines | AAOS | 2022 |
| Campbell's Operative Orthopaedics | Azar & Beaty | 2021 |
| Master Techniques: Knee Arthroplasty | Pagnano & Hanssen | 2019 |
| Noyes' Knee Disorders | Frank R. Noyes | 2017 |
| Partial Knee Arthroplasty | Berend & Cushner | 2022 |
| Total Knee Arthroplasty | Richard D. Scott | 2015 |
| Unicompartmental Knee Arthroplasty | Tad L. Gerlinger | 2020 |

> **Note:** PDFs are not included in this repository due to copyright. Please obtain them independently and place them in `data/`.

---

## Repository Structure

```
knee-arthroplasty-rag/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   └── README.md              # Instructions to add PDFs manually
│
├── cache/
│   └── .gitkeep              
│
├── src/
│   ├── __init__.py
│   ├── indexer.py            
│   ├── retriever.py          
│   ├── generator.py          
│   ├── evaluator.py         
│   └── logger.py             
│
├── tuning/
│   └── tune_hyperparams.py   
│
├── configs/
│   ├── default_params.json  
│   └── best_params.json     
│
├── experiments/
│   └── results.xlsx          
│
├── notebooks/
│   └── demo.ipynb            
│
└── main.py                   
```

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/knee-arthroplasty-rag.git
cd knee-arthroplasty-rag
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Add Documents
Place your PDF files in the `data/` folder. File names must match exactly as listed in `src/indexer.py` metadata config.

### 4. Authenticate HuggingFace (for MedGemma)
```bash
huggingface-cli login
```
> MedGemma-27B requires access approval at [huggingface.co/google/medgemma-27b-it](https://huggingface.co/google/medgemma-27b-it)

---

## Usage

### Run Full Pipeline
```bash
python main.py --query "What are the indications for unicompartmental knee arthroplasty?"
```

### Run with Custom Config
```bash
python main.py --query "Your query here" --config configs/best_params.json
```

### Run Hyperparameter Tuning
```bash
python tuning/tune_hyperparams.py
```

### Run on Google Colab
Open `notebooks/demo.ipynb` directly in [Google Colab](https://colab.research.google.com). Mount your Google Drive and point `DOC_FOLDER` to your PDF location.

---

## Hyperparameters

| Parameter | Search Space | Default |
|-----------|-------------|---------|
| `chunk_size` | 400, 500, 800 | 500 |
| `chunk_overlap` | 50, 100 | 100 |
| `top_k` | 3, 5, 7 | 5 |
| `rerank_enabled` | True, False | True |
| `temperature` | — | 0.0 |

Tuning selects the best combination using **BERTScore F1** over a held-out tuning set of 5 clinical queries.

---

## Evaluation

The system runs an **ablation study** comparing:

| Configuration | Description |
|--------------|-------------|
| **RAG** | Retrieval + reranking + grounded generation |
| **Vanilla** | Direct LLM generation without retrieval |

**Metrics computed:**
- ROUGE-1, ROUGE-2, ROUGE-L (Precision, Recall, F1)
- BERTScore (Precision, Recall, F1)
- METEOR

All results are logged to `experiments/results.xlsx`.

---

## Models

| Component | Model |
|-----------|-------|
| Embedding | `sentence-transformers/all-mpnet-base-v2` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | `google/medgemma-27b-it` |

---

## Requirements

- Python 3.9+
- CUDA-compatible GPU (recommended: A100 40GB for MedGemma-27B)
- Google Drive (if using Colab)

See `requirements.txt` for full package list.



## Citation

If you use this work, please cite:

```bibtex
@misc{knee_arthroplasty_rag_2024,
  title     = {RAG-Based Clinical Decision Support for Knee Arthroplasty},
  author    = {Rushikesh Khot, Teena Sharma},
  year      = {2026},
  note      = {IIT Guwahati -- AIIMS Guwahati Collaboration},
  url       = {https://github.com/your-username/knee-arthroplasty-rag}
}
```

---

## License

This project is for **research and educational purposes only**. Clinical decisions should always be made by qualified medical professionals.


## Acknowledgements

Developed as part of a research collaboration between **IIT Guwahati** and **AIIMS Guwahati**.
