# Evaluation of ArthroRAG: A Retrieval-Augmented Generation–Enabled Large Language Model for Clinical Decision Making in Arthroplasty

> **RAG-augmented clinical Q&A for Knee Arthroplasty** 
> Embedding: `BAAI/bge-m3` · Reranker: `BAAI/bge-reranker-v2-m3` · LLM: `google/medgemma-27b-it`

---

## Overview

This repository implements a **Retrieval-Augmented Generation (RAG)** system designed to answer complex orthopaedic clinical questions about knee arthroplasty. The pipeline retrieves relevant passages from eight authoritative textbooks using dense semantic search (FAISS + BGE-M3), optionally reranks them with a cross-encoder, and generates grounded answers with MedGemma-27B.

An **ablation study** compares RAG-augmented generation against vanilla (no-retrieval) MedGemma, evaluated using ROUGE-1/2/L, BERTScore, and METEOR on 15 expert-annotated queries.

---

## Architecture

```
PDFs ──► pdfplumber ──► clean_text ──► RecursiveCharacterTextSplitter
                                              │
                                         BGE-M3 encoder
                                              │
                                         FAISS IndexFlatIP  ◄── Query
                                              │
                                    (optional) BGE-reranker-v2-m3
                                              │
                                         MedGemma-27B-it
                                              │
                                           Answer + Citations
```

---

## Repository Structure

```
rag-knee-arthroplasty/
│
├── main.py                       
├── requirements.txt
│
├── src/
│   ├── __init__.py
│   ├── config.py                 
│   ├── data_io.py                
│   ├── cache_manager.py          
│   ├── indexer.py                 
│   ├── rag_pipeline.py           
│   ├── evaluation.py             
│   ├── tuner.py                   
│   └── test_queries.py           
│
├── notebooks/
│   └── RAG_LLM_FINAL_IITG_AIIMS.ipynb      
│
├── data/                          # PDF textbooks (NOT committed — see below)
├── cache/                        
└── results/                      
```

---

## Quickstart — Google Colab (recommended)

> **Requires**: A100 / H100 GPU · MedGemma licence accepted on HF

1. Open `notebooks/RAG_LLM_FINAL_IITG_AIIMS.ipynb` in Colab.
2. Set `HF_TOKEN` as a Colab Secret .
3. Place your PDFs in Google Drive at the path configured in `src/config.py`  
   *or* update `GDRIVE_DATA_FOLDER_ID` to your own Drive folder.
4. Run all cells — the notebook handles cloning, installing, downloading, and evaluation automatically.

---

## Quickstart — Local

```bash
# 1. Clone
git clone https://github.com/Rushikhot/rag-knee-arthroplasty.git
cd rag-knee-arthroplasty

# 2. Install (Python 3.10+, CUDA GPU strongly recommended)
pip install -r requirements.txt

# 3. Place your PDFs
cp /path/to/your/pdfs/*.pdf data/

# 4. Set your HF token
export HF_TOKEN=hf_...

# 5. Run
python main.py
```

Results are saved to `results/experiments.xlsx`.

---

## Data — Textbooks Used

| # | Title | Author | Year |
|---|-------|--------|------|
| 1 | Insall & Scott Surgery of the Knee (6e) | W. Norman Scott | 2018 |
| 2 | Surgical Management of Osteoarthritis of the Knee (AAOS Guidelines) | AAOS | 2022 |
| 3 | Campbell's Operative Orthopaedics (14e) | Azar & Beaty | 2021 |
| 4 | Master Techniques in Orthopaedic Surgery: Knee Arthroplasty | Pagnano & Hanssen | 2019 |
| 5 | Noyes' Knee Disorders: Surgery, Rehabilitation, Clinical Outcomes | Frank R. Noyes | 2017 |
| 6 | Partial Knee Arthroplasty: Techniques for Optimal Outcomes | Berend & Cushner | 2022 |
| 7 | Total Knee Arthroplasty (2e) | Richard D. Scott | 2015 |
| 8 | Unicompartmental Knee Arthroplasty | Tad L. Gerlinger | 2020 |

> **Note**: PDFs are not committed to this repository due to copyright.

---

## Configuration

All paths and model names are in `src/config.py` and can be overridden with environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_BASE_DIR` | repo root | Base directory |
| `RAG_DOC_FOLDER` | `data/` | PDF input folder |
| `RAG_CACHE_FOLDER` | `cache/` | Embedding cache |
| `RAG_RESULTS_DIR` | `results/` | Output Excel file |
| `HF_TOKEN` | — | Hugging Face token |

---

## Hyperparameter Tuning

To run grid-search before the final evaluation, use `OptimizedParameterTuner` from `src/tuner.py`. Best parameters are saved as JSON and automatically loaded by `main.py` on subsequent runs.

```python
from src.tuner import OptimizedParameterTuner
tuner = OptimizedParameterTuner(indexer, model, tokenizer, cache_manager)
best  = tuner.tune(tuning_queries, tuning_ground_truths,
                   save_path="results/best_parameters.json")
```

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| ROUGE-1/2/L | N-gram overlap (precision, recall, F1) |
| BERTScore | Contextual embedding similarity (F1) |
| METEOR | Precision + recall with stemming & synonyms |

Results for both **RAG** and **Vanilla** configurations are logged per-query to `results/experiments.xlsx`.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{knee_arthroplasty_rag_2026,
  title   = {RAG-Augmented Clinical Q\&A for Knee Arthroplasty},
  author  = {Rushikesh Khot, Dr.Teena sharma , Dr.Plaban Nath Chowdhury  Dr.Rajesh malhotra },
  year    = {2026},
  institution = {IIT Guwahati, AIIMS},
  url     = {https://github.com/Rushikhot/rag-knee-arthroplasty}
}
```

---

## License

MIT — see [LICENSE](LICENSE).  
Note: The orthopaedic textbooks used as data sources are subject to their respective publishers' copyright and are not redistributed in this repository.
