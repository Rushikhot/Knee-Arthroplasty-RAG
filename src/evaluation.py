"""
evaluation.py
-------------
ExperimentLogger  – writes results row-by-row to an Excel file.
ComprehensiveEvaluator – computes ROUGE / BERTScore / METEOR and
                         runs the RAG-vs-Vanilla ablation study.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from bert_score import score as bert_score
from rouge_score import rouge_scorer
import evaluate as hf_evaluate

from .rag_pipeline import rag_generate, vanilla_query


# ── Experiment logger ─────────────────────────────────────────────────────────

_COLUMNS = [
    "Timestamp", "Experiment_ID", "Query", "Configuration",
    "Ground_Truth", "Has_Ground_Truth", "Answer",
    "Answer_Length_Words", "Answer_Length_Chars",
    "Citations", "Citation_Count",
    "FAISS_Top_Similarity", "FAISS_Avg_Similarity", "FAISS_Min_Similarity",
    "CrossEncoder_Top_Score", "CrossEncoder_Avg_Score",
    "CrossEncoder_Min_Score", "CrossEncoder_Max_Score",
    "Total_Retrieved_Chunks", "Reranking_Applied",
    "ROUGE1_Precision", "ROUGE1_Recall", "ROUGE1_F1",
    "ROUGE2_Precision", "ROUGE2_Recall", "ROUGE2_F1",
    "ROUGEL_Precision", "ROUGEL_Recall", "ROUGEL_F1",
    "BERTScore_Precision", "BERTScore_Recall", "BERTScore_F1",
    "METEOR_Score",
    "chunk_size", "chunk_overlap", "top_k_retrieval",
    "rerank_enabled", "temperature",
    "embedding_model", "llm_model",
    "execution_time_seconds", "status", "error_message",
]


class ExperimentLogger:
    """Append experiment results row-by-row to an Excel workbook."""

    SHEET = "Experiment_Results"

    def __init__(self, excel_path: str, clear_existing: bool = False):
        self.excel_path = excel_path
        os.makedirs(os.path.dirname(excel_path), exist_ok=True)

        if clear_existing and os.path.exists(excel_path):
            try:
                os.remove(excel_path)
                print(f"  [logger] Cleared existing log: {os.path.basename(excel_path)}")
            except Exception as e:
                print(f"  [logger] Warning – could not clear log: {e}")

        if not os.path.exists(excel_path):
            pd.DataFrame(columns=_COLUMNS).to_excel(
                excel_path, sheet_name=self.SHEET, index=False
            )

    def log_experiment(
        self,
        query: str,
        answer: str,
        configuration: str,
        ground_truth:     Optional[str]       = None,
        rouge_scores:     Optional[dict]      = None,
        bertscore_metrics: Optional[dict]     = None,
        meteor_score:     Optional[float]     = None,
        retrieval_metrics: Optional[dict]     = None,
        citations:        Optional[List[str]] = None,
        hyperparameters:  Optional[dict]      = None,
        execution_time:   float               = 0.0,
        status:           str                 = "success",
        error_message:    str                 = "",
        experiment_id:    Optional[str]       = None,
    ) -> None:
        if experiment_id is None:
            experiment_id = f"EXP_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        rouge_scores      = rouge_scores      or {k: {'precision':0,'recall':0,'fmeasure':0} for k in ('rouge1','rouge2','rougeL')}
        bertscore_metrics = bertscore_metrics or {'precision':0,'recall':0,'f1':0}
        retrieval_metrics = retrieval_metrics or {}
        citations         = citations         or []
        hyperparameters   = hyperparameters   or {}

        row = {
            "Timestamp":            datetime.now(),
            "Experiment_ID":        experiment_id,
            "Query":                (query or "")[:1000],
            "Configuration":        configuration,
            "Ground_Truth":         (ground_truth or "")[:5000],
            "Has_Ground_Truth":     1 if ground_truth else 0,
            "Answer":               (answer or "")[:5000],
            "Answer_Length_Words":  len(answer.split()) if answer else 0,
            "Answer_Length_Chars":  len(answer) if answer else 0,
            "Citations":            " | ".join(citations),
            "Citation_Count":       len(citations),
            "FAISS_Top_Similarity": retrieval_metrics.get("faiss_top_similarity", 0),
            "FAISS_Avg_Similarity": retrieval_metrics.get("faiss_avg_similarity", 0),
            "FAISS_Min_Similarity": retrieval_metrics.get("faiss_min_similarity", 0),
            "CrossEncoder_Top_Score": retrieval_metrics.get("ce_top_score", 0),
            "CrossEncoder_Avg_Score": retrieval_metrics.get("ce_avg_score", 0),
            "CrossEncoder_Min_Score": retrieval_metrics.get("ce_min_score", 0),
            "CrossEncoder_Max_Score": retrieval_metrics.get("ce_max_score", 0),
            "Total_Retrieved_Chunks": retrieval_metrics.get("total_chunks", 0),
            "Reranking_Applied":    retrieval_metrics.get("reranking_applied", False),
            "ROUGE1_Precision":     rouge_scores['rouge1']['precision'],
            "ROUGE1_Recall":        rouge_scores['rouge1']['recall'],
            "ROUGE1_F1":            rouge_scores['rouge1']['fmeasure'],
            "ROUGE2_Precision":     rouge_scores['rouge2']['precision'],
            "ROUGE2_Recall":        rouge_scores['rouge2']['recall'],
            "ROUGE2_F1":            rouge_scores['rouge2']['fmeasure'],
            "ROUGEL_Precision":     rouge_scores['rougeL']['precision'],
            "ROUGEL_Recall":        rouge_scores['rougeL']['recall'],
            "ROUGEL_F1":            rouge_scores['rougeL']['fmeasure'],
            "BERTScore_Precision":  bertscore_metrics['precision'],
            "BERTScore_Recall":     bertscore_metrics['recall'],
            "BERTScore_F1":         bertscore_metrics['f1'],
            "METEOR_Score":         meteor_score or 0,
            "chunk_size":           hyperparameters.get('chunk_size', ""),
            "chunk_overlap":        hyperparameters.get('chunk_overlap', ""),
            "top_k_retrieval":      hyperparameters.get('top_k', ""),
            "rerank_enabled":       hyperparameters.get('rerank_enabled', ""),
            "temperature":          hyperparameters.get('temperature', ""),
            "embedding_model":      hyperparameters.get('embedding_model', ""),
            "llm_model":            hyperparameters.get('llm_model', ""),
            "execution_time_seconds": round(execution_time, 3),
            "status":               status,
            "error_message":        (error_message or "")[:200],
        }

        try:
            existing = pd.read_excel(self.excel_path, sheet_name=self.SHEET)
            next_row = len(existing) + 1
            with pd.ExcelWriter(
                self.excel_path, engine="openpyxl", mode="a",
                if_sheet_exists="overlay"
            ) as writer:
                pd.DataFrame([row]).to_excel(
                    writer, sheet_name=self.SHEET,
                    index=False, header=False, startrow=next_row
                )
        except Exception as e:
            print(f"  [logger] Error writing row: {e}")


# ── Comprehensive evaluator ───────────────────────────────────────────────────

class ComprehensiveEvaluator:
    """Compute NLG metrics and run the RAG-vs-Vanilla ablation study."""

    def __init__(self, experiment_logger: ExperimentLogger):
        self.logger        = experiment_logger
        self.rouge_sc      = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
        )
        self.meteor_metric = hf_evaluate.load("meteor")

    # ── Metric helpers ────────────────────────────────────────────────────

    def calculate_rouge(self, prediction: str, reference: str) -> dict:
        empty = {k: {'precision':0,'recall':0,'fmeasure':0}
                 for k in ('rouge1','rouge2','rougeL')}
        if not reference or not reference.strip():
            return empty
        s = self.rouge_sc.score(reference, prediction)
        return {
            'rouge1': {'precision': s['rouge1'].precision, 'recall': s['rouge1'].recall,  'fmeasure': s['rouge1'].fmeasure},
            'rouge2': {'precision': s['rouge2'].precision, 'recall': s['rouge2'].recall,  'fmeasure': s['rouge2'].fmeasure},
            'rougeL': {'precision': s['rougeL'].precision, 'recall': s['rougeL'].recall,  'fmeasure': s['rougeL'].fmeasure},
        }

    def calculate_bertscore(self, prediction: str, reference: str) -> dict:
        if not reference or not prediction:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        try:
            P, R, F1 = bert_score(
                [prediction], [reference], lang="en", verbose=False, device="cpu"
            )
            return {'precision': float(P[0]), 'recall': float(R[0]), 'f1': float(F1[0])}
        except Exception:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    def calculate_meteor(self, prediction: str, reference: str) -> float:
        if not reference:
            return 0.0
        try:
            result = self.meteor_metric.compute(
                predictions=[prediction], references=[reference]
            )
            return float(result['meteor'])
        except Exception:
            return 0.0

    # ── Ablation study ────────────────────────────────────────────────────

    def run_ablation_study(
        self,
        indexer,
        model,
        tokenizer,
        queries:       List[str],
        ground_truths: Optional[List[Optional[str]]] = None,
        hyperparameters: Optional[dict] = None,
    ) -> dict:
        from .config import DEFAULT_HYPERPARAMS
        if hyperparameters is None:
            hyperparameters = DEFAULT_HYPERPARAMS.copy()
        if ground_truths is None:
            ground_truths = [None] * len(queries)

        configurations = [
            {'name': 'RAG',     'use_rag': True},
            {'name': 'Vanilla', 'use_rag': False},
        ]
        all_results: List[dict] = []

        print("=" * 70)
        print("ABLATION STUDY: RAG vs Vanilla")
        for k, v in hyperparameters.items():
            print(f"  {k:<20}: {v}")
        print("=" * 70)

        for config in configurations:
            config_name = config['name']
            print(f"\n{'='*70}\nCONFIGURATION: {config_name}\n{'='*70}")

            for idx, (query, ground_truth) in enumerate(
                zip(queries, ground_truths), 1
            ):
                print(f"\nQuery {idx}/{len(queries)}: {query[:80]} …")
                try:
                    if config['use_rag']:
                        result = rag_generate(
                            indexer, model, tokenizer, query,
                            k=hyperparameters['top_k'],
                            use_reranking=hyperparameters['rerank_enabled'],
                        )
                    else:
                        result = vanilla_query(model, tokenizer, query)
                        result['retrieval_metrics'] = {}
                        result['citations']         = []

                    answer = result.get('answer') or ''
                    # Safety strip
                    import re as _re
                    m = _re.search(r'</think>(.*)', answer, _re.DOTALL)
                    if m:
                        answer = m.group(1).strip()
                        result['answer'] = answer

                    if len(answer.split()) < 10:
                        print(f"  [WARN] Very short answer ({len(answer.split())} words)")

                    print(f"  Preview: {answer[:120].strip()} …")

                    rouge_sc = bertscore_m = meteor_sc = None
                    if ground_truth and ground_truth.strip() and answer:
                        rouge_sc    = self.calculate_rouge(answer, ground_truth)
                        bertscore_m = self.calculate_bertscore(answer, ground_truth)
                        meteor_sc   = self.calculate_meteor(answer, ground_truth)
                        print(
                            f"  ROUGE-1 F1: {rouge_sc['rouge1']['fmeasure']:.3f} | "
                            f"BERTScore F1: {bertscore_m['f1']:.3f} | "
                            f"METEOR: {meteor_sc:.3f}"
                        )

                    self.logger.log_experiment(
                        query=query,
                        answer=answer,
                        configuration=config_name,
                        ground_truth=ground_truth,
                        rouge_scores=rouge_sc,
                        bertscore_metrics=bertscore_m,
                        meteor_score=meteor_sc,
                        retrieval_metrics=result.get('retrieval_metrics', {}),
                        citations=result.get('citations', []),
                        hyperparameters=hyperparameters,
                        execution_time=result.get('execution_time', 0),
                        status=result.get('status', 'unknown'),
                        error_message=result.get('error', ''),
                        experiment_id=f"{config_name}_{idx:03d}",
                    )

                    all_results.append({
                        'configuration': config_name,
                        'query_idx':     idx,
                        'result':        result,
                    })
                    print(f"  Status: {result.get('status')}")

                except Exception as e:
                    print(f"  [ERROR] {e}")

        print("\nABLATION STUDY COMPLETE")
        return {
            'all_results':    all_results,
            'configurations': [c['name'] for c in configurations],
            'num_queries':    len(queries),
        }
