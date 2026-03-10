# -*- coding: utf-8 -*-
"""
logger.py
Experiment logging to Excel for tracking RAG evaluation results.
"""

import os
from datetime import datetime
from typing import Dict, List, Optional, Any

import pandas as pd


class ExperimentLogger:
    def __init__(self, excel_path: str):
        self.excel_path = excel_path
        self.sheet_name = "Experiment_Results"
        self._create_if_not_exists()

    def _create_if_not_exists(self):
        if not os.path.exists(self.excel_path):
            os.makedirs(os.path.dirname(self.excel_path), exist_ok=True)

            headers = pd.DataFrame(columns=[
                "Timestamp", "Experiment_ID", "Query", "Configuration",
                "Ground_Truth", "Has_Ground_Truth", "Answer",
                "Answer_Length_Words", "Answer_Length_Chars",
                "Citations", "Citation_Count",
                "FAISS_Top_Similarity", "FAISS_Avg_Similarity",
                "FAISS_Min_Similarity", "FAISS_Max_Similarity",
                "CrossEncoder_Top_Score", "CrossEncoder_Avg_Score",
                "CrossEncoder_Min_Score", "CrossEncoder_Max_Score",
                "Total_Retrieved_Chunks", "Reranking_Applied",
                "ROUGE1_Precision", "ROUGE1_Recall", "ROUGE1_F1",
                "ROUGE2_Precision", "ROUGE2_Recall", "ROUGE2_F1",
                "ROUGEL_Precision", "ROUGEL_Recall", "ROUGEL_F1",
                "BERTScore_Precision", "BERTScore_Recall", "BERTScore_F1",
                "METEOR_Score", "chunk_size", "chunk_overlap",
                "top_k_retrieval", "rerank_enabled", "temperature",
                "embedding_model", "llm_model",
                "execution_time_seconds", "status", "error_message"
            ])

            with pd.ExcelWriter(self.excel_path, engine="openpyxl") as writer:
                headers.to_excel(writer, sheet_name=self.sheet_name, index=False)

    def log_experiment(
        self,
        query: str,
        answer: str,
        configuration: str,
        ground_truth: Optional[str] = None,
        rouge_scores: Optional[Dict[str, Dict[str, float]]] = None,
        bertscore_metrics: Optional[Dict[str, float]] = None,
        meteor_score: Optional[float] = None,
        retrieval_metrics: Optional[Dict[str, Any]] = None,
        citations: Optional[List[str]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        execution_time: float = 0.0,
        status: str = "success",
        error_message: str = "",
        experiment_id: Optional[str] = None
    ):
        if experiment_id is None:
            experiment_id = f"EXP_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        if rouge_scores is None:
            rouge_scores = {
                'rouge1': {'precision': 0, 'recall': 0, 'fmeasure': 0},
                'rouge2': {'precision': 0, 'recall': 0, 'fmeasure': 0},
                'rougeL': {'precision': 0, 'recall': 0, 'fmeasure': 0}
            }

        if bertscore_metrics is None:
            bertscore_metrics = {'precision': 0, 'recall': 0, 'f1': 0}

        if retrieval_metrics is None:
            retrieval_metrics = {}

        if citations is None:
            citations = []

        if hyperparameters is None:
            hyperparameters = {}

        row_data = {
            "Timestamp": datetime.now(),
            "Experiment_ID": experiment_id,
            "Query": query[:500] if query else "",
            "Configuration": configuration,
            "Ground_Truth": ground_truth[:500] if ground_truth else "",
            "Has_Ground_Truth": 1 if ground_truth else 0,
            "Answer": answer[:1000] if answer else "",
            "Answer_Length_Words": len(answer.split()) if answer else 0,
            "Answer_Length_Chars": len(answer) if answer else 0,
            "Citations": " | ".join(citations) if citations else "",
            "Citation_Count": len(citations),
            "FAISS_Top_Similarity": retrieval_metrics.get("faiss_top_similarity", 0),
            "FAISS_Avg_Similarity": retrieval_metrics.get("faiss_avg_similarity", 0),
            "FAISS_Min_Similarity": retrieval_metrics.get("faiss_min_similarity", 0),
            "FAISS_Max_Similarity": retrieval_metrics.get("faiss_max_similarity", 0),
            "CrossEncoder_Top_Score": retrieval_metrics.get("ce_top_score", 0),
            "CrossEncoder_Avg_Score": retrieval_metrics.get("ce_avg_score", 0),
            "CrossEncoder_Min_Score": retrieval_metrics.get("ce_min_score", 0),
            "CrossEncoder_Max_Score": retrieval_metrics.get("ce_max_score", 0),
            "Total_Retrieved_Chunks": retrieval_metrics.get("total_chunks", 0),
            "Reranking_Applied": retrieval_metrics.get("reranking_applied", False),
            "ROUGE1_Precision": rouge_scores.get('rouge1', {}).get('precision', 0),
            "ROUGE1_Recall": rouge_scores.get('rouge1', {}).get('recall', 0),
            "ROUGE1_F1": rouge_scores.get('rouge1', {}).get('fmeasure', 0),
            "ROUGE2_Precision": rouge_scores.get('rouge2', {}).get('precision', 0),
            "ROUGE2_Recall": rouge_scores.get('rouge2', {}).get('recall', 0),
            "ROUGE2_F1": rouge_scores.get('rouge2', {}).get('fmeasure', 0),
            "ROUGEL_Precision": rouge_scores.get('rougeL', {}).get('precision', 0),
            "ROUGEL_Recall": rouge_scores.get('rougeL', {}).get('recall', 0),
            "ROUGEL_F1": rouge_scores.get('rougeL', {}).get('fmeasure', 0),
            "BERTScore_Precision": bertscore_metrics.get('precision', 0),
            "BERTScore_Recall": bertscore_metrics.get('recall', 0),
            "BERTScore_F1": bertscore_metrics.get('f1', 0),
            "METEOR_Score": meteor_score if meteor_score is not None else 0,
            "chunk_size": hyperparameters.get('chunk_size', ""),
            "chunk_overlap": hyperparameters.get('chunk_overlap', ""),
            "top_k_retrieval": hyperparameters.get('top_k', ""),
            "rerank_enabled": hyperparameters.get('rerank_enabled', ""),
            "temperature": hyperparameters.get('temperature', ""),
            "embedding_model": hyperparameters.get('embedding_model', ""),
            "llm_model": hyperparameters.get('llm_model', ""),
            "execution_time_seconds": round(execution_time, 3),
            "status": status,
            "error_message": error_message[:200] if error_message else ""
        }

        try:
            if os.path.exists(self.excel_path):
                existing_df = pd.read_excel(self.excel_path, sheet_name=self.sheet_name)
                next_row = len(existing_df) + 1

                with pd.ExcelWriter(
                    self.excel_path,
                    engine="openpyxl",
                    mode="a",
                    if_sheet_exists="overlay"
                ) as writer:
                    pd.DataFrame([row_data]).to_excel(
                        writer,
                        sheet_name=self.sheet_name,
                        index=False,
                        header=False,
                        startrow=next_row
                    )
            else:
                pd.DataFrame([row_data]).to_excel(
                    self.excel_path,
                    sheet_name=self.sheet_name,
                    index=False
                )

        except Exception as e:
            print(f"Error logging experiment: {e}")
