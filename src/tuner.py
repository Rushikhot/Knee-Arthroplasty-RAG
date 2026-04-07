"""
tuner.py
--------
OptimizedParameterTuner: grid-search over (chunk_size, chunk_overlap,
top_k, rerank) using ROUGE-L F1 on a small tuning query set.
"""

from __future__ import annotations

import itertools
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch

from .rag_pipeline import _generate, RAG_PROMPT, _SYSTEM_RULES, _strip_think_block
from .evaluation import ComprehensiveEvaluator


class OptimizedParameterTuner:
    """
    Grid-search hyperparameters for the RAG pipeline.

    Usage
    -----
    tuner = OptimizedParameterTuner(
        indexer, model, tokenizer, cache_manager
    )
    best = tuner.tune(tuning_queries, tuning_ground_truths, save_path="results/best.json")
    """

    GRID = {
        "chunk_sizes":    [600, 800, 1000],
        "chunk_overlaps": [100, 150, 200],
        "top_k_values":   [5, 7, 10],
        "rerank_options": [True, False],
    }

    def __init__(self, indexer, model, tokenizer, cache_manager):
        self.indexer       = indexer
        self.model         = model
        self.tokenizer     = tokenizer
        self.cache_manager = cache_manager
        self.results: List[dict] = []
        self.best_score  = -1.0
        self.best_params: Dict[str, Any] = {}

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _rougeL(prediction: str, reference: str) -> float:
        from rouge_score import rouge_scorer as rs
        scorer = rs.RougeScorer(['rougeL'], use_stemmer=True)
        return scorer.score(reference, prediction)['rougeL'].fmeasure

    def _run_combo(
        self,
        chunk_size: int,
        overlap: int,
        top_k: int,
        rerank: bool,
        queries: List[str],
        ground_truths: List[str],
        verbose: bool,
    ) -> float:
        """Return mean ROUGE-L F1 for one hyperparameter combination."""
        # Build / reload index
        if self.indexer.is_built:
            pass  # reuse existing if params match
        else:
            ok = self.indexer.load_index_from_cache(chunk_size, overlap)
            if not ok:
                print(f"  [tuner] No cache for size={chunk_size} overlap={overlap}, skipping.")
                return 0.0

        self.indexer.load_embedding_model()
        import faiss, numpy as np

        scores_per_query: List[float] = []
        for query, gt in zip(queries, ground_truths):
            try:
                index, metadata = self.indexer.get_index()
                q_emb = self.indexer.embedding_model.encode(
                    [query], convert_to_numpy=True, normalize_embeddings=True
                ).astype("float32")

                k_retrieve = top_k * 2 if rerank else top_k
                raw_scores, indices = index.search(q_emb, k_retrieve)

                chunks = []
                for s, idx in zip(raw_scores[0], indices[0]):
                    if idx == -1 or idx >= len(metadata):
                        continue
                    meta = metadata.get(int(idx), {})
                    chunks.append({"content": meta.get("content", ""), "faiss_similarity": float(s)})

                if rerank and len(chunks) > top_k:
                    self.indexer.load_cross_encoder()
                    pairs = [[query, c['content']] for c in chunks]
                    re_scores = self.indexer.cross_encoder.predict(pairs, show_progress_bar=False)
                    for c, rs in zip(chunks, re_scores):
                        c['ce_score'] = float(rs)
                    chunks = sorted(chunks, key=lambda x: x['ce_score'], reverse=True)[:top_k]

                context = "\n\n".join(c['content'] for c in chunks)
                prompt  = RAG_PROMPT.format(context=context, rules=_SYSTEM_RULES, query=query)
                answer  = _generate(self.model, self.tokenizer, prompt)
                answer  = _strip_think_block(answer)

                rl = self._rougeL(answer, gt)
                scores_per_query.append(rl)
                if verbose:
                    print(f"    ROUGE-L={rl:.3f}  {query[:60]} …")

            except Exception as e:
                print(f"    [ERROR] {e}")
                scores_per_query.append(0.0)

        return sum(scores_per_query) / len(scores_per_query) if scores_per_query else 0.0

    # ── Public API ────────────────────────────────────────────────────────

    def tune(
        self,
        queries:       List[str],
        ground_truths: List[str],
        save_path:     Optional[str] = None,
        verbose:       bool          = True,
    ) -> Dict[str, Any]:
        """
        Run grid search and return the best hyperparameter dict.
        Optionally save best params to *save_path* as JSON.
        """
        g = self.GRID
        combos = list(itertools.product(
            g["chunk_sizes"],
            g["chunk_overlaps"],
            g["top_k_values"],
            g["rerank_options"],
        ))
        print(f"\n=== HYPERPARAMETER TUNING ({len(combos)} combinations) ===")

        for chunk_size, overlap, top_k, rerank in combos:
            label = f"size={chunk_size} overlap={overlap} k={top_k} rerank={rerank}"
            print(f"\n  Testing: {label}")

            avg = self._run_combo(
                chunk_size, overlap, top_k, rerank,
                queries, ground_truths, verbose
            )

            entry = {
                "params": {
                    "chunk_size":     chunk_size,
                    "chunk_overlap":  overlap,
                    "top_k":          top_k,
                    "rerank_enabled": rerank,
                },
                "avg_rougeL_f1": avg,
                "status":        "success",
            }
            self.results.append(entry)

            if avg > self.best_score:
                self.best_score  = avg
                self.best_params = entry["params"]
                print(f"  *** New best: ROUGE-L F1={avg:.3f} → {self.best_params}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"\n=== TUNING COMPLETE ===")
        print(f"  Best ROUGE-L F1 : {self.best_score:.3f}")
        print(f"  Best Params     : {self.best_params}")

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(self.best_params, f, indent=2)
            print(f"  Saved to: {save_path}")

        return self.best_params
