# -*- coding: utf-8 -*-
"""
evaluator.py
Comprehensive evaluation: ROUGE, BERTScore, METEOR, and RAG vs Vanilla ablation study.
Includes OptimizedParameterTuner for hyperparameter search with index caching.
"""

import itertools
import random
from typing import Any, Dict, List, Optional, Tuple

import torch
from bert_score import score as bert_score
from rouge_score import rouge_scorer
from tqdm import tqdm
import evaluate

from src.indexer import OptimizedRAGIndexer
from src.generator import rag_generate, vanilla_query
from src.logger import ExperimentLogger


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

class ComprehensiveEvaluator:
    def __init__(self, experiment_logger: ExperimentLogger):
        self.logger = experiment_logger
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.meteor_metric = evaluate.load("meteor")

    def calculate_rouge(self, prediction: str, reference: str) -> Dict:
        if not reference or not reference.strip():
            return {
                'rouge1': {'precision': 0, 'recall': 0, 'fmeasure': 0},
                'rouge2': {'precision': 0, 'recall': 0, 'fmeasure': 0},
                'rougeL': {'precision': 0, 'recall': 0, 'fmeasure': 0}
            }

        scores = self.rouge_scorer.score(reference, prediction)
        return {
            'rouge1': {
                'precision': scores['rouge1'].precision,
                'recall': scores['rouge1'].recall,
                'fmeasure': scores['rouge1'].fmeasure
            },
            'rouge2': {
                'precision': scores['rouge2'].precision,
                'recall': scores['rouge2'].recall,
                'fmeasure': scores['rouge2'].fmeasure
            },
            'rougeL': {
                'precision': scores['rougeL'].precision,
                'recall': scores['rougeL'].recall,
                'fmeasure': scores['rougeL'].fmeasure
            }
        }

    def calculate_bertscore(self, prediction: str, reference: str) -> Dict[str, float]:
        if not reference or not prediction:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

        try:
            P, R, F1 = bert_score([prediction], [reference], lang="en", verbose=False, device="cpu")
            return {'precision': float(P[0]), 'recall': float(R[0]), 'f1': float(F1[0])}
        except Exception:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    def calculate_meteor(self, prediction: str, reference: str) -> float:
        if not reference:
            return 0.0

        try:
            result = self.meteor_metric.compute(predictions=[prediction], references=[reference])
            return float(result['meteor'])
        except Exception:
            return 0.0

    def run_ablation_study(
        self,
        indexer: OptimizedRAGIndexer,
        model: Any,
        tokenizer: Any,
        queries: List[str],
        ground_truths: Optional[List[str]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run RAG vs Vanilla ablation across all provided queries.
        Logs all results to the ExperimentLogger.
        """
        from src.configs import EMBEDDING_MODEL_NAME, LLM_MODEL_NAME  # avoid circular at top

        if hyperparameters is None:
            hyperparameters = {
                'chunk_size': 500,
                'chunk_overlap': 100,
                'top_k': 5,
                'rerank_enabled': True,
                'temperature': 0.0,
                'embedding_model': EMBEDDING_MODEL_NAME,
                'llm_model': LLM_MODEL_NAME
            }

        if ground_truths is None:
            ground_truths = [None] * len(queries)

        print("=" * 70)
        print("ABLATION STUDY: RAG vs Vanilla")
        print("=" * 70)
        print(f"Queries: {len(queries)}")
        print("=" * 70 + "\n")

        configurations = [
            {'name': 'RAG', 'use_rag': True},
            {'name': 'Vanilla', 'use_rag': False}
        ]

        all_results = []

        for config in configurations:
            config_name = config['name']
            print(f"\n{'='*70}")
            print(f"CONFIGURATION: {config_name}")
            print(f"{'='*70}\n")

            for idx, (query, ground_truth) in enumerate(zip(queries, ground_truths), 1):
                print(f"Query {idx}/{len(queries)}: {query[:80]}...")

                try:
                    if config['use_rag']:
                        result = rag_generate(
                            indexer, model, tokenizer, query,
                            k=hyperparameters['top_k'],
                            use_reranking=hyperparameters['rerank_enabled'],
                            verbose=False
                        )
                    else:
                        result = vanilla_query(model, tokenizer, query, verbose=False)
                        result['retrieval_metrics'] = {}
                        result['citations'] = []

                    answer = result.get('answer') or ''

                    rouge_scores = None
                    bertscore_metrics = None
                    meteor_score = None

                    if ground_truth and ground_truth.strip() and answer:
                        rouge_scores = self.calculate_rouge(answer, ground_truth)
                        bertscore_metrics = self.calculate_bertscore(answer, ground_truth)
                        meteor_score = self.calculate_meteor(answer, ground_truth)

                    self.logger.log_experiment(
                        query=query,
                        answer=answer,
                        configuration=config_name,
                        ground_truth=ground_truth,
                        rouge_scores=rouge_scores,
                        bertscore_metrics=bertscore_metrics,
                        meteor_score=meteor_score,
                        retrieval_metrics=result.get('retrieval_metrics', {}),
                        citations=result.get('citations', []),
                        hyperparameters=hyperparameters,
                        execution_time=result.get('execution_time', 0),
                        status=result.get('status', 'unknown'),
                        error_message=result.get('error', ''),
                        experiment_id=f"{config_name}_{idx:03d}"
                    )

                    all_results.append({
                        'configuration': config_name,
                        'query_idx': idx,
                        'result': result
                    })

                    print(f"   Status: {result.get('status')}")

                except Exception as e:
                    print(f"   Error: {e}")

        print("ABLATION STUDY COMPLETE")
        return {
            'all_results': all_results,
            'configurations': [c['name'] for c in configurations],
            'num_queries': len(queries)
        }


# ---------------------------------------------------------------------------
# Hyperparameter tuner
# ---------------------------------------------------------------------------

class OptimizedParameterTuner:

    PARAMETER_SPACE = {
        'chunk_size': [400, 500, 800],
        'chunk_overlap': [50, 100],
        'top_k': [3, 5, 7],
        'rerank_enabled': [True, False],
    }

    def __init__(self, indexer: OptimizedRAGIndexer, seed: int = 42):
        self.indexer = indexer
        self.results = []
        self.best_params = None
        self.best_score = 0.0

        random.seed(seed)
        torch.manual_seed(seed)

    def _bertscore(self, prediction: str, reference: str) -> float:
        if not reference or not prediction:
            return 0.0
        try:
            P, R, F1 = bert_score([prediction], [reference], lang="en", verbose=False, device="cpu")
            return float(F1[0])
        except Exception:
            return 0.0

    def tune_parameters(
        self,
        documents: Dict,
        doc_sources: Dict,
        test_queries: List[str],
        ground_truths: List[str],
        model: Any,
        tokenizer: Any,
        num_combinations: int = 15,
        verbose: bool = True
    ) -> Optional[Dict]:
        print("OPTIMIZED HYPERPARAMETER TUNING (WITH CACHING)")
        print(f"Test queries: {len(test_queries)}")
        print(f"Metric: BERTScore F1")

        all_combinations = list(itertools.product(
            self.PARAMETER_SPACE['chunk_size'],
            self.PARAMETER_SPACE['chunk_overlap'],
            self.PARAMETER_SPACE['top_k'],
            self.PARAMETER_SPACE['rerank_enabled']
        ))

        test_combos = (
            random.sample(all_combinations, num_combinations)
            if len(all_combinations) > num_combinations
            else all_combinations
        )

        print(f"Testing {len(test_combos)} combinations...\n")

        # Group by (chunk_size, overlap) to minimise index rebuilds
        combo_groups: Dict[Tuple, List] = {}
        for combo in test_combos:
            key = (combo[0], combo[1])
            combo_groups.setdefault(key, []).append(combo)

        print(f"Optimized: only {len(combo_groups)} index builds needed\n")

        for group_idx, ((chunk_size, overlap), group_combos) in enumerate(combo_groups.items(), 1):
            print(f"\n[Group {group_idx}/{len(combo_groups)}] size={chunk_size}, overlap={overlap}")

            self.indexer.build_index(
                documents=documents,
                doc_sources=doc_sources,
                chunk_size=chunk_size,
                chunk_overlap=overlap
            )

            for combo in tqdm(group_combos, desc=f"Group {group_idx} combos"):
                _, _, top_k, rerank = combo
                combo_scores = []

                for query, ground_truth in zip(test_queries, ground_truths):
                    try:
                        query_embedding = self.indexer.embedding_model.encode(
                            [query], convert_to_numpy=True, normalize_embeddings=True
                        ).astype("float32")

                        index, metadata = self.indexer.get_index()
                        retrieve_k = top_k * 2 if rerank else top_k
                        scores, indices = index.search(query_embedding, retrieve_k)

                        retrieved_chunks = []
                        for score, idx in zip(scores[0], indices[0]):
                            if idx == -1 or idx >= len(metadata):
                                continue
                            meta = metadata.get(idx, {})
                            retrieved_chunks.append({
                                "content": meta.get("content", ""),
                                "faiss_similarity": float(score)
                            })

                        if rerank and len(retrieved_chunks) > top_k:
                            self.indexer.load_cross_encoder()
                            pairs = [[query, c['content']] for c in retrieved_chunks]
                            rerank_scores = self.indexer.cross_encoder.predict(pairs, show_progress_bar=False)
                            for i, chunk in enumerate(retrieved_chunks):
                                chunk['ce_score'] = float(rerank_scores[i])
                            retrieved_chunks = sorted(
                                retrieved_chunks, key=lambda x: x['ce_score'], reverse=True
                            )[:top_k]

                        context = "\n\n".join([c['content'] for c in retrieved_chunks])

                        prompt = (
                            "You are an expert orthopedic surgeon. "
                            "Answer the question using the provided context.\n\n"
                            f"CONTEXT:\n{context}\n\nQUESTION:\n{query}\n\nANSWER:\n"
                        )

                        inputs = tokenizer(
                            prompt, return_tensors="pt", truncation=True, max_length=4096
                        ).to(model.device)

                        with torch.no_grad():
                            outputs = model.generate(
                                **inputs,
                                max_new_tokens=400,
                                temperature=0.0,
                                do_sample=False,
                                pad_token_id=tokenizer.eos_token_id
                            )

                        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
                        answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

                        combo_scores.append(self._bertscore(answer, ground_truth))

                    except Exception as e:
                        print(f"\nError: {e}")
                        combo_scores.append(0.0)

                avg_score = sum(combo_scores) / len(combo_scores) if combo_scores else 0.0

                result_entry = {
                    "params": {
                        "chunk_size": chunk_size,
                        "chunk_overlap": overlap,
                        "top_k": top_k,
                        "rerank_enabled": rerank,
                    },
                    "avg_bertscore": avg_score,
                    "query_bertscores": combo_scores,
                    "status": "success"
                }

                self.results.append(result_entry)

                if avg_score > self.best_score:
                    self.best_score = avg_score
                    self.best_params = result_entry["params"]
                    if verbose:
                        print(f"\nNew best! BERTScore={avg_score:.3f}, params={self.best_params}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print("TUNING COMPLETE")
        print(f"Best BERTScore: {self.best_score:.3f}")
        print(f"Best Params: {self.best_params}")
        return self.best_params
