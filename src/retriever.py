# -*- coding: utf-8 -*-
"""
retriever.py
FAISS-based retrieval with optional cross-encoder reranking.
"""

from typing import Any, Dict, List

import numpy as np

from src.indexer import OptimizedRAGIndexer


def rag_query(
    indexer: OptimizedRAGIndexer,
    query: str,
    k: int = 5,
    use_reranking: bool = True,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Retrieve top-k relevant chunks for a query using FAISS + optional reranking.

    Returns a dict with context, citations, retrieval metrics, and raw chunks.
    """
    try:
        index, metadata = indexer.get_index()

        query_embedding = indexer.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        retrieve_k = k * 2 if use_reranking else k
        scores, indices = index.search(query_embedding, retrieve_k)

        retrieved_chunks = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= len(metadata):
                continue

            meta = metadata.get(idx, {})
            retrieved_chunks.append({
                "index": int(idx),
                "source": meta.get("source", "Unknown"),
                "content": meta.get("content", ""),
                "citation": meta.get("citation", "Unknown"),
                "faiss_similarity": float(score),
                "page_number": meta.get("page_number", 0)
            })

        if use_reranking and len(retrieved_chunks) > k:
            retrieved_chunks = indexer.rerank_results(query, retrieved_chunks, top_k=k)

        faiss_similarities = [c['faiss_similarity'] for c in retrieved_chunks]

        retrieval_metrics = {
            "total_chunks": len(retrieved_chunks),
            "reranking_applied": use_reranking,
            "faiss_top_similarity": max(faiss_similarities) if faiss_similarities else 0,
            "faiss_avg_similarity": sum(faiss_similarities) / len(faiss_similarities) if faiss_similarities else 0,
            "faiss_min_similarity": min(faiss_similarities) if faiss_similarities else 0,
            "faiss_max_similarity": max(faiss_similarities) if faiss_similarities else 0,
        }

        if use_reranking:
            ce_scores = [c.get('ce_score', 0) for c in retrieved_chunks]
            retrieval_metrics.update({
                "ce_top_score": max(ce_scores) if ce_scores else 0,
                "ce_avg_score": sum(ce_scores) / len(ce_scores) if ce_scores else 0,
                "ce_min_score": min(ce_scores) if ce_scores else 0,
                "ce_max_score": max(ce_scores) if ce_scores else 0,
            })
        else:
            retrieval_metrics.update({
                "ce_top_score": 0, "ce_avg_score": 0,
                "ce_min_score": 0, "ce_max_score": 0,
            })

        citations = list(dict.fromkeys([
            chunk.get("citation")
            for chunk in retrieved_chunks
            if chunk.get("citation")
        ]))

        context_chunks = [
            f"[Source: {chunk['citation']}]\n{chunk['content']}"
            for chunk in retrieved_chunks
        ]
        context = "\n\n".join(context_chunks)

        return {
            "query": query,
            "context": context,
            "citations": citations,
            "retrieval_metrics": retrieval_metrics,
            "status": "ready_for_generation",
            "retrieved_chunks": retrieved_chunks
        }

    except Exception as e:
        return {
            "query": query,
            "answer": None,
            "retrieval_metrics": {},
            "status": "error",
            "error": str(e),
        }
