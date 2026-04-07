"""
rag_pipeline.py
---------------
rag_query   – retrieve context chunks from FAISS (+ optional reranking)
rag_generate – retrieve + generate with MedGemma
vanilla_query – generate without retrieval (baseline)
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch


# ── Prompt template ───────────────────────────────────────────────────────────

_SYSTEM_RULES = """\
STRICT RULES:
1. Answer DIRECTLY — do NOT show reasoning, analysis steps, or thought process.
2. Do NOT use headings like "Analyze", "Synthesize", "Review Sources", or "Final Check".
3. Use numbered bullet points ONLY for actual answer steps.
4. If unsure, state uncertainty explicitly.
5. Do NOT fabricate statistics, references, or guidelines.
6. Do NOT speculate beyond standard clinical practice.
7. Maximum 300 words."""

RAG_PROMPT = """\
You are an expert orthopedic surgeon. Answer the question directly and \
concisely using ONLY the provided context.

CONTEXT:
{context}

{rules}

QUESTION:
{query}

ANSWER (direct, no reasoning steps):
"""

VANILLA_PROMPT = """\
You are an expert orthopedic surgeon. Answer the question directly and \
concisely based on established orthopedic medical knowledge.

{rules}

QUESTION:
{query}

ANSWER (direct, no reasoning steps):
"""


def _strip_think_block(text: str) -> str:
    """Remove any <think>...</think> block emitted by reasoning models."""
    m = re.search(r'</think>(.*)', text, re.DOTALL)
    return m.group(1).strip() if m else text


def _generate(model, tokenizer, prompt: str) -> str:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=8192,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2000,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return _strip_think_block(text)


# ── Public API ────────────────────────────────────────────────────────────────

def rag_query(
    indexer,
    query: str,
    k: int = 5,
    use_reranking: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Retrieve top-k chunks for *query* and return metadata + context string.
    Does NOT call the LLM.
    """
    import numpy as np
    try:
        index, metadata = indexer.get_index()

        query_embedding = indexer.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        retrieve_k = k * 2 if use_reranking else k
        scores, indices = index.search(query_embedding, retrieve_k)

        retrieved: List[dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= len(metadata):
                continue
            meta = metadata.get(int(idx), {})
            retrieved.append({
                "index":            int(idx),
                "source":           meta.get("source", "Unknown"),
                "content":          meta.get("content", ""),
                "citation":         meta.get("citation", "Unknown"),
                "faiss_similarity": float(score),
                "page_number":      meta.get("page_number", 0),
            })

        if use_reranking and len(retrieved) > k:
            retrieved = indexer.rerank_results(query, retrieved, top_k=k)

        faiss_sims = [c["faiss_similarity"] for c in retrieved]
        retrieval_metrics: Dict[str, Any] = {
            "total_chunks":        len(retrieved),
            "reranking_applied":   use_reranking,
            "faiss_top_similarity": max(faiss_sims, default=0),
            "faiss_avg_similarity": sum(faiss_sims) / len(faiss_sims) if faiss_sims else 0,
            "faiss_min_similarity": min(faiss_sims, default=0),
            "faiss_max_similarity": max(faiss_sims, default=0),
        }
        if use_reranking:
            ce_scores = [c.get("ce_score", 0) for c in retrieved]
            retrieval_metrics.update({
                "ce_top_score": max(ce_scores, default=0),
                "ce_avg_score": sum(ce_scores) / len(ce_scores) if ce_scores else 0,
                "ce_min_score": min(ce_scores, default=0),
                "ce_max_score": max(ce_scores, default=0),
            })
        else:
            retrieval_metrics.update(
                ce_top_score=0, ce_avg_score=0, ce_min_score=0, ce_max_score=0
            )

        citations = list(dict.fromkeys(
            c["citation"] for c in retrieved if c.get("citation")
        ))
        context = "\n\n".join(
            f"[Source: {c['citation']}]\n{c['content']}" for c in retrieved
        )

        return {
            "query":             query,
            "context":           context,
            "citations":         citations,
            "retrieval_metrics": retrieval_metrics,
            "retrieved_chunks":  retrieved,
            "status":            "ready_for_generation",
        }

    except Exception as e:
        return {
            "query":             query,
            "answer":            None,
            "retrieval_metrics": {},
            "status":            "error",
            "error":             str(e),
        }


def rag_generate(
    indexer,
    model,
    tokenizer,
    query: str,
    k: int = 5,
    use_reranking: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Retrieve + generate an answer with the LLM."""
    t0 = datetime.now()
    try:
        rag_result = rag_query(indexer, query, k, use_reranking, verbose)
        if rag_result["status"] == "error":
            return rag_result

        prompt = RAG_PROMPT.format(
            context=rag_result["context"],
            rules=_SYSTEM_RULES,
            query=query,
        )
        answer = _generate(model, tokenizer, prompt)

        return {
            "query":             query,
            "answer":            answer,
            "context":           rag_result["context"],
            "citations":         rag_result["citations"],
            "retrieval_metrics": rag_result["retrieval_metrics"],
            "retrieved_chunks":  rag_result.get("retrieved_chunks", []),
            "status":            "success",
            "execution_time":    (datetime.now() - t0).total_seconds(),
        }
    except Exception as e:
        return {
            "query":          query,
            "answer":         None,
            "citations":      [],
            "retrieval_metrics": {},
            "status":         "error",
            "error":          str(e),
            "execution_time": (datetime.now() - t0).total_seconds(),
        }


def vanilla_query(
    model,
    tokenizer,
    query: str,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Generate an answer without retrieval (baseline comparison)."""
    t0 = datetime.now()
    prompt = VANILLA_PROMPT.format(rules=_SYSTEM_RULES, query=query)
    answer = _generate(model, tokenizer, prompt)
    return {
        "query":          query,
        "answer":         answer,
        "model":          "vanilla",
        "execution_time": (datetime.now() - t0).total_seconds(),
        "status":         "success",
    }
