# -*- coding: utf-8 -*-
"""
generator.py
LLM-based answer generation for RAG and vanilla (no-context) modes.
"""

from datetime import datetime
from typing import Any, Dict

import torch

from src.indexer import OptimizedRAGIndexer
from src.retriever import rag_query


def rag_generate(
    indexer: OptimizedRAGIndexer,
    model: Any,
    tokenizer: Any,
    query: str,
    k: int = 5,
    use_reranking: bool = True,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Full RAG pipeline: retrieve context then generate an answer grounded in it.
    """
    start_time = datetime.now()

    try:
        rag_result = rag_query(indexer, query, k, use_reranking, verbose)

        if rag_result["status"] == "error":
            return rag_result

        context = rag_result["context"]

        prompt = f"""You are an expert orthopedic surgeon.

STRICT RULES:
1. Use the information from the CONTEXT below to answer the question.
2. If unsure, state uncertainty explicitly.
3. Do NOT fabricate specific statistics, references, or guidelines.
4. Do NOT speculate beyond standard clinical practice.
5. Keep the answer concise (maximum 150 words).
6. Use bullet points if listing steps.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""

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

        execution_time = (datetime.now() - start_time).total_seconds()

        return {
            "query": query,
            "answer": answer,
            "context": context,
            "citations": rag_result["citations"],
            "retrieval_metrics": rag_result["retrieval_metrics"],
            "status": "success",
            "execution_time": execution_time,
            "retrieved_chunks": rag_result.get("retrieved_chunks", [])
        }

    except Exception as e:
        execution_time = (datetime.now() - start_time).total_seconds()
        return {
            "query": query,
            "answer": None,
            "citations": [],
            "retrieval_metrics": {},
            "status": "error",
            "error": str(e),
            "execution_time": execution_time
        }


def vanilla_query(
    model: Any,
    tokenizer: Any,
    query: str,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Baseline: answer the query using only the LLM's parametric knowledge (no retrieval).
    """
    start_time = datetime.now()

    prompt = f"""You are an expert orthopedic surgeon.

INSTRUCTIONS:
1. Answer based on established orthopedic medical knowledge.
2. If unsure, state uncertainty explicitly.
3. Do NOT fabricate specific statistics, references, or guidelines.
4. Do NOT speculate beyond standard clinical practice.
5. Keep the answer concise (maximum 150 words).
6. Use bullet points if listing steps.

QUESTION:
{query}

ANSWER:
"""

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

    execution_time = (datetime.now() - start_time).total_seconds()

    return {
        "query": query,
        "answer": answer,
        "model": "vanilla_medgemma_27b",
        "execution_time": execution_time,
        "status": "success"
    }
