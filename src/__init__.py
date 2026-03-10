# -*- coding: utf-8 -*-
"""
src/
Knee Arthroplasty RAG — modular source package.

Modules:
  utils      — PageSource, text cleaning, PDF reading, document loading/chunking
  logger     — ExperimentLogger (Excel-based result logging)
  indexer    — CacheManager, OptimizedRAGIndexer (FAISS + cross-encoder)
  retriever  — rag_query (dense retrieval + reranking)
  generator  — rag_generate, vanilla_query (LLM answer generation)
  evaluator  — ComprehensiveEvaluator, OptimizedParameterTuner
"""

from src.utils import PageSource, clean_text, read_pdf, load_documents, split_documents
from src.logger import ExperimentLogger
from src.indexer import CacheManager, OptimizedRAGIndexer
from src.retriever import rag_query
from src.generator import rag_generate, vanilla_query
from src.evaluator import ComprehensiveEvaluator, OptimizedParameterTuner

__all__ = [
    "PageSource", "clean_text", "read_pdf", "load_documents", "split_documents",
    "ExperimentLogger",
    "CacheManager", "OptimizedRAGIndexer",
    "rag_query",
    "rag_generate", "vanilla_query",
    "ComprehensiveEvaluator", "OptimizedParameterTuner",
]
