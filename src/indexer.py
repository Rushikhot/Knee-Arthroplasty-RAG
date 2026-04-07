"""
indexer.py
----------
OptimizedRAGIndexer: builds and queries a FAISS index backed by
BGE-M3 dense embeddings, with optional BGE cross-encoder reranking.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
import torch

from .cache_manager import CacheManager
from .data_io import split_documents


class OptimizedRAGIndexer:
    """
    Manages the full retrieval pipeline:
      1. Chunk → embed (BGE-M3) → FAISS index
      2. Optional cross-encoder (BGE-reranker-v2-m3) reranking
    """

    def __init__(self, embedding_model_name: str, cache_manager: CacheManager):
        self.embedding_model_name = embedding_model_name
        self.cache_manager        = cache_manager
        self.embedding_model      = None
        self.cross_encoder        = None
        self.index:          Optional[faiss.Index]     = None
        self.chunk_metadata: Optional[Dict[int, dict]] = None
        self.chunk_embeddings: Optional[np.ndarray]    = None
        self.is_built = False

    # ── Model loading ─────────────────────────────────────────────────────

    def load_embedding_model(self) -> None:
        if self.embedding_model is not None:
            return
        from sentence_transformers import SentenceTransformer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Loading embedding model ({self.embedding_model_name}) on {device} …")
        self.embedding_model = SentenceTransformer(
            self.embedding_model_name, device=device
        )
        print("  Embedding model ready.")

    def load_cross_encoder(self) -> None:
        if self.cross_encoder is not None:
            return
        from sentence_transformers import CrossEncoder
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("  Loading cross-encoder …")
        self.cross_encoder = CrossEncoder(
            "BAAI/bge-reranker-v2-m3",
            device=device,
            max_length=1024,
        )
        print("  Cross-encoder ready.")

    # ── Index building ────────────────────────────────────────────────────

    def build_index(
        self,
        documents: Dict[str, List[str]],
        doc_sources: Dict,
        chunk_size: int,
        chunk_overlap: int,
        force_rebuild: bool = False,
    ) -> None:
        print(f"\n=== BUILDING INDEX (chunk_size={chunk_size}, overlap={chunk_overlap}) ===")

        # Step 1 – chunks
        if not force_rebuild and self.cache_manager.chunks_exist(chunk_size, chunk_overlap):
            chunks = self.cache_manager.load_chunks(chunk_size, chunk_overlap)
        else:
            chunks = split_documents(documents, doc_sources, chunk_size, chunk_overlap)
            self.cache_manager.save_chunks(chunks, chunk_size, chunk_overlap)

        # Step 2 – embeddings
        if not force_rebuild and self.cache_manager.embeddings_exist(
            chunk_size, chunk_overlap, self.embedding_model_name
        ):
            embeddings, chunk_metadata = self.cache_manager.load_embeddings(
                chunk_size, chunk_overlap, self.embedding_model_name
            )
        else:
            self.load_embedding_model()
            batch_size = 64 if torch.cuda.is_available() else 16
            embeddings = self.embedding_model.encode(
                [c["content"] for c in chunks],
                convert_to_numpy=True,
                show_progress_bar=True,
                batch_size=batch_size,
                normalize_embeddings=True,
            ).astype("float32")

            chunk_metadata = {
                i: {
                    "source":      c["source"],
                    "chunk_id":    c["chunk_id"],
                    "content":     c["content"],
                    "citation":    c["citation"],
                    "word_count":  c["word_count"],
                    "char_count":  c["char_count"],
                    "page_number": c["page_number"],
                }
                for i, c in enumerate(chunks)
            }
            self.cache_manager.save_embeddings(
                embeddings, chunk_metadata, chunk_size, chunk_overlap,
                self.embedding_model_name
            )

        # Step 3 – FAISS
        self._build_faiss(embeddings, chunk_metadata)
        print("=== Index build complete ===\n")

    def load_index_from_cache(self, chunk_size: int, chunk_overlap: int) -> bool:
        """
        Restore FAISS index from saved embeddings without needing raw PDFs.
        Returns True on success.
        """
        if not self.cache_manager.embeddings_exist(
            chunk_size, chunk_overlap, self.embedding_model_name
        ):
            print("  No cached embeddings found for these parameters.")
            return False

        embeddings, chunk_metadata = self.cache_manager.load_embeddings(
            chunk_size, chunk_overlap, self.embedding_model_name
        )
        self._build_faiss(embeddings, chunk_metadata)
        print(f"  Index loaded from cache ({self.index.ntotal} vectors).")
        return True

    def _build_faiss(
        self,
        embeddings: np.ndarray,
        chunk_metadata: Dict[int, dict],
    ) -> None:
        dimension  = embeddings.shape[1]
        index      = faiss.IndexFlatIP(dimension)   # inner-product == cosine for L2-normed vecs
        index.add(embeddings)
        self.index          = index
        self.chunk_metadata = chunk_metadata
        self.chunk_embeddings = embeddings
        self.is_built       = True
        print(f"  FAISS index: {index.ntotal} vectors (dim={dimension}).")

    # ── Retrieval helpers ─────────────────────────────────────────────────

    def get_index(self) -> Tuple[faiss.Index, Dict[int, dict]]:
        if not self.is_built:
            raise RuntimeError("Index not built. Call build_index() or load_index_from_cache() first.")
        return self.index, self.chunk_metadata

    def rerank_results(
        self,
        query: str,
        chunks: List[dict],
        top_k: int,
    ) -> List[dict]:
        self.load_cross_encoder()
        pairs = [[query, c["content"]] for c in chunks]
        scores = self.cross_encoder.predict(pairs, show_progress_bar=False)
        for c, s in zip(chunks, scores):
            c["ce_score"] = float(s)
        ranked = sorted(chunks, key=lambda x: x["ce_score"], reverse=True)
        return ranked[:top_k]
