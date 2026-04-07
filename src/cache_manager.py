"""
cache_manager.py
----------------
Disk-based cache for chunks and embeddings, keyed by
(chunk_size, chunk_overlap, model_name) to avoid redundant recomputation.
"""

import os
import pickle
from datetime import datetime
from typing import Dict, Tuple

import numpy as np


class CacheManager:
    """Save and restore chunk lists and FAISS-ready embedding arrays."""

    def __init__(self, cache_folder: str):
        self.cache_folder = cache_folder
        os.makedirs(cache_folder, exist_ok=True)

    # ── Internal path helpers ──────────────────────────────────────────────

    def _chunks_path(self, chunk_size: int, chunk_overlap: int) -> str:
        return os.path.join(
            self.cache_folder,
            f"chunks_size{chunk_size}_overlap{chunk_overlap}.pkl"
        )

    def _embeddings_path(
        self, chunk_size: int, chunk_overlap: int, model_name: str
    ) -> str:
        model_short = model_name.split('/')[-1]
        return os.path.join(
            self.cache_folder,
            f"embeddings_size{chunk_size}_overlap{chunk_overlap}_{model_short}.pkl"
        )

    # ── Chunks ────────────────────────────────────────────────────────────

    def chunks_exist(self, chunk_size: int, chunk_overlap: int) -> bool:
        path = self._chunks_path(chunk_size, chunk_overlap)
        exists = os.path.exists(path)
        if exists:
            print(f"  [cache] Found chunks: {os.path.basename(path)}")
        return exists

    def save_chunks(self, chunks, chunk_size: int, chunk_overlap: int) -> None:
        path = self._chunks_path(chunk_size, chunk_overlap)
        print(f"  [cache] Saving {len(chunks)} chunks …")
        with open(path, 'wb') as f:
            pickle.dump(chunks, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  [cache] Chunks saved: {os.path.basename(path)}")

    def load_chunks(self, chunk_size: int, chunk_overlap: int):
        path = self._chunks_path(chunk_size, chunk_overlap)
        print(f"  [cache] Loading chunks …")
        with open(path, 'rb') as f:
            chunks = pickle.load(f)
        print(f"  [cache] Loaded {len(chunks)} chunks.")
        return chunks

    # ── Embeddings ────────────────────────────────────────────────────────

    def embeddings_exist(
        self, chunk_size: int, chunk_overlap: int, model_name: str
    ) -> bool:
        path = self._embeddings_path(chunk_size, chunk_overlap, model_name)
        exists = os.path.exists(path)
        if exists:
            print(f"  [cache] Found embeddings: {os.path.basename(path)}")
        return exists

    def save_embeddings(
        self,
        embeddings: np.ndarray,
        chunk_metadata: Dict[int, dict],
        chunk_size: int,
        chunk_overlap: int,
        model_name: str,
    ) -> None:
        path = self._embeddings_path(chunk_size, chunk_overlap, model_name)
        print(f"  [cache] Saving embeddings {embeddings.shape} …")
        payload = {
            'embeddings':     embeddings,
            'chunk_metadata': chunk_metadata,
            'model_name':     model_name,
            'timestamp':      datetime.now().isoformat(),
        }
        with open(path, 'wb') as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  [cache] Embeddings saved: {os.path.basename(path)}")

    def load_embeddings(
        self, chunk_size: int, chunk_overlap: int, model_name: str
    ) -> Tuple[np.ndarray, Dict[int, dict]]:
        path = self._embeddings_path(chunk_size, chunk_overlap, model_name)
        print(f"  [cache] Loading embeddings …")
        with open(path, 'rb') as f:
            data = pickle.load(f)
        embeddings     = data['embeddings']
        chunk_metadata = data['chunk_metadata']
        print(f"  [cache] Loaded embeddings {embeddings.shape}.")
        return embeddings, chunk_metadata
