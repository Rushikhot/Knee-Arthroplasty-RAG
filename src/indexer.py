# -*- coding: utf-8 -*-
"""
indexer.py
Handles caching, embedding generation, FAISS index construction, and cross-encoder reranking.
"""

import os
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

from src.utils import PageSource, split_documents


class CacheManager:
    def __init__(self, cache_folder: str):
        self.cache_folder = cache_folder
        os.makedirs(cache_folder, exist_ok=True)

    def _get_chunks_path(self, chunk_size: int, chunk_overlap: int) -> str:
        return os.path.join(
            self.cache_folder,
            f"chunks_size{chunk_size}_overlap{chunk_overlap}.pkl"
        )

    def _get_embeddings_path(self, chunk_size: int, chunk_overlap: int, model_name: str) -> str:
        model_short = model_name.split('/')[-1]
        return os.path.join(
            self.cache_folder,
            f"embeddings_size{chunk_size}_overlap{chunk_overlap}_{model_short}.pkl"
        )

    def chunks_exist(self, chunk_size: int, chunk_overlap: int) -> bool:
        path = self._get_chunks_path(chunk_size, chunk_overlap)
        exists = os.path.exists(path)
        if exists:
            print(f"Found cached chunks: {os.path.basename(path)}")
        return exists

    def embeddings_exist(self, chunk_size: int, chunk_overlap: int, model_name: str) -> bool:
        path = self._get_embeddings_path(chunk_size, chunk_overlap, model_name)
        exists = os.path.exists(path)
        if exists:
            print(f"Found cached embeddings: {os.path.basename(path)}")
        return exists

    def save_chunks(self, chunks: List[Dict], chunk_size: int, chunk_overlap: int):
        path = self._get_chunks_path(chunk_size, chunk_overlap)
        print(f"Saving {len(chunks)} chunks to cache...")
        with open(path, 'wb') as f:
            pickle.dump(chunks, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Chunks saved: {os.path.basename(path)}")

    def load_chunks(self, chunk_size: int, chunk_overlap: int) -> List[Dict]:
        path = self._get_chunks_path(chunk_size, chunk_overlap)
        print("Loading chunks from cache...")
        with open(path, 'rb') as f:
            chunks = pickle.load(f)
        print(f"Loaded {len(chunks)} chunks from cache")
        return chunks

    def save_embeddings(
        self,
        embeddings: np.ndarray,
        chunk_metadata: Dict[int, Dict],
        chunk_size: int,
        chunk_overlap: int,
        model_name: str
    ):
        path = self._get_embeddings_path(chunk_size, chunk_overlap, model_name)
        print(f"Saving embeddings to cache ({embeddings.shape})...")

        data = {
            'embeddings': embeddings,
            'chunk_metadata': chunk_metadata,
            'model_name': model_name,
            'timestamp': datetime.now().isoformat()
        }

        with open(path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Embeddings saved: {os.path.basename(path)}")

    def load_embeddings(
        self,
        chunk_size: int,
        chunk_overlap: int,
        model_name: str
    ) -> Tuple[np.ndarray, Dict[int, Dict]]:
        path = self._get_embeddings_path(chunk_size, chunk_overlap, model_name)
        print("Loading embeddings from cache...")

        with open(path, 'rb') as f:
            data = pickle.load(f)

        embeddings = data['embeddings']
        chunk_metadata = data['chunk_metadata']
        print(f"Loaded embeddings from cache ({embeddings.shape})")
        return embeddings, chunk_metadata


class OptimizedRAGIndexer:
    def __init__(self, embedding_model_name: str, cache_manager: CacheManager):
        self.embedding_model_name = embedding_model_name
        self.cache_manager = cache_manager
        self.embedding_model = None
        self.cross_encoder = None
        self.index = None
        self.chunk_metadata = None
        self.chunk_embeddings = None
        self.is_built = False

    def load_embedding_model(self):
        if self.embedding_model is None:
            print(f"Loading embedding model: {self.embedding_model_name}")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.embedding_model = SentenceTransformer(self.embedding_model_name, device=device)
            print(f"Model loaded on {device}")

    def load_cross_encoder(self):
        if self.cross_encoder is None:
            print("Loading cross-encoder...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.cross_encoder = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
                device=device,
                max_length=512
            )
            print("Cross-encoder loaded")

    def build_index(
        self,
        documents: Dict[str, List[str]],
        doc_sources: Dict[str, List[PageSource]],
        chunk_size: int,
        chunk_overlap: int,
        force_rebuild: bool = False
    ) -> None:
        print(f"BUILDING INDEX (chunk_size={chunk_size}, overlap={chunk_overlap})")

        # Step 1: Get or create chunks
        if not force_rebuild and self.cache_manager.chunks_exist(chunk_size, chunk_overlap):
            chunks = self.cache_manager.load_chunks(chunk_size, chunk_overlap)
        else:
            print("Creating chunks from scratch...")
            chunks = split_documents(documents, doc_sources, chunk_size, chunk_overlap)
            self.cache_manager.save_chunks(chunks, chunk_size, chunk_overlap)

        # Step 2: Get or create embeddings
        if not force_rebuild and self.cache_manager.embeddings_exist(
            chunk_size, chunk_overlap, self.embedding_model_name
        ):
            embeddings, chunk_metadata = self.cache_manager.load_embeddings(
                chunk_size, chunk_overlap, self.embedding_model_name
            )
        else:
            print("Generating embeddings from scratch...")
            self.load_embedding_model()

            chunk_texts = [chunk["content"] for chunk in chunks]
            batch_size = 64 if torch.cuda.is_available() else 16

            embeddings = self.embedding_model.encode(
                chunk_texts,
                convert_to_numpy=True,
                show_progress_bar=True,
                batch_size=batch_size,
                normalize_embeddings=True
            ).astype("float32")

            print(f"Embeddings generated: {embeddings.shape}")

            chunk_metadata = {}
            for i, chunk in enumerate(chunks):
                chunk_metadata[i] = {
                    "source": chunk["source"],
                    "chunk_id": chunk["chunk_id"],
                    "content": chunk["content"],
                    "citation": chunk["citation"],
                    "word_count": chunk["word_count"],
                    "char_count": chunk["char_count"],
                    "page_number": chunk["page_number"]
                }

            self.cache_manager.save_embeddings(
                embeddings, chunk_metadata, chunk_size, chunk_overlap, self.embedding_model_name
            )

        # Step 3: Build FAISS index
        print("Building FAISS index...")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        print(f"FAISS index created: {index.ntotal} vectors")

        self.index = index
        self.chunk_metadata = chunk_metadata
        self.chunk_embeddings = embeddings
        self.is_built = True
        print("Index build complete")

    def get_index(self) -> Tuple[faiss.Index, Dict[int, Dict]]:
        if not self.is_built:
            raise ValueError("Index not built. Call build_index() first.")
        return self.index, self.chunk_metadata

    def rerank_results(
        self,
        query: str,
        initial_results: List[Dict],
        top_k: int = 5,
        batch_size: int = 32
    ) -> List[Dict]:
        if not initial_results:
            return []

        self.load_cross_encoder()

        pairs = [[query, chunk['content']] for chunk in initial_results]
        scores = self.cross_encoder.predict(pairs, batch_size=batch_size, show_progress_bar=False)

        for i, result in enumerate(initial_results):
            result['ce_score'] = float(scores[i])

        reranked = sorted(initial_results, key=lambda x: x['ce_score'], reverse=True)
        return reranked[:top_k]
