# -*- coding: utf-8 -*-
"""
utils.py
Utility functions for text cleaning, PDF reading, and document loading/chunking.
"""

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm


@dataclass
class PageSource:
    document_name: str
    page_number: int
    book_title: str
    author: Optional[str] = None
    publication_year: Optional[int] = None

    def get_citation(self) -> str:
        parts = []
        if self.author:
            parts.append(self.author.strip())
        if self.book_title:
            parts.append(self.book_title.strip())
        else:
            parts.append(self.document_name)
        if self.publication_year:
            parts.append(f"({self.publication_year})")
        citation = ". ".join(parts)
        citation += f". Page {max(1, self.page_number)}"
        return citation


def clean_text(text: str, lowercase: bool = False, preserve_newlines: bool = False) -> str:
    if not text or not isinstance(text, str):
        return ""

    text = re.sub(r'[\x00-\x1F\x7F]', ' ', text)

    if preserve_newlines:
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = re.sub(r'\s+', ' ', line).strip()
            if line:
                cleaned_lines.append(line)
        text = "\n".join(cleaned_lines)
    else:
        text = re.sub(r'\s+', ' ', text).strip()

    text = re.sub(r'[^\w\s\.\,\;\:\-\+\(\)\/\%\=\<\>\°\±\[\]\'\"]', '', text)

    if lowercase:
        text = text.lower()

    return text


def read_pdf(
    file_path: str,
    book_title: str,
    author: Optional[str] = None,
    publication_year: Optional[int] = None
) -> Tuple[List[str], List[PageSource]]:
    if not os.path.exists(file_path):
        return [], []

    try:
        pdf_reader = PdfReader(file_path)
        texts: List[str] = []
        sources: List[PageSource] = []

        for page_num, page in enumerate(pdf_reader.pages):
            try:
                raw_text = page.extract_text()
                if raw_text is None or len(raw_text.strip()) < 100:
                    continue

                cleaned_text = re.sub(r'\s+', ' ', raw_text.strip())

                source = PageSource(
                    document_name=os.path.basename(file_path),
                    page_number=page_num + 1,
                    book_title=book_title,
                    author=author,
                    publication_year=publication_year
                )

                texts.append(cleaned_text)
                sources.append(source)

            except Exception:
                continue

        return texts, sources

    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
        return [], []


def load_documents(
    folder_path: str,
    doc_metadata: Dict[str, dict]
) -> Tuple[Dict[str, List[str]], Dict[str, List[PageSource]]]:
    documents: Dict[str, List[str]] = {}
    doc_sources: Dict[str, List[PageSource]] = {}

    if not os.path.isdir(folder_path):
        return documents, doc_sources

    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.pdf', '.txt', '.md'))]
    print(f"Loading {len(files)} documents...")

    for filename in tqdm(files, desc="Loading"):
        file_path = os.path.join(folder_path, filename)
        metadata = doc_metadata.get(filename, {})

        try:
            if filename.lower().endswith('.pdf'):
                texts, sources = read_pdf(
                    file_path=file_path,
                    book_title=metadata.get('book_title', filename),
                    author=metadata.get('author'),
                    publication_year=metadata.get('year')
                )

                if texts:
                    documents[filename] = texts
                    doc_sources[filename] = sources
        except Exception as e:
            print(f"Skipping {filename}: {e}")

    print(f"Loaded {len(documents)} documents\n")
    return documents, doc_sources


def split_documents(
    documents: Dict[str, List[str]],
    doc_sources: Dict[str, List[PageSource]],
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> List[Dict]:
    if not documents:
        return []

    if chunk_overlap >= chunk_size:
        chunk_overlap = chunk_size // 5

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
        length_function=len
    )

    all_chunks = []

    for doc_name in tqdm(documents.keys(), desc="Chunking"):
        page_texts = documents[doc_name]
        sources = doc_sources.get(doc_name, [])

        if not sources:
            continue

        for page_idx, page_text in enumerate(page_texts):
            if page_idx >= len(sources):
                continue

            page_source = sources[page_idx]
            cleaned_page = clean_text(page_text, lowercase=False, preserve_newlines=False)

            if not cleaned_page or len(cleaned_page) < 50:
                continue

            chunks = text_splitter.split_text(cleaned_page)

            for chunk_id, chunk_text in enumerate(chunks):
                chunk_obj = {
                    'source': doc_name,
                    'page_number': page_source.page_number,
                    'chunk_id': f"{doc_name}_p{page_source.page_number}_c{chunk_id}",
                    'content': chunk_text,
                    'char_count': len(chunk_text),
                    'word_count': len(chunk_text.split()),
                    'citation': page_source.get_citation(),
                }
                all_chunks.append(chunk_obj)

    print(f"Created {len(all_chunks)} chunks")
    return all_chunks
