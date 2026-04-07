"""
data_io.py
----------
PDF loading, text cleaning, and document chunking utilities.
"""

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class PageSource:
    """Lightweight citation object attached to every text chunk."""
    document_name: str
    page_number: int
    book_title: str
    author: Optional[str] = None
    publication_year: Optional[int] = None

    def get_citation(self) -> str:
        parts = []
        if self.author:
            parts.append(self.author.strip())
        parts.append(self.book_title.strip() if self.book_title else self.document_name)
        if self.publication_year:
            parts.append(f"({self.publication_year})")
        return ". ".join(parts) + f". Page {max(1, self.page_number)}"


# ── Text cleaning ─────────────────────────────────────────────────────────────

def clean_text(
    text: str,
    lowercase: bool = False,
    preserve_newlines: bool = False
) -> str:
    """
    Remove control characters while keeping all printable Unicode
    (Greek letters, superscripts, degree symbols, etc.).
    """
    if not text or not isinstance(text, str):
        return ""

    # Strip true control characters only
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', ' ', text)

    if preserve_newlines:
        lines = text.split('\n')
        cleaned = [re.sub(r'[ \t]+', ' ', ln).strip() for ln in lines if ln.strip()]
        text = "\n".join(cleaned)
    else:
        text = re.sub(r'\s+', ' ', text).strip()

    # Keep medical / scientific punctuation
    text = re.sub(
        r'[^\w\s\.\,\;\:\-\+\(\)\/\%\=\<\>\°\±\[\]\'\"\!\?\@\#\&\*\~\`\^\{\}\|\\]',
        '',
        text,
        flags=re.UNICODE,
    )
    if lowercase:
        text = text.lower()
    return text


# ── PDF reading ───────────────────────────────────────────────────────────────

def read_pdf(
    file_path: str,
    book_title: str,
    author: Optional[str] = None,
    publication_year: Optional[int] = None,
) -> Tuple[List[str], List[PageSource]]:
    """
    Extract page-level text from a PDF using pdfplumber.
    Returns parallel lists: (page_texts, page_sources).
    """
    if not os.path.exists(file_path):
        print(f"  [WARN] File not found: {file_path}")
        return [], []

    texts: List[str] = []
    sources: List[PageSource] = []

    try:
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                try:
                    raw = page.extract_text(x_tolerance=2, y_tolerance=3)
                    if not raw or len(raw.strip()) < 100:
                        continue
                    cleaned = clean_text(raw, lowercase=False, preserve_newlines=False)
                    if not cleaned:
                        continue
                    source = PageSource(
                        document_name=os.path.basename(file_path),
                        page_number=page_num + 1,
                        book_title=book_title,
                        author=author,
                        publication_year=publication_year,
                    )
                    texts.append(cleaned)
                    sources.append(source)
                except Exception:
                    continue
    except Exception as e:
        print(f"  [ERROR] reading {file_path}: {e}")

    return texts, sources


def load_documents(
    folder_path: str,
    doc_metadata: Dict[str, dict],
) -> Tuple[Dict[str, List[str]], Dict[str, List[PageSource]]]:
    """
    Load all PDFs from *folder_path*.
    Returns (documents, doc_sources) dicts keyed by filename.
    """
    documents: Dict[str, List[str]] = {}
    doc_sources: Dict[str, List[PageSource]] = {}

    if not os.path.isdir(folder_path):
        print(f"  [WARN] Document folder not found: {folder_path}")
        return documents, doc_sources

    files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
    print(f"  Loading {len(files)} PDF(s) from {folder_path} ...")

    for filename in tqdm(files, desc="Loading PDFs"):
        file_path = os.path.join(folder_path, filename)
        meta = doc_metadata.get(filename, {})
        try:
            texts, sources = read_pdf(
                file_path=file_path,
                book_title=meta.get('book_title', filename),
                author=meta.get('author'),
                publication_year=meta.get('year'),
            )
            if texts:
                documents[filename] = texts
                doc_sources[filename] = sources
        except Exception as e:
            print(f"  [WARN] Skipping {filename}: {e}")

    print(f"  Loaded {len(documents)} document(s).\n")
    return documents, doc_sources


# ── Chunking ──────────────────────────────────────────────────────────────────

def split_documents(
    documents: Dict[str, List[str]],
    doc_sources: Dict[str, List[PageSource]],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List[dict]:
    """
    Split page texts into overlapping chunks using LangChain's splitter.
    Each chunk dict carries its citation string and metadata.
    """
    if not documents:
        return []

    if chunk_overlap >= chunk_size:
        chunk_overlap = chunk_size // 5

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
        length_function=len,
        keep_separator=True,
    )

    all_chunks: List[dict] = []

    for doc_name in tqdm(documents, desc="Chunking"):
        page_texts = documents[doc_name]
        sources    = doc_sources.get(doc_name, [])
        if not sources:
            continue

        for page_idx, page_text in enumerate(page_texts):
            if page_idx >= len(sources):
                continue
            page_source = sources[page_idx]
            if not page_text or len(page_text) < 50:
                continue

            for chunk_id, chunk_text in enumerate(splitter.split_text(page_text)):
                all_chunks.append({
                    "source":      doc_name,
                    "page_number": page_source.page_number,
                    "chunk_id":    f"{doc_name}_p{page_source.page_number}_c{chunk_id}",
                    "content":     chunk_text,
                    "char_count":  len(chunk_text),
                    "word_count":  len(chunk_text.split()),
                    "citation":    page_source.get_citation(),
                })

    print(f"  Created {len(all_chunks)} chunk(s).")
    return all_chunks
