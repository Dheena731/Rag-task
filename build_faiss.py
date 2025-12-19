import os
import pickle
from typing import List

import numpy as np

from resume_extract import extract_text_from_pdf, chunk_text

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    import faiss
except Exception:
    faiss = None


def build_faiss_index_from_pdf(pdf_path: str, index_path: str = "resume_index.faiss", meta_path: str = "resume_meta.pkl"):
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text, chunk_size=400, overlap=80)

    if not chunks:
        raise ValueError("No text extracted from PDF or text too short to chunk.")

    if SentenceTransformer is None:
        raise ImportError("sentence-transformers is required. Install with `pip install sentence-transformers`")

    if faiss is None:
        raise ImportError("faiss is required. On many platforms install `faiss-cpu` with pip.")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # persist
    faiss.write_index(index, index_path)
    with open(meta_path, "wb") as f:
        pickle.dump({"chunks": chunks}, f)

    return index_path, meta_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python build_faiss.py <path-to-resume-pdf>")
        raise SystemExit(1)

    pdf = sys.argv[1]
    print("Building FAISS index from", pdf)
    ix, meta = build_faiss_index_from_pdf(pdf)
    print("Saved:", ix, meta)
