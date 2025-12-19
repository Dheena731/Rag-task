import os
import pickle
from typing import List

import numpy as np

from llm import ask_profile_bot

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    import faiss
except Exception:
    faiss = None


def load_index_and_meta(index_path: str = "resume_index.faiss", meta_path: str = "resume_meta.pkl"):
    if faiss is None:
        raise ImportError("faiss is required to load index. Install faiss-cpu or appropriate wheel.")
    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        raise FileNotFoundError("Index or metadata not found; run build_faiss.py first.")

    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    return index, meta


def query_index(question: str, top_k: int = 4, index_path: str = "resume_index.faiss", meta_path: str = "resume_meta.pkl") -> List[str]:
    if SentenceTransformer is None:
        raise ImportError("sentence-transformers is required. Install with `pip install sentence-transformers`")

    index, meta = load_index_and_meta(index_path, meta_path)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    q_emb = model.encode([question], convert_to_numpy=True)

    D, I = index.search(q_emb, top_k)
    chunks = []
    for idx in I[0]:
        if idx < len(meta["chunks"]):
            chunks.append(meta["chunks"][idx])
    return chunks


def build_context_from_chunks(chunks: List[str]) -> str:
    # Build a compact context string from retrieved chunks
    parts = [c.strip() for c in chunks if c and len(c.strip())]
    return "\n\n".join(parts)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python query_resume.py 'your question here'")
        raise SystemExit(1)

    question = sys.argv[1]
    chunks = query_index(question)
    context = build_context_from_chunks(chunks)
    ans = ask_profile_bot(question, context)
    print(ans)
