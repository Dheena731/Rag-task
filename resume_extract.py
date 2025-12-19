import os
from typing import List

try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None


def extract_text_from_pdf(path: str) -> str:
    """Extract text from a PDF file. Tries PyPDF2 first; if unavailable, raises.

    Returns the full concatenated text.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    if PdfReader is None:
        raise ImportError("PyPDF2 is required to extract PDF text. Install with `pip install PyPDF2`")

    reader = PdfReader(path)
    pages = []
    for p in reader.pages:
        try:
            text = p.extract_text() or ""
        except Exception:
            text = ""
        pages.append(text)

    return "\n\n".join(pages).strip()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Simple word-based sliding window chunker.

    chunk_size and overlap are measured in words.
    """
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    n = len(words)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = end - overlap
    return chunks


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python resume_extract.py <path-to-pdf>")
        raise SystemExit(1)

    path = sys.argv[1]
    txt = extract_text_from_pdf(path)
    print(txt[:2000])
