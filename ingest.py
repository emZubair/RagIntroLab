"""
Step 1 of the RAG pipeline: LOADING + CHUNKING.

What this does:
  1. Reads a PDF and extracts its raw text, page by page.
  2. Splits that text into small, overlapping "chunks".
  3. Saves the chunks to chunks.json for the next step (embedding).

Run:  python ingest.py /Users/zubair/Downloads/pak-cons.pdf
"""

import bisect
import json
import sys

from pypdf import PdfReader

# ---- Configuration knobs (try changing these later to see the effect) ----
CHUNK_SIZE = 900  # max characters per chunk
CHUNK_OVERLAP = 150  # characters shared between neighbouring chunks
OUTPUT_FILE = "chunks.json"


def load_pdf(path):
    """LOADING: return (full_text, page_starts).

    page_starts[i] is the character offset in full_text where page i+1 begins,
    so we can later figure out which page any chunk came from.
    """
    reader = PdfReader(path)
    pages_text = []
    page_starts = []
    cursor = 0
    for page in reader.pages:
        text = (page.extract_text() or "").strip()
        page_starts.append(cursor)
        pages_text.append(text)
        cursor += len(text) + 1  # +1 for the "\n" we join with below
    full_text = "\n".join(pages_text)
    return full_text, page_starts


def page_for_offset(offset, page_starts):
    """Given a character offset, return the 1-based page number it falls on."""
    # bisect_right finds the first page_start greater than offset; subtract 1.
    return bisect.bisect_right(page_starts, offset)


def chunk_text(text, page_starts, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """CHUNKING: slide a window of `size` chars across the text, stepping
    forward by (size - overlap) each time so windows overlap.

    We nudge each cut to the nearest space so we don't slice a word in half.
    """
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        # Don't cut mid-word: back up to the last space (unless we're at the end).
        if end < n:
            last_space = text.rfind(" ", start, end)
            if last_space > start:
                end = last_space
        chunk = text[start:end].strip()
        if chunk:  # skip empty windows
            chunks.append(
                {
                    "id": len(chunks),
                    "page": page_for_offset(start, page_starts),
                    "text": chunk,
                }
            )
        if end >= n:
            break
        start = end - overlap  # step forward, keeping `overlap` chars of context
    return chunks


def main():
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "data/pak-cons.pdf"

    print(f"Loading: {pdf_path}")
    full_text, page_starts = load_pdf(pdf_path)
    print(f"  pages extracted : {len(page_starts)}")
    print(f"  total characters: {len(full_text):,}")

    chunks = chunk_text(full_text, page_starts)
    print(f"\nChunking with size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
    print(f"  chunks produced : {len(chunks)}")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"  saved to        : {OUTPUT_FILE}")

    # Show a sample so the abstract idea becomes visible.
    sample = chunks[len(chunks) // 2]  # one from the middle
    print("\n--- sample chunk (middle of the document) ---")
    print(f"id={sample['id']}  page={sample['page']}  length={len(sample['text'])} chars")
    print(sample["text"][:400] + ("..." if len(sample["text"]) > 400 else ""))


if __name__ == "__main__":
    main()
