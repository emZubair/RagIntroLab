"""
Step 3 of the RAG pipeline: SIMILARITY SEARCH + RETRIEVAL.

What this does:
  1. Loads the vector store (store.npz).
  2. Embeds a user question with the SAME embedding model.
  3. Scores every chunk by cosine similarity and returns the top-k.

This file has NO language model in it -- it proves retrieval works on its own.

Run:  python retrieve.py "How is the President elected?"
"""

import sys

import numpy as np
import requests

import config

# Use the configured Ollama endpoint/model so the same code works locally
# (localhost) and in production (the Ollama accessory container).
OLLAMA_URL = f"{config.OLLAMA_BASE_URL}/api/embeddings"
EMBED_MODEL = config.EMBED_MODEL
STORE_FILE = "store.npz"
TOP_K = 4


def embed(text):
    resp = requests.post(OLLAMA_URL, json={"model": EMBED_MODEL, "prompt": text})
    resp.raise_for_status()
    return np.array(resp.json()["embedding"], dtype=np.float32)


def load_store():
    data = np.load(STORE_FILE, allow_pickle=True)
    return data["vectors"], data["texts"], data["pages"]


def retrieve(question, vectors, texts, pages, k=TOP_K):
    """Return the k chunks most similar to the question."""
    q = embed(question)
    q = q / np.linalg.norm(q)  # normalise the question vector too

    # Cosine similarity for ALL chunks at once: one matrix-vector multiply.
    # vectors is (625, 768), q is (768,), so scores is (625,).
    scores = vectors @ q

    # Indices of the top-k highest scores, sorted best-first.
    top_idx = np.argsort(scores)[::-1][:k]
    return [(int(i), float(scores[i]), int(pages[i]), str(texts[i])) for i in top_idx]


def main():
    if len(sys.argv) < 2:
        print('Usage: python retrieve.py "your question here"')
        sys.exit(1)
    question = sys.argv[1]

    vectors, texts, pages = load_store()
    results = retrieve(question, vectors, texts, pages)

    print(f"Question: {question}\n")
    print(f"Top {len(results)} matching chunks:\n" + "=" * 60)
    for rank, (idx, score, page, text) in enumerate(results, 1):
        print(f"\n#{rank}  similarity={score:.3f}  chunk_id={idx}  page={page}")
        print("-" * 60)
        print(text[:500] + ("..." if len(text) > 500 else ""))


if __name__ == "__main__":
    main()
