"""
Step 2 of the RAG pipeline: EMBEDDING + STORING (building the vector store).

What this does:
  1. Loads the chunks from chunks.json.
  2. Sends each chunk to the local embedding model (via Ollama) to get a vector.
  3. Stacks all vectors into one matrix, normalises them, and saves to disk.

The saved store.npz IS our "vector database".

Run:  python embed_store.py
"""

import json

import numpy as np
import requests

import config

OLLAMA_URL = f"{config.OLLAMA_BASE_URL}/api/embeddings"
EMBED_MODEL = config.EMBED_MODEL
CHUNKS_FILE = "chunks.json"
STORE_FILE = "store.npz"


def embed(text):
    """Ask Ollama to turn one piece of text into a vector (a list of floats)."""
    resp = requests.post(OLLAMA_URL, json={"model": EMBED_MODEL, "prompt": text})
    resp.raise_for_status()
    return resp.json()["embedding"]


def main():
    with open(CHUNKS_FILE) as f:
        chunks = json.load(f)
    print(f"Embedding {len(chunks)} chunks with '{EMBED_MODEL}' ...")

    vectors = []
    for i, chunk in enumerate(chunks):
        chunk_text = chunk["text"]
        embed_text = embed(chunk_text)
        print(f"Chunk {i}: {chunk_text} vector: {embed_text}")
        vectors.append(embed_text)
        # Simple progress indicator (overwrites the same line).
        if (i + 1) % 25 == 0 or i + 1 == len(chunks):
            print(f"\r  {i + 1}/{len(chunks)} embedded", end="", flush=True)
    print()

    # Shape = (n_chunks, embedding_dim). For nomic-embed-text, dim = 768.
    matrix = np.array(vectors, dtype=np.float32)
    print(f"Vector matrix shape: {matrix.shape}  (chunks x dimensions)")

    # NORMALISE: scale each row to length 1. After this, the cosine similarity
    # between any two vectors equals their dot product -> retrieval is trivial.
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / norms

    # Save the matrix together with the chunk metadata (text + page) so the
    # retrieve step has everything it needs in one file.
    np.savez(
        STORE_FILE,
        vectors=matrix,
        texts=np.array([c["text"] for c in chunks], dtype=object),
        pages=np.array([c["page"] for c in chunks]),
    )
    print(f"Saved vector store -> {STORE_FILE}")
    print("\nWhat one vector looks like (first 8 of 768 numbers):")
    print(matrix[0][:8])


if __name__ == "__main__":
    main()
