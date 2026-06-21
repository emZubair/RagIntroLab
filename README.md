# RagIntroLab

A **local-first Retrieval-Augmented Generation (RAG) system** that answers natural-language
questions about your own documents — grounded in the source text, with **page-level citations**
on every answer.

Out of the box, all components run on your own machine via [Ollama](https://ollama.com): no
third-party APIs, no API keys, and **no data leaves your environment** — making it suitable for
confidential or regulated material. When you want stronger reasoning, generation can optionally be
pointed at a **remote model (Claude or OpenAI)** via a single `.env` setting, while embeddings and
retrieval stay local — see [Switching the generation model](#switching-the-generation-model-local--remote).
The pipeline is built from transparent, dependency-light Python, so its behaviour is fully
inspectable and straightforward to adapt to a specific domain or document set.

### What it does

- **Ingests** a document (PDF), splits it into passages, and indexes them as semantic vectors.
- **Answers** questions using only the retrieved passages, and **cites the page(s)** each answer
  draws from — so every response is verifiable against the source.
- **Refuses to fabricate** — when the source doesn't contain the answer, it says so rather than
  guessing.
- Ships with both a **command-line interface** and a **browser UI** (with question history).

### Why this approach

- **Data privacy** — local by default; nothing is sent to external services unless you explicitly
  opt into a remote generation provider (and even then, embeddings/retrieval stay on-machine).
- **Grounded & verifiable** — page citations make every answer auditable against the source.
- **No vendor lock-in** — swap the document, the embedding model, or the LLM (local or remote)
  independently, with no code changes.
- **Transparent & extensible** — no heavyweight framework hiding the retrieval logic, so behaviour
  is easy to tune and the architecture is easy to scale (see the
  [production roadmap](#13-scope-and-production-roadmap)).

---

## Quick start

Get up and running in about five minutes. (Sections [5–7](#5-prerequisites) cover each step in
more detail.)

**What you need first:**

- **[Ollama](https://ollama.com)** installed and running (it provides the local models).
- **Python 3.10+**.
- A few hundred MB of disk for the embedding model, ~10 GB for the generator model.

**Steps:**

```bash
# 1. Pull the two models Ollama will serve locally
ollama pull gemma4            # the generator (writes answers, ~9.6 GB)
ollama pull nomic-embed-text  # the embedder (turns text into vectors, ~274 MB)
ollama list                   # confirm both are listed

# 2. Get the project and enter it
git clone git@github.com:emZubair/RagIntroLab.git
cd RagIntroLab                     # the project folder (contains this README)

# 3. Create/activate a Python environment and install dependencies
python -m venv .venv && source .venv/bin/activate   # or your own env
pip install -r requirements.txt
#   (shortcut once a Makefile env is set up:  make deps)

# 4. Build the index from a document (one-time; re-run only when the doc changes)
make index PDF=/path/to/your.pdf
#   or:  python ingest.py /path/to/your.pdf && python embed_store.py

# 5a. Ask from the command line
make ask Q="What is this document about?"
#   or:  python ask.py "What is this document about?"

# 5b. ...or use the web UI (ask box + history) in your browser
make serve                    # then open http://localhost:5000
#   or:  python server.py
```

That's it. Step 4 turns your PDF into a searchable vector store; steps 5a/5b answer questions
grounded in it, with page citations. To switch to a different document later, see
[Updating the source knowledge file](#updating-the-source-knowledge-file).

> **Not using Make?** Every `make` target maps to a plain Python command (shown beside it above and
> in [section 7](#7-how-to-run-it)). The Makefile is just a convenience wrapper.

---

## Table of contents

0. [Quick start](#quick-start)

1. [What is RAG and why it exists](#1-what-is-rag-and-why-it-exists)
2. [The two phases](#2-the-two-phases)
3. [Architecture](#3-architecture)
4. [Key concepts glossary](#4-key-concepts-glossary)
5. [Prerequisites](#5-prerequisites)
6. [Setup](#6-setup)
7. [How to run it](#7-how-to-run-it)
8. [The pipeline, file by file](#8-the-pipeline-file-by-file)
9. [What a full query looks like](#9-what-a-full-query-looks-like)
10. [Configuration knobs](#10-configuration-knobs)
11. [Tuning the system](#11-tuning-the-system)
12. [Troubleshooting](#12-troubleshooting)
13. [Scope and production roadmap](#13-scope-and-production-roadmap)

---

## 1. What is RAG and why it exists

A large language model (LLM) like Gemma is a *frozen brain*. It only knows:

- What it saw during training — it has a **knowledge cutoff** and doesn't know your private
  documents, recent events, or niche material.
- When it doesn't know something, it will often **hallucinate** — produce a confident,
  wrong answer.

You *could* fine-tune or retrain the model on your data, but that is expensive, slow, and must be
redone every time your data changes.

**RAG (Retrieval-Augmented Generation)** avoids all of that. Instead of baking knowledge into the
model, you keep your knowledge in an external store. At question time you **retrieve the relevant
passages and paste them into the prompt**. The model then answers from the text you handed it —
like an **open-book exam**.

> Core idea: **Don't make the model memorize. Make it read.**

Benefits:

- **Up-to-date / private knowledge** without retraining.
- **Reduced hallucination** — the model is instructed to answer only from supplied context.
- **Traceability** — answers can cite the exact source (here, page numbers) so you can verify them.

---

## 2. The two phases

RAG splits cleanly into two phases. Internalize this — everything hangs off it.

### Phase A — Indexing (done once, offline)

```
Documents → Split into chunks → Embed each chunk → Store vectors in a DB
```

### Phase B — Querying (every time a user asks)

```
Question → Embed it → Find nearest chunks → Stuff them into a prompt → LLM answers
```

In this project, **Phase A** is `ingest.py` + `embed_store.py`, and **Phase B** is
`retrieve.py` + `ask.py`.

---

## 3. Architecture

```
                          INDEXING  (run once)
   ┌─────────────────────────────────────────────────────────────────┐
   │                                                                 │
   │   pak-cons.pdf ──ingest.py──► chunks.json ──embed_store.py──►   │
   │   (PDF, 222 pp)   load+chunk   (625 chunks)   embed (768-d)     │
   │                                                  │              │
   │                                                  ▼              │
   │                                            store.npz            │
   │                                     (625 × 768 vector matrix    │
   │                                      + chunk text + page nums)  │
   └─────────────────────────────────────────────────────────────────┘

                          QUERYING  (every question)
   ┌─────────────────────────────────────────────────────────────────┐
   │                                                                 │
   │   "How is the President elected?"                               │
   │            │                                                    │
   │            ▼  retrieve.py                                       │
   │   embed question ──► dot-product vs store.npz ──► top-4 chunks  │
   │            │                                                    │
   │            ▼  ask.py                                            │
   │   build prompt:  instructions + 4 chunks + question             │
   │            │                                                    │
   │            ▼                                                    │
   │   gemma4 (LLM) ──► grounded answer + page citations             │
   └─────────────────────────────────────────────────────────────────┘
```

**Two models, two jobs** — a defining feature of almost every RAG system:

| Model | Role | Endpoint | Output |
|-------|------|----------|--------|
| `nomic-embed-text` | **Embedder** — turns text into meaning-vectors | `/api/embeddings` | 768 floats |
| `gemma4` | **Generator** — writes the answer | `/api/generate` | text |

Both are served locally by Ollama at `http://localhost:11434`.

---

## 4. Key concepts glossary

- **Document** — the raw knowledge source (here, a PDF).
- **Chunking** — splitting a long document into small, topically-focused windows (~900 characters
  here). Small chunks make retrieval precise; a whole document has one blurry "average" meaning.
- **Overlap** — neighbouring chunks share some text (~150 characters) so an idea that straddles a
  chunk boundary survives intact in at least one chunk.
- **Embedding** — a list of numbers (a **vector**, 768 of them here) that captures the *meaning* of
  a piece of text. Texts with similar meaning produce vectors that are close together. This enables
  **semantic search** (matching by meaning, not keywords).
- **Vector store** — a structure that holds all the chunk vectors and can quickly find the ones
  closest to a query vector. Here it's just a normalized NumPy matrix saved as `store.npz`.
- **Cosine similarity** — the measure of "closeness" between two vectors. Because we **normalize**
  every vector to length 1, cosine similarity becomes a simple **dot product**.
- **Normalization** — scaling each vector to length 1, so similarity = dot product (one fast matrix
  multiply for all chunks at once).
- **Retrieval** — fetching the **top-k** chunks most similar to the question.
- **Augmentation** — building the final prompt: `instructions + retrieved context + question`.
- **Grounding** — instructing the LLM to answer **only** from the provided context and to say so
  when the answer isn't there. This is what suppresses hallucination.
- **Generation** — the LLM reading the augmented prompt and producing the answer.

---

## 5. Prerequisites

- **macOS / Linux** with a terminal.
- **[Ollama](https://ollama.com)** installed and running.
- The two models pulled:
  ```bash
  ollama pull gemma4            # generator (~9.6 GB)
  ollama pull nomic-embed-text  # embedder (~274 MB)
  ```
  Verify with `ollama list`.
- **Python 3.10+** (developed on Python 3.14).
- Python packages: `pypdf`, `numpy`, `requests`, `cryptography` (required because the sample PDF is
  encrypted), `python-dotenv` (config loading), and `flask` (for the optional web UI). All are in
  `requirements.txt`. Remote providers (`anthropic` / `openai`) are optional — see
  [Switching the generation model](#switching-the-generation-model-local--remote).

---

## 6. Setup

```bash
# 1. Create and activate a Python virtual environment:
python -m venv .venv && source .venv/bin/activate
#    (any environment manager works — e.g. a pyenv virtualenv:
#     eval "$(pyenv init -)"; eval "$(pyenv virtualenv-init -)"; pyenv activate <name>)

# 2. Install dependencies:
pip install -r requirements.txt   # or, explicitly: pip install pypdf numpy requests cryptography flask python-dotenv

# 3. Make sure Ollama is running and the models are present:
ollama list   # should show gemma4 and nomic-embed-text
```

### Development tooling (optional)

Contributors should install the dev dependencies and the git hooks. The hooks run **ruff** (lint +
format) automatically, matching what CI checks (`.gitlab-ci.yml`):

```bash
pip install -r requirements-dev.txt   # ruff + pre-commit
pre-commit install --install-hook-types pre-commit,pre-push
#   or simply:  make hooks
```

- **pre-commit**: lints and formats the files you're about to commit (autofixes where possible).
- **pre-push**: lints + format-checks the whole repo, so nothing slips through the per-commit net.

Run them manually any time with `make lint` (or `pre-commit run --all-files`).

---

## 7. How to run it

Run the four steps in order. Steps 1–2 (indexing) only need to be re-run when the PDF or chunking
settings change. Steps 3–4 can be run as often as you like.

```bash
# Step 1 — load the PDF and split it into chunks
python ingest.py data/pak-cons.pdf
#   -> writes chunks.json   (625 chunks)

# Step 2 — embed every chunk and build the vector store
python embed_store.py
#   -> writes store.npz      (625 × 768 matrix)

# Step 3 — (optional) inspect retrieval on its own, no LLM
python retrieve.py "How is the President of Pakistan elected?"
#   -> prints the top-4 matching chunks with similarity scores and pages

# Step 4 — ask a question end-to-end (the full RAG)
python ask.py "How is the President of Pakistan elected?"
#   -> prints a grounded answer with page citations
```

Example output of Step 4:

```
Retrieved 4 chunks from pages [31, 33, 192, 195]

Answer:
------------------------------------------------------------
The President shall be elected by an electoral college consisting of the
members of both Houses and the members of the Provincial Assemblies, in
accordance with the provisions of the Second Schedule.

(Page 31)
------------------------------------------------------------
```

Ask something not in the document and grounding kicks in:

```
$ python ask.py "What is the best pizza topping according to the constitution?"
Answer:
------------------------------------------------------------
I couldn't find that in the provided text.
------------------------------------------------------------
```

### Shortcuts via Make

A `Makefile` wraps these commands so you don't have to remember them. Run `make help` to see all
targets. The most useful ones:

```bash
make deps                              # install Python dependencies
make index   PDF=/path/to/doc.pdf      # ingest + embed (full index build)
make ask     Q="How is the President elected?"
make retrieve Q="presidential election"   # show retrieved chunks only, no LLM
make clean                             # delete the index (chunks.json, store.npz)
```

`PDF`, `Q`, and `PYTHON` all have defaults you can override on the command line (see `make help`).

### Updating the source knowledge file

To point RagIntroLab at a **different document** — this is the whole reason RAG is flexible: the
models never change, only the indexed knowledge does — use the **`reindex`** target:

```bash
make reindex PDF=/path/to/new_document.pdf
```

`reindex` does three things in order:

1. **`clean`** — deletes the old `chunks.json` and `store.npz` so no stale data lingers.
2. **`ingest`** — loads and chunks the new PDF.
3. **`embed`** — embeds the new chunks into a fresh `store.npz`.

After it finishes, every `make ask Q="..."` answers from the **new** document. Nothing else needs
to change — same code, same models, new knowledge.

> **Doing it manually** (without Make) is the same three steps:
> ```bash
> rm -f chunks.json store.npz                 # 1. clear the old index
> python ingest.py /path/to/new_document.pdf  # 2. load + chunk
> python embed_store.py                       # 3. embed -> store.npz
> ```
>
> **Important:** always re-run **both** `ingest.py` *and* `embed_store.py` together. The vector
> store (`store.npz`) is built from `chunks.json`; if you re-chunk without re-embedding (or vice
> versa) the two fall out of sync and retrieval breaks. The `reindex` target enforces the correct
> order for you.
>
> The new file needs a real text layer (not a scanned image). For scanned PDFs you'd OCR them
> first — see [Troubleshooting](#12-troubleshooting).

### Web interface (HTML UI)

Besides the command line, RagIntroLab ships with a small browser UI where you can ask questions and
see your question **history**. It's served by a tiny Flask app (`server.py`) that reuses the exact
same pipeline code as the CLI.

```bash
make serve                 # starts the server on http://localhost:5000
#   or:  python server.py
#   or:  make serve PORT=8080
```

Then open **http://localhost:5000** in your browser. You get:

- an **ask box** (type a question, click *Ask* or press Cmd/Ctrl+Enter),
- an **answer panel** showing the grounded answer and the **page(s)** it was sourced from,
- a **history sidebar** listing every past question (newest first) — click any item to re-display
  its stored answer.

The index must be built first (`make index PDF=...`); the server loads `store.npz` once at startup.
History is persisted to **`history.json`** in the project folder, so it survives restarts. Delete
that file to clear the history.

How the pieces talk:

```
browser (static/index.html)
   │  POST /api/ask {"question": "..."}        GET /api/history
   ▼
server.py (Flask)
   │  retrieve() ─► build_prompt() ─► generate_stream()   (same code as ask.py)
   ▼
Ollama: nomic-embed-text (embed)  +  gemma4 (generate)
   │
   └─► answer + pages  ──►  appended to history.json  ──►  returned as JSON
```

### Switching the generation model (local ↔ remote)

By default, answers are generated **locally** by gemma4 via Ollama — no API key, nothing
leaves your machine. You can instead route **generation** to a remote provider (Claude or
OpenAI) without touching any code, by creating a `.env` file:

```bash
cp .env.example .env
```

Then set these variables in `.env`:

| Variable | Purpose |
|----------|---------|
| `LLM_PROVIDER` | `ollama` (default) · `anthropic` · `openai` |
| `LLM_MODEL` | Remote model name (default: `claude-opus-4-8` for anthropic, `gpt-4o` for openai) |
| `LLM_API_KEY` | The remote provider's API key |

The rule: **if `LLM_PROVIDER` names a remote provider *and* `LLM_API_KEY` is set, generation
uses that provider; otherwise it falls back to local gemma4.** Example — answer with Claude:

```bash
# .env
LLM_PROVIDER=anthropic
LLM_API_KEY=sk-ant-...
```

Install the matching SDK for the provider you choose (they're imported lazily, so a local-only
setup needs neither):

```bash
pip install anthropic   # for LLM_PROVIDER=anthropic
pip install openai      # for LLM_PROVIDER=openai
```

Every entry point prints which backend is active (e.g. `Generating with: anthropic
(claude-opus-4-8)`), so you always know where a given answer came from.

> **Important:** only the **generation** step switches. **Embeddings always run locally**
> (`nomic-embed-text` via Ollama), so your document text is never sent to the remote provider —
> only the already-retrieved passages plus the question are. This is the hybrid setup: local,
> private retrieval + a stronger remote model for composing the answer. For a public-document
> demo this is ideal; for confidential corpora, weigh that the retrieved passages do leave your
> machine when a remote provider is used (see [Scope and production roadmap](#13-scope-and-production-roadmap)).

---

## 8. The pipeline, file by file

| File | Phase | Concept | Reads | Writes |
|------|-------|---------|-------|--------|
| `ingest.py` | Indexing | Loading + **chunking** | the PDF | `chunks.json` |
| `embed_store.py` | Indexing | **Embeddings** + **vector store** | `chunks.json` | `store.npz` |
| `retrieve.py` | Querying | **Cosine similarity** + **retrieval** | `store.npz` | (prints) |
| `ask.py` | Querying | **Augmentation** + **generation** | `store.npz` | (prints) |
| `server.py` | Querying | **Web UI** (Flask) — reuses the above | `store.npz` | `history.json` |
| `static/index.html` | Querying | Browser front-end (ask box + history) | — | — |
| `config.py` | Both | Loads `.env`; selects local vs remote generation backend | `.env` | — |
| `llm.py` | Querying | Provider-agnostic `generate_stream` (Ollama / Claude / OpenAI) | — | — |

### `ingest.py` — Loading + Chunking

- Uses `pypdf` to extract text from each page, recording where each page starts so chunks can be
  traced back to a page number.
- Slides a window of `CHUNK_SIZE` characters across the text, stepping forward by
  `CHUNK_SIZE - CHUNK_OVERLAP` each time so consecutive chunks overlap.
- Nudges each cut to the nearest space so words aren't sliced in half.
- Saves a list of `{id, page, text}` objects to `chunks.json`.

### `embed_store.py` — Embeddings + Vector store

- Sends each chunk's text to Ollama's `/api/embeddings` with the `nomic-embed-text` model and gets
  back a 768-number vector.
- Stacks all vectors into a `(625, 768)` NumPy matrix.
- **Normalizes** every row to length 1 (so later, cosine similarity = dot product).
- Saves the matrix plus the chunk texts and page numbers into a single `store.npz`.

### `retrieve.py` — Similarity search + Retrieval

- Embeds the question with the **same** model (so it lands in the same meaning-space as the chunks).
- Normalizes the question vector, then computes `store_matrix @ question_vector` — one matrix-vector
  multiply giving a similarity score for **all 625 chunks at once**.
- Returns the top-k (default 4) highest-scoring chunks with their scores and pages.
- Contains **no LLM** — it proves retrieval works on its own.

### `ask.py` — Augmentation + Generation (the full system)

- Reuses `load_store` and `retrieve` from `retrieve.py` (no duplicated logic).
- **Augments**: inserts the retrieved chunks (each tagged `[page N]`) into a prompt template that
  instructs the model to answer only from the context and to cite pages.
- **Generates**: streams the answer token-by-token from `gemma4` via `/api/generate`.

### Design note: the vector store (NumPy now, a vector DB later)

The vector store here is a single normalized `(N × 768)` NumPy matrix in `store.npz`, and retrieval
is **exact brute-force search** — one matrix–vector multiply (`store_matrix @ query`) scores every
chunk, then `argsort` takes the top-k. This is a **deliberate choice**, not a shortcut, and it's
worth being explicit about why — and where it stops being the right call.

**Why it's efficient at this scale.** For a corpus of hundreds to low-thousands of chunks (this
PDF is 625), a `625 × 768` dot product is a single BLAS call that runs in well under a millisecond.
Brute force here is **exact** — it returns the true nearest neighbours with no recall loss — needs
**zero extra infrastructure**, and keeps the whole system inspectable. At this size a dedicated
vector database would add operational weight without measurably improving latency or quality.

**Where it stops scaling.** The approach has three known limits, each with a clear trigger:

- **Latency is O(N).** Every query scans the entire matrix. Fine at thousands; at millions of
  vectors a linear scan becomes the bottleneck.
- **It's all in RAM.** The full matrix is loaded into memory at startup. `N × 768 × 4 bytes` is
  under 2 MB here, but ~3 GB at one million vectors — eventually it won't fit.
- **No persistence layer, filtering, or concurrency.** There's no incremental update (re-embedding
  rebuilds the whole file), no metadata filtering (e.g. "only this document / date range"), and no
  concurrent-writer story.

**The concept that matters:** beyond a certain size you trade **exact** search for **approximate
nearest neighbour (ANN)** search — index structures (IVF, HNSW, product quantization) that find the
*almost*-nearest vectors far faster, accepting a small, tunable recall loss for a large speed gain.
That recall/speed/memory trade-off is the core decision in any production vector store.

**Alternatives, and when each earns its place:**

| Option | What it adds | Reach for it when… |
|--------|--------------|--------------------|
| **NumPy brute force** (this project) | Exact, zero infra, fully transparent | ≲ 10K–50K vectors; simplicity and exactness matter |
| **[FAISS](https://github.com/facebookresearch/faiss)** | In-process ANN indexes (IVF/HNSW/PQ), GPU support | Millions of vectors but you still want a library, not a server |
| **[Chroma](https://www.trychroma.com/)** | Embedded vector DB: persistence + metadata filtering, minimal setup | You want a real store without running a separate service |
| **[Qdrant](https://qdrant.tech/) / [Weaviate](https://weaviate.io/) / [Milvus](https://milvus.io/)** | Standalone vector DB servers: ANN at scale, filtering, hybrid search, sharding, replication | Large corpora, multi-tenant, production SLAs |
| **[pgvector](https://github.com/pgvector/pgvector)** | Vector search inside PostgreSQL — SQL filters + transactions alongside your data | You already run Postgres and want one system of record |
| **Managed (Pinecone, etc.)** | Fully hosted ANN, no ops | You'd rather pay to not operate the index |

Because the store is touched only through `embed_store.py` (write) and `retrieve.py`'s `load_store`
/ `retrieve` (read), swapping in any of these is a **contained change** behind a small interface —
the chunking, prompting, and generation code is untouched. See
[Scope and production roadmap](#13-scope-and-production-roadmap).

---

## 9. What a full query looks like

```
User: "How is the President elected?"
   │
   ├─▶ retrieve.py: embed question  ──►  [0.12, -0.4, ...]   (768-d vector)
   │
   ├─▶ retrieve.py: store_matrix @ q ──►  625 scores ──► top 4 chunks (pages 31, 33, 192, 195)
   │
   ├─▶ ask.py: build prompt
   │       "Use ONLY the context below... cite page numbers.
   │        Context:
   │          [page 31] ...There shall be a President of Pakistan...
   │          [page 33] ...
   │          [page 192] ...
   │          [page 195] ...
   │        Question: How is the President elected?
   │        Answer:"
   │
   └─▶ gemma4 ──► "The President shall be elected by an electoral college... (Page 31)"
```

The model never "knew" the constitution. The knowledge was handed to it at the last second, and it
was told to use nothing else.

> **Subtlety worth remembering:** similarity search *always* returns its top-k chunks, even for a
> nonsense question — it has no concept of "no good match." It is the **generator**, following the
> grounding instruction, that recognizes the chunks don't answer the question and refuses. The
> retriever *finds*; the generator *judges*.

---

## 10. Configuration knobs

| Setting | File | Default | Effect |
|---------|------|---------|--------|
| `CHUNK_SIZE` | `ingest.py` | 900 | Max characters per chunk. Bigger = more context per chunk, fuzzier retrieval. |
| `CHUNK_OVERLAP` | `ingest.py` | 150 | Characters shared between neighbours. Bigger = fewer ideas split across boundaries, more redundancy. |
| `EMBED_MODEL` | `embed_store.py`, `retrieve.py` | `nomic-embed-text` | The embedding model. **Must be identical** in both files. |
| `TOP_K` | `retrieve.py` | 4 | How many chunks to retrieve. Fewer = thinner context; more = noisier. |
| `PROMPT_TEMPLATE` | `ask.py` | — | The grounding instructions. Edit to change behaviour/tone. |
| `LLM_PROVIDER` | `.env` | `ollama` | Generation backend: `ollama` (local gemma4), `anthropic`, or `openai`. See [Switching the generation model](#switching-the-generation-model-local--remote). |
| `LLM_MODEL` | `.env` | per-provider | Remote model name when a remote provider is selected. |
| `LLM_API_KEY` | `.env` | — | Remote provider API key. Generation stays local if unset. |
| `LOCAL_MODEL` | `.env` | `gemma4` | The local Ollama generation model. |

> If you change `CHUNK_SIZE` or `CHUNK_OVERLAP`, re-run **both** `ingest.py` and `embed_store.py`.
> If you change the embedding model, re-run `embed_store.py` (and keep `retrieve.py` in sync).

---

## 11. Tuning the system

The pipeline exposes a few levers for adapting retrieval and answer quality to a given document set:

1. **`TOP_K`** (`retrieve.py`): how many passages feed the answer. Lower is more focused; higher
   gives broader context at the cost of added noise.
2. **Chunk size / overlap** (`ingest.py`, then re-index): controls how finely the source is split,
   trading retrieval precision against context completeness.
3. **Similarity scores** (`retrieve.py`): inspect the per-passage scores to gauge retrieval
   confidence — relevant queries score high, off-topic ones score low, which is useful for setting
   relevance thresholds.
4. **Source document** (`ingest.py`, then re-index): point the pipeline at any document; the same
   code answers about the new source with no other changes.
5. **Prompt template** (`ask.py`): adjust tone, verbosity, or constraints (e.g. require verbatim
   quotation of the source clause).

---

## 12. Troubleshooting

- **`cryptography>=3.1 is required for AES algorithm`** — the PDF is encrypted; install
  `cryptography` (`pip install cryptography`). pypdf then decrypts it transparently.
- **`requests.exceptions.ConnectionError` to `localhost:11434`** — Ollama isn't running. Start it
  (`ollama serve` or the Ollama app) and confirm with `curl http://localhost:11434/api/tags`.
- **`model not found`** — pull the model: `ollama pull gemma4` / `ollama pull nomic-embed-text`.
- **Empty or garbled chunks** — the PDF may be scanned images (no text layer). You'd need OCR
  (e.g. Tesseract) to extract text first. This project assumes a real text layer.
- **`store.npz` not found when running `retrieve.py`/`ask.py`** — run `ingest.py` then
  `embed_store.py` first.

---

## 13. Scope and production roadmap

The current implementation is intentionally lean and transparent so it can be adapted quickly to a
specific domain or document set. Scaling it to a large production deployment follows a
well-understood path; because each stage (**ingest → embed → retrieve → generate**) is decoupled,
these upgrades can be made independently without rewriting the system:

- **Vector store** — the in-process NumPy index is ideal for thousands of passages. For large
  corpora, swap in a dedicated vector database (FAISS, Qdrant, Chroma, pgvector) for fast
  approximate search and persistence. The `embed_store.py` / `retrieve.py` interface is small and
  isolated, so this is a contained change.
- **Structure-aware chunking** — upgrade fixed-size windows to split on document structure
  (sections, articles, headings) to further sharpen retrieval.
- **Hybrid retrieval & re-ranking** — combine semantic and keyword search, then re-rank the top
  candidates for higher precision on large or heterogeneous collections.
- **Conversational context** — extend the single-turn API to carry multi-turn history.
- **Evaluation & monitoring** — add retrieval and answer-quality metrics to measure and track
  accuracy over time.
- **Multi-format ingestion** — add loaders for DOCX, HTML, and scanned PDFs (via OCR).

---

*RagIntroLab — a local, source-grounded RAG system with verifiable citations.*
