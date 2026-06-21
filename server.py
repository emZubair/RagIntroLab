"""
Web interface for RagIntroLab.

A small Flask server that:
  - serves the single-page HTML UI (static/index.html),
  - exposes POST /api/ask    -> runs the full RAG pipeline for a question,
  - exposes GET  /api/history -> returns previously asked questions,
  - persists every Q&A to history.json.

It reuses the exact same pipeline code as the command-line scripts:
  retrieve.py  -> load_store, retrieve
  ask.py       -> build_prompt, generate_stream

Run:  python server.py      (then open http://localhost:5000)
"""

import datetime
import json
import os

from flask import Flask, jsonify, request, send_from_directory
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.middleware.proxy_fix import ProxyFix

import config
from ask import build_prompt, generate_stream
from retrieve import load_store, retrieve

HISTORY_FILE = "history.json"
STORE_FILE = "store.npz"

app = Flask(__name__, static_folder="static", static_url_path="")

# Behind kamal-proxy/nginx, trust one layer of X-Forwarded-* headers so the
# rate limiter sees the real client IP (not the proxy's).
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)

# Per-IP throttling for the /api/ask endpoint — caps abuse (and remote token
# spend) on a public deployment. Default storage is in-memory (fine for a single
# gunicorn worker); see DEPLOY.md to harden for multiple workers.
limiter = Limiter(get_remote_address, app=app)

# --- Load the vector store ONCE at startup (not on every request). ---
# This is the whole point of the offline indexing phase: querying is cheap.
if not os.path.exists(STORE_FILE):
    raise SystemExit(
        f"'{STORE_FILE}' not found. Build the index first:\n"
        f"  python ingest.py <your.pdf> && python embed_store.py\n"
        f"  (or: make index PDF=<your.pdf>)"
    )
VECTORS, TEXTS, PAGES = load_store()
print(f"Loaded vector store: {VECTORS.shape[0]} chunks, dim {VECTORS.shape[1]}")
print(f"Generation backend: {config.backend_label()}")


def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE) as f:
            return json.load(f)
    return []


def save_history_entry(entry):
    history = load_history()
    history.append(entry)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def answer_question(question):
    """Run the full RAG pipeline and return a result dict."""
    # 1. RETRIEVE
    results = retrieve(question, VECTORS, TEXTS, PAGES)
    pages = sorted({page for _, _, page, _ in results})
    # 2. AUGMENT
    prompt = build_prompt(question, results)
    # 3. GENERATE (collect the streamed tokens into one string)
    answer = "".join(generate_stream(prompt)).strip()
    return {"answer": answer, "pages": pages}


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/ask", methods=["POST"])
@limiter.limit(config.RATE_LIMIT)
def api_ask():
    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Empty question."}), 400

    result = answer_question(question)

    entry = {
        "id": int(datetime.datetime.now().timestamp() * 1000),
        "question": question,
        "answer": result["answer"],
        "pages": result["pages"],
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    save_history_entry(entry)
    return jsonify(entry)


@app.route("/api/history", methods=["GET"])
def api_history():
    # Newest first.
    return jsonify(list(reversed(load_history())))


if __name__ == "__main__":
    # threaded=True so the UI can fetch history while an answer is generating.
    port = int(os.environ.get("PORT", 5000))
    print(f"Open http://localhost:{port}")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
