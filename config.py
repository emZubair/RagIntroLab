"""
Central configuration for RagIntroLab, loaded from environment variables (.env).

The generation step can run on a LOCAL model (Ollama / gemma4 — the default) or a
REMOTE provider (Anthropic Claude or OpenAI), chosen by these variables:

  LLM_PROVIDER = ollama | anthropic | openai    (default: ollama)
  LLM_MODEL    = the remote model name (optional; sensible default per provider)
  LLM_API_KEY  = the remote provider's API key

Rule: if LLM_PROVIDER names a remote provider AND LLM_API_KEY is set, generation
routes to that provider; otherwise it falls back to the local Ollama model.

Embeddings ALWAYS run locally (see embed_store.py / retrieve.py) — only generation
is switchable, so your document text never leaves your machine.
"""

import os

from dotenv import load_dotenv

# Load variables from a .env file in the project directory, if present.
load_dotenv()

# --- Remote provider settings ---
PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
REMOTE_MODEL = os.getenv("LLM_MODEL", "").strip()
API_KEY = os.getenv("LLM_API_KEY", "").strip()

# --- Local (Ollama) settings ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "gemma4")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

# --- Cost / abuse controls (mainly for a public deployment) ---
# Cap the answer length so a remote provider can't run up large output bills.
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "512"))
# Per-IP request limit for the web UI's /api/ask endpoint (Flask-Limiter syntax).
# Generous by default for local use; tighten in production (e.g. "10 per hour").
RATE_LIMIT = os.getenv("RATE_LIMIT", "240 per hour")

# Default remote model per provider, used when LLM_MODEL is not set.
_DEFAULT_REMOTE_MODEL = {
    "anthropic": "claude-opus-4-8",
    "openai": "gpt-4o-mini",
}


def active_provider():
    """Which provider will actually be used, given the env + whether a key is set."""
    if PROVIDER in ("anthropic", "openai") and API_KEY:
        return PROVIDER
    return "ollama"


def remote_model():
    """The remote model name to use (explicit LLM_MODEL, or a per-provider default)."""
    if REMOTE_MODEL:
        return REMOTE_MODEL
    return _DEFAULT_REMOTE_MODEL.get(PROVIDER, LOCAL_MODEL)


def backend_label():
    """Human-readable description of where generation will run, for logging."""
    provider = active_provider()
    if provider == "ollama":
        return f"local Ollama ({LOCAL_MODEL})"
    return f"{provider} ({remote_model()})"
