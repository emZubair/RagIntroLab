"""
Provider-agnostic text generation for RagIntroLab.

`generate_stream(prompt)` yields the answer token-by-token from whichever backend
config.active_provider() selects:

  - "ollama"     -> local gemma4 via the Ollama HTTP API (default, no API key)
  - "anthropic"  -> Claude via the official `anthropic` SDK
  - "openai"     -> GPT/Codex via the official `openai` SDK

The remote SDKs are imported lazily, so a local-only install doesn't need them.
Install only what you use:  pip install anthropic   /   pip install openai
"""

import json

import requests

import config


def _ollama_stream(prompt):
    """LOCAL: stream tokens from gemma4 via Ollama's /api/generate."""
    resp = requests.post(
        f"{config.OLLAMA_BASE_URL}/api/generate",
        json={"model": config.LOCAL_MODEL, "prompt": prompt, "stream": True},
        stream=True,
    )
    resp.raise_for_status()
    for line in resp.iter_lines():
        if not line:
            continue
        data = json.loads(line)
        if "response" in data:
            yield data["response"]
        if data.get("done"):
            break


def _anthropic_stream(prompt):
    """REMOTE: stream tokens from Claude via the official Anthropic SDK."""
    import anthropic  # lazy import — only needed when this provider is active

    client = anthropic.Anthropic(api_key=config.API_KEY)
    with client.messages.stream(
        model=config.remote_model(),
        max_tokens=config.MAX_OUTPUT_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        yield from stream.text_stream


def _openai_stream(prompt):
    """REMOTE: stream tokens from OpenAI via the official OpenAI SDK."""
    from openai import OpenAI  # lazy import

    client = OpenAI(api_key=config.API_KEY)
    stream = client.chat.completions.create(
        model=config.remote_model(),
        messages=[{"role": "user", "content": prompt}],
        max_tokens=config.MAX_OUTPUT_TOKENS,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def generate_stream(prompt):
    """Route to the active provider and stream its answer."""
    provider = config.active_provider()
    if provider == "anthropic":
        yield from _anthropic_stream(prompt)
    elif provider == "openai":
        yield from _openai_stream(prompt)
    else:
        yield from _ollama_stream(prompt)
