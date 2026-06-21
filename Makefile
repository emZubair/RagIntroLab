# RagIntroLab — Makefile
#
# Common tasks for building and querying the local RAG system.
# The most important target is `reindex`, which swaps in a NEW source document:
#
#     make reindex PDF=/path/to/another.pdf
#
# PYTHON points at the project's interpreter (the pyenv "v314" virtualenv).
# Override it if you use a different environment, e.g.:
#     make index PYTHON=.venv/bin/python PDF=mydoc.pdf

PYTHON ?= /Users/zubair/.pyenv/versions/v314/bin/python
PDF    ?= data/pak-cons.pdf
Q      ?= How is the President of Pakistan elected?
PORT   ?= 5000

# Generated artifacts (the index).
CHUNKS := chunks.json
STORE  := store.npz

.DEFAULT_GOAL := help

.PHONY: help deps hooks lint ingest embed index reindex ask retrieve serve serve-prod clean

help:                ## Show this help
	@echo "RagIntroLab — available targets:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Variables you can override:"
	@echo "  PDF=<path>     source document to index   (current: $(PDF))"
	@echo "  Q=\"<text>\"     question to ask"
	@echo "  PYTHON=<path>  python interpreter          (current: $(PYTHON))"
	@echo ""
	@echo "Change the knowledge source:  make reindex PDF=/path/to/new.pdf"

deps:                ## Install Python dependencies (from requirements.txt)
	$(PYTHON) -m pip install --quiet -r requirements.txt
	@echo "Dependencies installed."

hooks:               ## Install dev deps + git pre-commit/pre-push hooks
	$(PYTHON) -m pip install --quiet -r requirements-dev.txt
	$(PYTHON) -m pre_commit install --install-hook-types pre-commit,pre-push
	@echo "Git hooks installed (pre-commit, pre-push)."

lint:                ## Run all pre-commit hooks against every file
	$(PYTHON) -m pre_commit run --all-files

ingest:              ## Load + chunk the PDF -> chunks.json   (PDF=<path>)
	$(PYTHON) ingest.py "$(PDF)"

embed:               ## Embed chunks -> store.npz (needs chunks.json first)
	$(PYTHON) embed_store.py

# Build the full index from scratch. `embed` depends on `ingest` having run.
index: ingest embed  ## Full index build: ingest + embed   (PDF=<path>)
	@echo "Index built from: $(PDF)"

# Swap in a NEW source document: wipe the old index, then rebuild from PDF.
reindex: clean index ## Replace the source document and rebuild   (PDF=<path>)
	@echo "Source switched. Now querying answers from: $(PDF)"

ask:                 ## Ask a question end-to-end   (Q="...")
	$(PYTHON) ask.py "$(Q)"

retrieve:            ## Show retrieved chunks only, no LLM   (Q="...")
	$(PYTHON) retrieve.py "$(Q)"

serve:               ## Start the web UI dev server (default http://localhost:5000; override PORT=)
	PORT=$(PORT) $(PYTHON) server.py

serve-prod:          ## Start the web UI under gunicorn (production server, like the Docker image)
	$(PYTHON) -m gunicorn --bind 0.0.0.0:8000 --workers 1 --threads 4 --timeout 120 server:app

clean:               ## Delete the generated index (chunks.json, store.npz)
	@rm -f $(CHUNKS) $(STORE)
	@echo "Removed $(CHUNKS) and $(STORE)."
