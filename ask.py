"""
Step 4 of the RAG pipeline: AUGMENTATION + GENERATION (the full system).

What this does:
  1. Retrieves the top-k chunks for a question (reuses retrieve.py).
  2. Builds an "augmented" prompt: instructions + context + question.
  3. Streams the answer from the active generation backend (local or remote).

The generation backend (local gemma4 vs a remote Claude/OpenAI model) is chosen
by environment variables in .env -- see config.py / llm.py.

Run:  python ask.py "How is the President of Pakistan elected?"
"""

import sys

import config
from llm import generate_stream
from retrieve import load_store, retrieve

# The instruction template. This is "prompt engineering" -- the rules that turn
# a general chat model into a grounded, citing question-answerer.
PROMPT_TEMPLATE = """You are a careful assistant answering questions about \
the Constitution of Pakistan.
Use ONLY the context below to answer. If the answer is not in the context, say
"I couldn't find that in the provided text." Cite the page number(s) you used.

Context:
{context}

Question: {question}

Answer:"""


def build_prompt(question, results):
    """AUGMENTATION: stitch the retrieved chunks into the prompt template."""
    context_blocks = []
    for _idx, _score, page, text in results:
        context_blocks.append(f"[page {page}] {text}")
    context = "\n\n".join(context_blocks)
    return PROMPT_TEMPLATE.format(context=context, question=question)


def main():
    if len(sys.argv) < 2:
        print('Usage: python ask.py "your question here"')
        sys.exit(1)
    question = sys.argv[1]

    # 1. RETRIEVE
    vectors, texts, pages = load_store()
    results = retrieve(question, vectors, texts, pages)
    cited_pages = sorted({page for _, _, page, _ in results})
    print(f"Retrieved {len(results)} chunks from pages {cited_pages}")
    print(f"Generating with: {config.backend_label()}\n")

    # 2. AUGMENT
    prompt = build_prompt(question, results)

    # 3. GENERATE (streamed via the active local/remote backend)
    print("Answer:\n" + "-" * 60)
    for token in generate_stream(prompt):
        print(token, end="", flush=True)
    print("\n" + "-" * 60)


if __name__ == "__main__":
    main()
