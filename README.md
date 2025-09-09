# NICU Protocols Q&A Agent (RAG)

This is a **ready-to-run** Retrieval-Augmented Generation (RAG) agent that answers questions
**only** from the included PDF: `data/NICU protocols.pdf`.

## What you get
- **Index builder** (`index.py`): chunks the PDF, generates embeddings with OpenAI, and stores a FAISS index.
- **Flask server** (`serve.py`): a tiny REST API and a minimal web UI at `/`.
- **System prompt** (`system_prompt.md`): guardrails for safe, cite-as-you-go answers tailored to clinical protocols.
- **.env.example**: put your API key here and rename to `.env`.

## Quickstart

1) **Python 3.10+ recommended**

2) Install deps:
```bash
pip install -r requirements.txt
```

3) Set your API key:
```bash
cp .env.example .env
# edit .env and add your OpenAI API key
```

4) Build the index (first time only):
```bash
python index.py
```

5) Run the server:
```bash
python serve.py
```
Then open your browser at **http://127.0.0.1:8000**.

## How it works
- **index.py** reads `data/NICU protocols.pdf` → splits into overlapping chunks with page numbers →
  gets embeddings using `text-embedding-3-large` → builds a FAISS vector store saved to `rag.index` and `rag.meta.json`.
- **serve.py** loads the store, performs top-k semantic search for each user question, then calls an OpenAI chat model
  with a strict **system prompt** and the retrieved context. The agent must **cite page numbers** and
  avoid hallucinating outside the PDF.

## Swap in a new file
Replace the file at `data/NICU protocols.pdf` and **re-run `python index.py`**.

## Notes
- This is **for education/reference only**. It does **not** provide medical advice and must not replace clinical judgment.
- If you deploy on a server, consider authentication, rate limits, PHI handling, and logging policies.
