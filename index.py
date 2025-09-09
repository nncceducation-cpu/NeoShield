import os, json, math
from typing import List, Dict
import numpy as np
from pypdf import PdfReader
from openai import OpenAI
import faiss
from dotenv import load_dotenv

load_dotenv()
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")
PDF_PATH = "data/NICU protocols.pdf"

OUT_INDEX = "rag.index"
OUT_META  = "rag.meta.json"

def read_pdf_with_pages(path:str)->List[Dict]:
    reader = PdfReader(path)
    chunks = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        # normalize whitespace
        text = " ".join(text.split())
        chunks.append({"page": i, "text": text})
    return chunks

def split_into_chunks(pages: List[Dict], max_tokens: int = 800, overlap: int = 120) -> List[Dict]:
    # Token-less simple splitter by characters; suitable for many cases.
    # You can replace with tiktoken-aware splitter if desired.
    chunks = []
    for item in pages:
        txt = item["text"]
        page = item["page"]
        if not txt.strip():
            continue
        step = max_tokens - overlap
        start = 0
        while start < len(txt):
            end = min(len(txt), start + max_tokens)
            chunk = txt[start:end]
            chunks.append({"page": page, "text": chunk})
            start += step
    return chunks

def embed_texts(client: OpenAI, texts: List[str]) -> np.ndarray:
    # Batch for efficiency
    embs = []
    B = 64
    for i in range(0, len(texts), B):
        batch = texts[i:i+B]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        for d in resp.data:
            embs.append(d.embedding)
    return np.array(embs, dtype="float32")

def main():
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"Missing {PDF_PATH}. Put your PDF there.")
    client = OpenAI()
    pages = read_pdf_with_pages(PDF_PATH)
    chunks = split_into_chunks(pages)

    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks...")
    mat = embed_texts(client, texts)

    dim = mat.shape[1]
    index = faiss.IndexFlatIP(dim)
    # normalize to cosine
    faiss.normalize_L2(mat)
    index.add(mat)

    meta = {"chunks": chunks}
    with open(OUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)

    faiss.write_index(index, OUT_INDEX)
    print(f"Saved index to {OUT_INDEX} and metadata to {OUT_META}.")

if __name__ == "__main__":
    main()
