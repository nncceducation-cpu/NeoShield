import os, json
from typing import List, Dict
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from openai import OpenAI
import faiss
from dotenv import load_dotenv

load_dotenv()
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")
CHAT_MODEL  = os.getenv("CHAT_MODEL", "gpt-4o-mini")
PORT        = int(os.getenv("PORT", "8000"))
TOP_K       = int(os.getenv("TOP_K", "6"))

INDEX_PATH = "rag.index"
META_PATH  = "rag.meta.json"

app = Flask(__name__, static_url_path="/static", static_folder="static")

def load_index():
    if not (os.path.exists(INDEX_PATH) and os.path.exists(META_PATH)):
        raise FileNotFoundError("Index not found. Run: python index.py")
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return index, meta

index, meta = load_index()
chunks = meta["chunks"]

def embed(client: OpenAI, texts: List[str]):
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = [d.embedding for d in resp.data]
    import numpy as np
    vecs = np.array(vecs, dtype="float32")
    faiss.normalize_L2(vecs)
    return vecs

def retrieve(query: str, top_k: int = TOP_K) -> List[Dict]:
    client = OpenAI()
    qvec = embed(client, [query])
    D, I = index.search(qvec, top_k)
    out = []
    for idx, score in zip(I[0], D[0]):
        ch = chunks[int(idx)]
        out.append({"page": ch["page"], "text": ch["text"], "score": float(score)})
    return out

def build_prompt(system_prompt: str, query: str, contexts: List[Dict]) -> List[Dict]:
    # Compose the context with page labels for citation
    context_strs = []
    for c in contexts:
        context_strs.append(f"[p.{c['page']}] {c['text']}")
    ctx = "\n\n".join(context_strs)

    messages = [
        {"role":"system","content": system_prompt},
        {"role":"user","content": f"Question: {query}\n\nContext from NICU protocols (with page markers):\n{ctx}\n\nRules: Answer ONLY from the context. Cite pages like [NICU Protocols, p. X]. If missing, say you couldn't find it."}
    ]
    return messages

def chat(query: str) -> Dict:
    client = OpenAI()
    with open("system_prompt.md","r",encoding="utf-8") as f:
        sys_prompt = f.read()

    ctx = retrieve(query)
    messages = build_prompt(sys_prompt, query, ctx)
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.2,
    )
    answer = resp.choices[0].message.content
    return {"answer": answer, "contexts": ctx}

@app.route("/", methods=["GET"])
def home():
    return app.send_static_file("index.html")

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(force=True)
    q = data.get("q","").strip()
    if not q:
        return jsonify({"error":"Empty query"}), 400
    try:
        result = chat(q)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=PORT, debug=False)
