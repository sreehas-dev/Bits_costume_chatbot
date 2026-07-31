import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ---------- CONFIG ----------
FAISS_INDEX_PATH = "faiss_index.bin"
METADATA_PATH = "metadata.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3

# ---------- LOAD INDEX ----------
index = faiss.read_index(FAISS_INDEX_PATH)

# ---------- LOAD METADATA ----------
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# ---------- LOAD MODEL ----------
model = SentenceTransformer(EMBEDDING_MODEL)

# ---------- SEARCH FUNCTION ----------
def semantic_search(query: str, top_k: int = TOP_K):
    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    ).astype("float32")

    query_embedding = np.array([query_embedding])

    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        item = metadata[idx]
        results.append({
            "id": item["id"],
            "category": item["category"],
            "question": item["question"],
            "answer": item["answer"],
            "similarity_score": round(float(score), 4)
        })

    return results


# ---------- TEST ----------
if __name__ == "__main__":
    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() == "exit":
            break

        results = semantic_search(query)

        print("\n🔍 Top Matches:\n")
        for i, r in enumerate(results, start=1):
            print(f"{i}. [{r['similarity_score']}] {r['question']}")
