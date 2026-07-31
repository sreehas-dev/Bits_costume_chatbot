import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

K = 3

model = SentenceTransformer(
    os.path.join(BASE_DIR, "finetuned_model")
)

index = faiss.read_index(
    os.path.join(BASE_DIR, "embeddings", "faiss_index.bin")
)

with open(os.path.join(BASE_DIR, "data", "val.json")) as f:
    data = json.load(f)

with open(os.path.join(BASE_DIR, "data", "faq_merged_v2.json")) as f:
    full = json.load(f)

correct = 0

for item in data:
    emb = model.encode(item["question"], normalize_embeddings=True)
    _, I = index.search(np.array([emb]).astype("float32"), K)

    retrieved = [full[i]["answer"] for i in I[0]]
    if item["answer"] in retrieved:
        correct += 1

print(f"Recall@{K}: {correct/len(data):.2f}")
