import numpy as np
import os
import json
import faiss
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = SentenceTransformer(
    os.path.join(BASE_DIR, "finetuned_model")
)

with open(os.path.join(BASE_DIR, "data", "val.json")) as f:
    data = json.load(f)

index = faiss.read_index(
    os.path.join(BASE_DIR, "embeddings", "faiss_index.bin")
)

scores = []

for item in data:
    emb = model.encode(item["question"], normalize_embeddings=True)
    D, _ = index.search(np.array([emb]).astype("float32"), 1)
    scores.append(float(D[0][0]))

print("Confidence distribution:")
print("Min:", min(scores))
print("Max:", max(scores))
print("Mean:", sum(scores)/len(scores))
