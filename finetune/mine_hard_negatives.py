import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(str(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "finetuned_model")
DATA_PATH = os.path.join(BASE_DIR, "data")
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "embeddings")
FAQ_DATA_PATH = os.path.join(DATA_PATH, "faq_merged_v2.json")
INDEX_PATH = os.path.join(EMBEDDINGS_PATH, "faiss_index.bin")

model = SentenceTransformer(str(MODEL_PATH))
with open(FAQ_DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)
index = faiss.read_index(INDEX_PATH)

hard_triplets = []

for i, item in enumerate(data):
    q = item["question"]
    q_emb = model.encode(q, normalize_embeddings=True)

    scores, indices = index.search(
        np.array([q_emb]).astype("float32"), 5
    )

    for idx in indices[0]:
        if idx != i:
            hard_triplets.append({
                "anchor": q,
                "positive": item["answer"],
                "negative": data[idx]["answer"]
            })
            break

json.dump(hard_triplets, open("hard_triplets.json", "w"), indent=2)
print("✅ Hard negatives generated")
