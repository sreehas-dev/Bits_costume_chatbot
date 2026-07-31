import os
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(str(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "finetuned_model")
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "embeddings")
DATA_PATH = os.path.join(BASE_DIR, "data")
INDEX_PATH = os.path.join(EMBEDDINGS_PATH, "faiss_index.bin")
FAQ_DATA_PATH = os.path.join(DATA_PATH, "faq_merged_v2.json")

THRESHOLD = 0.60   # tune between 0.65–0.80

model = SentenceTransformer(str(MODEL_PATH))
index = faiss.read_index(INDEX_PATH)
with open(FAQ_DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

while True:
    q = input("Ask: ")
    if q == "exit":
        break

    emb = model.encode(
        q,
        normalize_embeddings=True
    ).astype("float32")

    D, I = index.search(np.array([emb]), 3)

    if D[0][0] < THRESHOLD:
        print("❌ Sorry, I don’t know the answer.")
        continue

    print("Confidence:", round(float(D[0][0]), 3))
    for i in I[0]:
        print("→", data[i]["question"])
