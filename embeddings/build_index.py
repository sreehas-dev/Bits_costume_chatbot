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

texts = [d["question"] + " " + d["answer"] for d in data]

embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=True
)

embeddings = np.array(embeddings).astype("float32")

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, INDEX_PATH)
print("✅ FAISS index rebuilt")
