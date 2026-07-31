import os
import json
import torch
import faiss
import numpy as np
from model.encoder import BiEncoder

BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(str(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data")
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "embeddings")
LEGACY_MODEL_PATH = os.path.join(BASE_DIR, "model")
FAQ_DATA_PATH = os.path.join(DATA_PATH, "faq_merged_v2.json")
VOCAB_PATH = os.path.join(LEGACY_MODEL_PATH, "vocab.json")
ENCODER_PATH = os.path.join(LEGACY_MODEL_PATH, "encoder.pth")
INDEX_PATH = os.path.join(EMBEDDINGS_PATH, "faiss_index.bin")

with open(FAQ_DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)
with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    vocab = json.load(f)

def tokenize(text, max_len=40):
    ids = [vocab.get(w, vocab["<unk>"]) for w in text.lower().split()]
    return torch.tensor(ids[:max_len] + [0]*(max_len-len(ids)))

model = BiEncoder(len(vocab))
model.load_state_dict(torch.load(ENCODER_PATH))
model.eval()

vectors = []
for d in data:
    x = tokenize(d["question"] + " " + d["answer"]).unsqueeze(0)
    with torch.no_grad():
        emb = model(x).numpy()
    vectors.append(emb[0])

vectors = np.array(vectors).astype("float32")
index = faiss.IndexFlatL2(vectors.shape[1])
index.add(vectors)

faiss.write_index(index, INDEX_PATH)
print("✅ FAISS index built")
