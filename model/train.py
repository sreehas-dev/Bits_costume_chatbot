import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import FAQDataset
from encoder import BiEncoder
from tqdm import tqdm
import json

BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(str(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data")
TRAIN_DATA_PATH = os.path.join(DATA_PATH, "train.json")

# Load data
with open(TRAIN_DATA_PATH, "r", encoding="utf-8") as f:
    train_data = json.load(f)
texts = [d["question"] + " " + d["answer"] for d in train_data]

# Build vocab
vocab = {"<pad>": 0, "<unk>": 1}
for t in texts:
    for w in t.lower().split():
        if w not in vocab:
            vocab[w] = len(vocab)

dataset = FAQDataset(TRAIN_DATA_PATH, vocab)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = BiEncoder(len(vocab))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CosineEmbeddingLoss()

model.train()
for epoch in range(800):
    total_loss = 0
    for q, a in tqdm(loader):
        optimizer.zero_grad()
        q_emb = model(q)
        a_emb = model(a)
        labels = torch.ones(q.size(0))
        loss = criterion(q_emb, a_emb, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "encoder.pth")
json.dump(vocab, open("vocab.json", "w"))
print("✅ Model training completed")
