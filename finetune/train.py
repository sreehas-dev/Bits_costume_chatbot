import os
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import json

BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(str(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "finetuned_model")

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

triplets = json.load(open("hard_triplets.json"))

train_examples = [
    InputExample(texts=[t["anchor"], t["positive"], t["negative"]])
    for t in triplets
]

loader = DataLoader(train_examples, shuffle=True, batch_size=16)

loss = losses.TripletLoss(
    model=model,
    distance_metric=losses.TripletDistanceMetric.COSINE
)

model.fit(
    [(loader, loss)],
    epochs=3,
    warmup_steps=100
)

model.save(str(MODEL_PATH))
print("✅ Retrained with hard negatives")
