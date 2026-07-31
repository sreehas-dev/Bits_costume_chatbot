import json
import random
import torch
from torch.utils.data import Dataset

class FAQTripletDataset(Dataset):
    def __init__(self, path):
        with open(path) as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        anchor = self.data[idx]["question"]
        positive = self.data[idx]["answer"]

        # Hard negative: different answer
        neg_idx = random.choice([i for i in range(len(self.data)) if i != idx])
        negative = self.data[neg_idx]["answer"]

        return anchor, positive, negative
