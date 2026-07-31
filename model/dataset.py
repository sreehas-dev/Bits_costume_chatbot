import json
import torch
from torch.utils.data import Dataset

def tokenize(text, vocab, max_len=40):
    tokens = text.lower().split()
    ids = [vocab.get(t, vocab["<unk>"]) for t in tokens][:max_len]
    return ids + [0] * (max_len - len(ids))

class FAQDataset(Dataset):
    def __init__(self, path, vocab):
        with open(path) as f:
            self.data = json.load(f)
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        q = self.data[idx]["question"]
        a = self.data[idx]["answer"]

        q_ids = tokenize(q, self.vocab)
        a_ids = tokenize(a, self.vocab)

        return torch.tensor(q_ids), torch.tensor(a_ids)
