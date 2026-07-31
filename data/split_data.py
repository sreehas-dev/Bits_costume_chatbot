import json
import random

with open("faq_dataset_v1.json") as f:
    data = json.load(f)

random.shuffle(data)

split = int(0.9 * len(data))
train, val = data[:split], data[split:]

with open("train.json", "w") as f:
    json.dump(train, f, indent=2)

with open("val.json", "w") as f:
    json.dump(val, f, indent=2)

print("✅ Dataset split completed")
