import json

with open("faq_dataset_v1.json") as f:
    original = json.load(f)

with open("faq_augmented.json") as f:
    augmented = json.load(f)

merged = []

# Keep original as-is
for item in original:
    merged.append(item)

# Add augmented (map fields correctly)
for idx, item in enumerate(augmented):
    merged.append({
        "id": f"AUG_{idx:05d}",
        "category": "AUGMENTED",
        "question": item["question"],
        "answer": item["answer"],
        "difficulty": "Medium",
        "source": "Synthetic-Augmented"
    })

with open("faq_merged_v2.json", "w") as f:
    json.dump(merged, f, indent=2)

print(f"✅ Merged dataset size: {len(merged)}")
