import json

with open("data/processed/embedded_docs.json", "r", encoding="utf-8") as f:
    original = json.load(f)

removed = [
    doc for doc in original
    if len(doc["text"].strip()) < 100
]

print("========== REMOVED CHUNKS ==========")
print("Total removed:", len(removed))

for i, doc in enumerate(removed, 1):
    print(f"\n--- Chunk {i} ---")
    print("Source:", doc["source"])
    print("Chunk index:", doc["chunk_index"])
    print("Text:", repr(doc["text"]))
    print("Length:", len(doc["text"].strip()))
