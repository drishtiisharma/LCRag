import json
from collections import Counter

DATA_PATH = "data/processed/embedded_docs.json"

with open(DATA_PATH, "r", encoding="utf-8") as f:
    documents = json.load(f)

print("========== DATASET OVERVIEW ==========")

print("Total chunks:", len(documents))

# Count documents by source
sources = Counter(doc["source"] for doc in documents)

print("Unique source files:", len(sources))

print("\nTop 10 source files:")
for source, count in sources.most_common(10):
    print(f"{source}: {count} chunks")

# Analyze text lengths
text_lengths = [len(doc["text"]) for doc in documents]

print("\n========== TEXT STATISTICS ==========")

print("Average text length:", sum(text_lengths) // len(text_lengths))
print("Shortest chunk:", min(text_lengths))
print("Longest chunk:", max(text_lengths))

# Check embedding dimensions
embedding_dimensions = set(
    len(doc["embedding"]) for doc in documents
)

print("\n========== EMBEDDING INFORMATION ==========")

print("Embedding dimensions:", embedding_dimensions)

# Check missing or empty text
empty_chunks = sum(
    1 for doc in documents if not doc["text"].strip()
)

print("Empty chunks:", empty_chunks)

print("\nDataset inspection completed!")
