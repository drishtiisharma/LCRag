import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

INPUT_FILE = Path("data/processed/chunked_docs.json")
OUTPUT_FILE = Path("data/processed/embedded_docs.json")

# Load chunked documents
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    documents = json.load(f)

# Load embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

texts = [doc["text"] for doc in documents]

# Generate embeddings
embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)

# Attach embeddings to documents
embedded_documents = []

for doc, embedding in zip(documents, embeddings):
    embedded_documents.append({
        "source": doc["source"],
        "chunk_index": doc["chunk_index"],
        "text": doc["text"],
        "embedding": embedding.tolist()
    })

# Save results
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(embedded_documents, f, ensure_ascii=False)

print("\nEmbedding generation completed!")
print("Total chunks:", len(embedded_documents))
print("Embedding dimensions:", len(embeddings[0]))
print("Saved to:", OUTPUT_FILE)
