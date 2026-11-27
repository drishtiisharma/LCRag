import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# File containing our previously generated embeddings
INPUT_FILE = Path("data/processed/embedded_docs.json")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    documents = json.load(f)

# Load the same model used to generate document embeddings
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Convert stored embeddings into a NumPy array
document_embeddings = np.array([
    doc["embedding"] for doc in documents
])

def search(query, top_k=5):

    # Generate embedding for the user's question
    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    )

    # Calculate similarity against every document chunk
    similarities = np.dot(
        document_embeddings,
        query_embedding
    )

    # Get the indices of the highest-scoring chunks
    top_indices = np.argsort(similarities)[::-1][:top_k]

    print("\nQuery:", query)
    print("=" * 70)

    for rank, index in enumerate(top_indices, start=1):

        doc = documents[index]

        print(f"\nResult {rank}")
        print("Source:", doc["source"])
        print("Chunk:", doc["chunk_index"])
        print("Similarity:", round(float(similarities[index]), 4))
        print("Content:", doc["text"][:500])
        print("-" * 70)


# Test queries
search("How does an AI agent use tools?")
# search("What is retrieval augmented generation?")
# search("How does document chunking work?")
