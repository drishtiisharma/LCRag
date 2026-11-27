import json
import os
import chromadb

# Paths
INPUT_PATH = "scripts/data/processed/embedded_docs.json"
DB_PATH = "scripts/data/chroma_db"

COLLECTION_NAME = "langchain_docs"

# Load embedded documents
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    documents = json.load(f)

print("Total documents loaded:", len(documents))

# Initialize persistent ChromaDB
client = chromadb.PersistentClient(path=DB_PATH)

# Create or get collection
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

# Prepare valid documents
valid_ids = []
valid_texts = []
valid_embeddings = []
valid_metadatas = []

for index, doc in enumerate(documents):

    text = doc.get("text", "").strip()
    embedding = doc.get("embedding")

    # Skip empty text
    if not text:
        continue

    # Skip missing embeddings
    if not embedding:
        print(f"Skipping chunk {index}: Missing embedding")
        continue

    # Validate embedding values
    if not all(isinstance(x, (int, float)) for x in embedding):
        print(f"Skipping chunk {index}: Invalid embedding")
        continue

    valid_ids.append(str(index))
    valid_texts.append(text)
    valid_embeddings.append(embedding)

    valid_metadatas.append({
        "source": str(doc.get("source", "unknown")),
        "chunk_index": int(doc.get("chunk_index", index))
    })

print("Valid documents:", len(valid_ids))

# Insert into ChromaDB in batches
BATCH_SIZE = 100

for start in range(0, len(valid_ids), BATCH_SIZE):

    end = start + BATCH_SIZE

    collection.upsert(
        ids=valid_ids[start:end],
        documents=valid_texts[start:end],
        embeddings=valid_embeddings[start:end],
        metadatas=valid_metadatas[start:end]
    )

    print(f"Stored {min(end, len(valid_ids))}/{len(valid_ids)}")

print("\nChromaDB storage completed!")

print("Collection:", COLLECTION_NAME)
print("Total stored:", collection.count())
print("Database location:", DB_PATH)
