import json
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

INPUT_FILE = Path("data/processed/cleaned_docs.json")
OUTPUT_FILE = Path("data/processed/chunked_docs.json")

# Load cleaned documents
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    documents = json.load(f)

# Configure the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)

chunked_documents = []

for doc in documents:
    chunks = text_splitter.split_text(doc["text"])

    for index, chunk in enumerate(chunks):
        chunked_documents.append({
            "source": doc["source"],
            "chunk_index": index,
            "text": chunk
        })

# Save chunks
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(chunked_documents, f, indent=2, ensure_ascii=False)

print("Chunking completed!")
print("Original documents:", len(documents))
print("Total chunks:", len(chunked_documents))
print("Saved to:", OUTPUT_FILE)
