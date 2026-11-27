
import json
import re
from pathlib import Path

INPUT_FILE = Path("data/raw/langchain_docs.json")
OUTPUT_FILE = Path("data/processed/cleaned_docs.json")

# Load raw documents
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    documents = json.load(f)

cleaned_documents = []

for doc in documents:

    # Extract actual text using the correct key
    text = doc.get("text", "")

    # Clean excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove trailing spaces from each line
    text = "\n".join(line.strip() for line in text.splitlines())

    # Remove excessive spaces
    text = re.sub(r" {2,}", " ", text)

    # Skip empty documents
    if not text.strip():
        continue

    cleaned_documents.append({
        "source": doc.get("source", "unknown"),
        "text": text
    })

# Create output directory
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# Save cleaned documents
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(cleaned_documents, f, indent=2, ensure_ascii=False)

print("Preprocessing completed!")
print("Original documents:", len(documents))
print("Cleaned documents:", len(cleaned_documents))
print("Saved to:", OUTPUT_FILE)
