from pathlib import Path
import re
import json

# Path to the cloned LangChain documentation
SOURCE_DIR = Path(r"C:\Users\freedom\Desktop\langchain-rag\docs\src\oss\langchain")

# Path to save extracted documents
OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

documents = []

# Find all Markdown and MDX files
files = list(SOURCE_DIR.rglob("*.md"))
files += list(SOURCE_DIR.rglob("*.mdx"))

print("Total files found:", len(files))

# Read each file
for file_path in files:
    text = file_path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    # Remove YAML frontmatter
    text = re.sub(
        r"\A---\s*\n.*?\n---\s*\n",
        "",
        text,
        flags=re.S
    )

    # Remove import/export statements
    text = re.sub(
        r"^\s*(import|export)\s+.*?$",
        "",
        text,
        flags=re.M
    )

    # Remove JSX tags
    text = re.sub(
        r"</?[A-Za-z][^>]*>",
        " ",
        text
    )

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    text = text.strip()

    if len(text) < 100:
        continue

    documents.append({
        "source": str(file_path.relative_to(SOURCE_DIR)),
        "text": text
    })

# Save extracted documents
output_file = OUTPUT_DIR / "langchain_docs.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(
        documents,
        f,
        ensure_ascii=False,
        indent=2
    )

print("\nExtraction completed!")
print("Documents extracted:", len(documents))
print("Saved to:", output_file)
