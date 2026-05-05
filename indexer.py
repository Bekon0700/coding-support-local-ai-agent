import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# ─── Config ───────────────────────────────────────────────
CODEBASE_PATH = "./code-files"
DB_PATH = "./db"
EMBED_MODEL = "nomic-embed-text"

INCLUDE_EXTENSIONS = {
    ".ts", ".tsx", ".js", ".jsx",
    ".json", ".css", ".mjs", ".md"
}

EXCLUDE_DIRS = {
    "node_modules", ".next", ".git",
    ".vscode", "__pycache__"
}
# ──────────────────────────────────────────────────────────


def load_files(base_path):
    """Walk the codebase and load all relevant files."""
    docs = []
    skipped = []

    for root, dirs, files in os.walk(base_path):
        # Remove excluded dirs in-place so os.walk skips them
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

        for filename in files:
            ext = os.path.splitext(filename)[1].lower()
            if ext not in INCLUDE_EXTENSIONS:
                continue

            filepath = os.path.join(root, filename)
            try:
                loader = TextLoader(filepath, encoding="utf-8")
                loaded = loader.load()
                # Tag each doc with its file path as metadata
                for doc in loaded:
                    doc.metadata["source"] = filepath
                docs.extend(loaded)
                print(f"  ✅ Loaded: {filepath}")
            except Exception as e:
                skipped.append(filepath)
                print(f"  ⚠️  Skipped: {filepath} ({e})")

    return docs, skipped


def split_documents(docs):
    """Split documents into smaller chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    return chunks

def enrich_chunks(chunks):
    """
    Prepend filename context to each chunk's content.
    This helps the embeddings carry file identity,
    so semantic search can find chunks by filename too.
    """
    for chunk in chunks:
        source = chunk.metadata.get("source", "")
        filename = os.path.basename(source)
        # Prepend: "File: types.ts\n\n<actual code>"
        chunk.page_content = f"File: {filename}\n\n{chunk.page_content}"
    return chunks

def build_vectorstore(chunks):
    """Embed chunks and store in ChromaDB."""
    print(f"\n🔢 Embedding {len(chunks)} chunks using '{EMBED_MODEL}'...")
    print("   (This may take a few minutes the first time)\n")

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )

    return vectorstore


def main():
    print("=" * 50)
    print("🗂️  Code Assistant — Indexer")
    print("=" * 50)

    # Step 1: Load files
    print(f"\n📂 Loading files from: {CODEBASE_PATH}\n")
    docs, skipped = load_files(CODEBASE_PATH)
    print(f"\n📄 Loaded {len(docs)} files, skipped {len(skipped)}")

    if not docs:
        print("❌ No files loaded. Check your CODEBASE_PATH.")
        return

    # Step 2: Split into chunks
    print(f"\n✂️  Splitting into chunks...")
    chunks = split_documents(docs)
    print(f"   → {len(docs)} files split into {len(chunks)} chunks")

    # Step 2.5: Enrich chunks with filename context  ← ADD THIS
    chunks = enrich_chunks(chunks)
    print(f"   → Enriched {len(chunks)} chunks with filename metadata")

    # Step 3: Embed and store
    vectorstore = build_vectorstore(chunks)

    print("\n" + "=" * 50)
    print(f"✅ Done! Vector DB saved to: {DB_PATH}")
    print(f"   Total chunks indexed: {len(chunks)}")
    print("=" * 50)


if __name__ == "__main__":
    main()