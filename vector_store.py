
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from document_processor import extract_text_from_pdf, chunk_text

# Step 1: Define project paths
BASE_DIR = Path(__file__).resolve().parent

PDF_PATH = BASE_DIR / "data" / "rag_sample_text_document.pdf"

DB_PATH = BASE_DIR / "chroma_db"

# Step 2: Load the embedding model
print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded successfully!")

# Step 3: Extract text from PDF
print("\nExtracting PDF text...")

text = extract_text_from_pdf(str(PDF_PATH))

# Step 4: Divide text into chunks
print("\nCreating text chunks...")

chunks = chunk_text(text)

print("Total chunks:", len(chunks))

# Step 5: Generate embeddings
print("\nGenerating embeddings...")

embeddings = model.encode(
    chunks,
    show_progress_bar=True
)

print("Embeddings generated successfully!")

print("Embedding dimensions:", embeddings.shape)

# Step 6: Create persistent ChromaDB database
print("\nConnecting to ChromaDB...")

client = chromadb.PersistentClient(
    path=str(DB_PATH)
)

collection = client.get_or_create_collection(
    name="document_collection",
    metadata={"hnsw:space": "cosine"}
)

# Step 7: Store chunks and embeddings
ids = [
    f"document-chunk-{i}"
    for i in range(len(chunks))
]

collection.upsert(
    ids=ids,
    documents=chunks,
    embeddings=embeddings.tolist(),
    metadatas=[
        {"source": PDF_PATH.name, "chunk_number": i}
        for i in range(len(chunks))
    ]
)

print("\nVECTOR DATABASE STORAGE SUCCESSFUL!")

print("Total stored chunks:", collection.count())

# Step 8: Test semantic search
query = "What is a vector database?"

print("\nSearching for:", query)

query_embedding = model.encode(query).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=min(3, collection.count())
)

# Step 9: Display retrieved results
print("\nRETRIEVED DOCUMENT CHUNKS")
print("=" * 50)

for i, document in enumerate(results["documents"][0]):
    print(f"\nResult {i + 1}:")
    print(document)

print("\nSTAGE 3 COMPLETED SUCCESSFULLY!")