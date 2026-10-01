from pathlib import Path
import requests
import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# STEP 1: DEFINE PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "chroma_db"


# ============================================================
# STEP 2: LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded successfully!")


# ============================================================
# STEP 3: CONNECT TO CHROMADB
# ============================================================

print("\nConnecting to ChromaDB...")

client = chromadb.PersistentClient(
    path=str(DB_PATH)
)

collection = client.get_collection(
    name="document_collection"
)

total_chunks = collection.count()

print("Connected to vector database!")
print("Total stored chunks:", total_chunks)


# Check if database contains documents
if total_chunks == 0:
    print("\nERROR: No document chunks found in ChromaDB.")
    print("Please run vector_store.py first.")
    exit()


# ============================================================
# STEP 4: GET USER QUESTION
# ============================================================

query = input("\nAsk a question about your PDF: ").strip()

if not query:
    print("\nERROR: Question cannot be empty.")
    exit()


# ============================================================
# STEP 5: CONVERT QUESTION INTO EMBEDDING
# ============================================================

print("\nConverting question into embedding...")

query_embedding = model.encode(query).tolist()

print("Question embedding generated successfully!")


# ============================================================
# STEP 6: RETRIEVE RELEVANT DOCUMENT CHUNKS
# ============================================================

print("\nSearching vector database...")

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=min(3, total_chunks),
    include=["documents", "distances"]
)

retrieved_chunks = results["documents"][0]
distances = results["distances"][0]


# ============================================================
# STAGE 5: RETRIEVAL TEST
# ============================================================

print("\n" + "=" * 60)
print("STAGE 5: RETRIEVAL TEST")
print("=" * 60)

print("\nQuery:")
print(query)

print("\nRetrieved Document Chunks:")
print("-" * 60)

for i, chunk in enumerate(retrieved_chunks):

    print(f"\nChunk {i + 1}")
    print("-" * 40)

    print(chunk)

    print(f"\nCosine Distance: {distances[i]:.4f}")


print("\nTotal Retrieved Chunks:", len(retrieved_chunks))


# ============================================================
# STEP 7: CHECK RETRIEVAL RELEVANCE
# ============================================================

print("\n" + "=" * 60)
print("RETRIEVAL RELEVANCE CHECK")
print("=" * 60)

# ChromaDB cosine distance:
# Lower distance = more similar
#
# We use a conservative threshold for this test.
# If the best result is above this value, the retrieved
# information is treated as insufficiently relevant.

RELEVANCE_THRESHOLD = 0.70

best_distance = distances[0]

print(f"\nBest retrieval distance: {best_distance:.4f}")
print(f"Relevance threshold: {RELEVANCE_THRESHOLD:.2f}")


if best_distance > RELEVANCE_THRESHOLD:

    print("\nRETRIEVAL RESULT: NOT RELEVANT")
    print("-" * 60)

    print(
        "The retrieved document context does not appear "
        "sufficiently relevant to the question."
    )

    print(
        "\nThe information is not available in the "
        "provided document."
    )

    print("\n" + "=" * 60)
    print("STAGE 5 TEST COMPLETED")
    print("=" * 60)

    exit()


# ============================================================
# STEP 8: RETRIEVAL PASSED
# ============================================================

print("\nRETRIEVAL RESULT: RELEVANT")
print("Relevant document context found.")


# ============================================================
# STEP 9: COMBINE RETRIEVED CHUNKS
# ============================================================

context = "\n\n".join(retrieved_chunks)


# ============================================================
# STEP 10: DISPLAY CONTEXT SENT TO LLAMA
# ============================================================

print("\n" + "=" * 60)
print("CONTEXT SENT TO LLAMA")
print("=" * 60)

print(context)


# ============================================================
# STEP 11: PREPARE RAG PROMPT
# ============================================================

prompt = f"""
You are a document question-answering assistant.

IMPORTANT RULES:

1. Answer ONLY using the DOCUMENT CONTEXT provided below.

2. Do NOT use outside knowledge.

3. Do NOT guess.

4. Do NOT invent information.

5. Do NOT explain how a RAG pipeline works.

6. If the answer is not present in the DOCUMENT CONTEXT,
   respond exactly:

"The information is not available in the provided document."

DOCUMENT CONTEXT:
-----------------
{context}
-----------------

USER QUESTION:
{query}

FINAL ANSWER:
"""


# ============================================================
# STEP 12: GENERATE ANSWER USING OLLAMA
# ============================================================

print("\n" + "=" * 60)
print("GENERATING ANSWER USING LLAMA 3.2")
print("=" * 60)


try:

    response = requests.post(
        "http://localhost:11434/api/generate",

        json={
            "model": "llama3.2:1b",
            "prompt": prompt,
            "stream": False,

            "options": {
                "temperature": 0
            }
        },

        timeout=180
    )

    response.raise_for_status()

except requests.exceptions.RequestException as error:

    print("\nERROR: Could not connect to Ollama.")
    print("Make sure Ollama is running.")
    print("\nDetails:", error)

    exit()


# ============================================================
# STEP 13: EXTRACT ANSWER
# ============================================================

answer = response.json()["response"].strip()


# ============================================================
# STEP 14: DISPLAY FINAL ANSWER
# ============================================================

print("\n" + "=" * 60)
print("RAG GENERATED ANSWER")
print("=" * 60)

print(answer)


# ============================================================
# STAGE 5 COMPLETION
# ============================================================

print("\n" + "=" * 60)
print("STAGE 5 TEST COMPLETED SUCCESSFULLY!")
print("=" * 60)