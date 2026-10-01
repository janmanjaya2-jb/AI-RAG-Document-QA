from pathlib import Path
import hashlib
import uuid

import chromadb
import requests
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI RAG Document Q&A",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI RAG Document Q&A")
st.write(
    "Upload PDF documents and ask questions about their content "
    "using Retrieval-Augmented Generation (RAG)."
)

st.divider()

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "chroma_db"

COLLECTION_NAME = "document_collection"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:1b"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 3

# --------------------------------------------------
# LOAD MODEL AND DATABASE
# --------------------------------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

@st.cache_resource
def connect_database():
    client = chromadb.PersistentClient(path=str(DB_PATH))

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "PDF document chunks for RAG"}
    )

    return collection

try:
    with st.spinner("Loading embedding model and vector database..."):
        embedding_model = load_embedding_model()
        collection = connect_database()

    st.success("Embedding model and vector database connected!")

except Exception as e:
    st.error(f"Could not initialize the application: {e}")
    st.stop()

# --------------------------------------------------
# TEXT CHUNKING
# --------------------------------------------------

def split_text_into_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Split extracted text into overlapping chunks.
    """

    text = " ".join(text.split())

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks

# --------------------------------------------------
# PDF EXTRACTION AND STORAGE
# --------------------------------------------------

def process_uploaded_pdf(uploaded_file):
    """
    Extract text page by page, split it into chunks,
    generate embeddings, and store the chunks in ChromaDB.
    """

    pdf_bytes = uploaded_file.getvalue()
    reader = PdfReader(uploaded_file)

    document_chunks = []
    chunk_metadata = []

    for page_index, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""

        if not page_text.strip():
            continue

        chunks = split_text_into_chunks(page_text)

        for chunk in chunks:
            document_chunks.append(chunk)

            chunk_metadata.append({
                "source": uploaded_file.name,
                "page": page_index + 1
            })

    if not document_chunks:
        return 0, "No extractable text found. This may be a scanned PDF."

    # Generate embeddings for all chunks
    embeddings = embedding_model.encode(
        document_chunks,
        show_progress_bar=False
    ).tolist()

    # Unique IDs prevent uploaded documents from overwriting each other
    file_hash = hashlib.sha256(pdf_bytes).hexdigest()[:12]

    ids = [
        f"{file_hash}_{uuid.uuid4().hex}"
        for _ in document_chunks
    ]

    # Store chunks, embeddings, and source metadata
    collection.add(
        ids=ids,
        documents=document_chunks,
        embeddings=embeddings,
        metadatas=chunk_metadata
    )

    return len(document_chunks), None

# --------------------------------------------------
# PDF UPLOAD INTERFACE
# --------------------------------------------------

st.header("1. Upload PDF Documents")

uploaded_files = st.file_uploader(
    "Choose one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

if st.button("Process Uploaded PDFs", type="primary"):
    if not uploaded_files:
        st.warning("Please select at least one PDF file.")
    else:
        total_added = 0

        for uploaded_file in uploaded_files:
            try:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    number_of_chunks, error_message = process_uploaded_pdf(
                        uploaded_file
                    )

                if error_message:
                    st.warning(
                        f"{uploaded_file.name}: {error_message}"
                    )
                else:
                    st.success(
                        f"{uploaded_file.name}: "
                        f"Processed and stored {number_of_chunks} chunks."
                    )
                    total_added += number_of_chunks

            except Exception as e:
                st.error(
                    f"Error processing {uploaded_file.name}: {e}"
                )

        if total_added > 0:
            st.success(
                f"Finished! Added {total_added} chunks to the vector database."
            )

# --------------------------------------------------
# DATABASE INFORMATION
# --------------------------------------------------

total_chunks = collection.count()

col1, col2 = st.columns(2)

with col1:
    st.metric("Stored Document Chunks", total_chunks)

with col2:
    st.metric("Language Model", OLLAMA_MODEL)

st.divider()

# --------------------------------------------------
# QUESTION AND ANSWER
# --------------------------------------------------

st.header("2. Ask Questions About Your Documents")

query = st.text_input(
    "Enter your question:",
    placeholder="What is the main topic of the uploaded document?"
)

if st.button("Generate Answer"):
    if not query.strip():
        st.warning("Please enter a question.")

    elif total_chunks == 0:
        st.warning(
            "No document chunks are stored yet. Upload and process a PDF first."
        )

    else:
        try:
            # ------------------------------------------
            # RETRIEVE RELEVANT CHUNKS
            # ------------------------------------------

            with st.spinner("Searching your documents..."):
                query_embedding = embedding_model.encode(query).tolist()

                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(TOP_K, total_chunks),
                    include=["documents", "metadatas", "distances"]
                )

                retrieved_chunks = results["documents"][0]
                retrieved_metadata = results["metadatas"][0]

            if not retrieved_chunks:
                st.warning("No document content was retrieved.")

            else:
                # --------------------------------------
                # BUILD CONTEXT WITH SOURCE INFORMATION
                # --------------------------------------

                context_parts = []

                for index, chunk in enumerate(retrieved_chunks):
                    metadata = retrieved_metadata[index] or {}

                    source = metadata.get("source", "Unknown document")
                    page = metadata.get("page", "Unknown")

                    context_parts.append(
                        f"[Source: {source}, Page: {page}]\n{chunk}"
                    )

                context = "\n\n".join(context_parts)

                # --------------------------------------
                # DISPLAY RETRIEVED CONTEXT
                # --------------------------------------

                st.subheader("Retrieved Document Context")

                for index, chunk in enumerate(retrieved_chunks):
                    metadata = retrieved_metadata[index] or {}

                    source = metadata.get("source", "Unknown document")
                    page = metadata.get("page", "Unknown")

                    with st.expander(
                        f"Chunk {index + 1} — {source}, Page {page}"
                    ):
                        st.write(chunk)

                # --------------------------------------
                # CONSTRUCT GROUNDED PROMPT
                # --------------------------------------

                prompt = f"""
You are a document question-answering assistant.

Answer the user's question using only the DOCUMENT CONTEXT below.

Rules:
1. Do not use outside knowledge.
2. If the answer is not present in the context, say:
   "The information is not available in the uploaded documents."
3. Do not invent facts, names, dates, or citations.
4. Keep the answer clear and concise.
5. When possible, mention the source document and page number.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{query}

ANSWER:
"""

                # --------------------------------------
                # GENERATE ANSWER USING OLLAMA
                # --------------------------------------

                st.subheader("AI Generated Answer")

                with st.spinner("Generating answer with Llama..."):
                    response = requests.post(
                        OLLAMA_URL,
                        json={
                            "model": OLLAMA_MODEL,
                            "prompt": prompt,
                            "stream": False
                        },
                        timeout=180
                    )

                    response.raise_for_status()
                    answer = response.json().get("response", "").strip()

                if answer:
                    st.success("Answer generated successfully!")
                    st.markdown(answer)
                else:
                    st.warning("The model returned an empty answer.")

                # --------------------------------------
                # DISPLAY SOURCES
                # --------------------------------------

                st.subheader("📌 Sources")

                seen_sources = set()

                for metadata in retrieved_metadata:
                    metadata = metadata or {}

                    source = metadata.get("source", "Unknown document")
                    page = metadata.get("page", "Unknown")

                    source_label = f"{source} — Page {page}"

                    if source_label not in seen_sources:
                        st.write(f"- {source_label}")
                        seen_sources.add(source_label)

                st.caption(
                    f"Answer generated using {len(retrieved_chunks)} "
                    f"retrieved document chunks."
                )

        except requests.exceptions.RequestException as e:
            st.error(
                "Could not connect to Ollama. Make sure Ollama is running "
                f"and the model '{OLLAMA_MODEL}' is available.\n\nDetails: {e}"
            )

        except Exception as e:
            st.error(f"Something went wrong: {e}")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Powered by Streamlit | ChromaDB | Sentence Transformers | Ollama"
)