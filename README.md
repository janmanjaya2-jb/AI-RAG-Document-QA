# AI RAG Document Q&A

A Retrieval-Augmented Generation (RAG) application that allows users to ask
questions about PDF documents and receive answers grounded in retrieved
document content.

The application uses semantic search to retrieve relevant document chunks
and a local language model to generate responses.

## Features

- Extracts text from PDF documents
- Splits document text into smaller chunks
- Generates embeddings using Sentence Transformers
- Stores embeddings in ChromaDB
- Retrieves relevant chunks using semantic search
- Generates answers using a local Ollama model
- Provides a Streamlit web interface
- Displays retrieved document context and source information

## Tech Stack

- Python
- Streamlit
- ChromaDB
- Sentence Transformers
- Ollama
- Llama 3.2
- PyMuPDF (PDF text extraction)

## Project Structure

```text
AI-RAG-Document-QA/
│
├── app.py                  # Streamlit user interface
├── document_processor.py   # PDF text extraction and chunking
├── vector_store.py         # Embedding generation and vector storage
├── rag_qa.py               # Retrieval and answer generation
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
│
├── data/                   # Sample PDF documents
└── chroma_db/              # Persistent vector database