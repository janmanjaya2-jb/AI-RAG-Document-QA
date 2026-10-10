# AI RAG Document Q&A

A Retrieval-Augmented Generation (RAG) application that lets users
upload PDF documents, ask questions about their content, and receive
answers grounded in retrieved document passages.

**Live demo:** https://ai-rag-doc-assistant-jb.streamlit.app/
**GitHub repository:**
https://github.com/janmanjaya2-jb/AI-RAG-Document-QA

## Features

-   Upload one or more PDF documents through a Streamlit interface.
-   Extract text from PDFs using PyMuPDF.
-   Split extracted text into smaller chunks for retrieval.
-   Generate text embeddings using Sentence Transformers.
-   Store and search document embeddings with ChromaDB.
-   Retrieve relevant document chunks using semantic similarity.
-   Generate answers using the Groq API with the `openai/gpt-oss-20b`
    model.
-   Display retrieved context and source information to help users
    verify answers.
-   Respond that information is unavailable when the provided document
    context does not support an answer.

## Technology Stack

-   **Language:** Python
-   **Web interface:** Streamlit
-   **PDF text extraction:** PyMuPDF
-   **Embeddings:** Sentence Transformers
-   **Vector database:** ChromaDB
-   **Answer generation:** Groq API
-   **Language model:** `openai/gpt-oss-20b`

## How the RAG pipeline works

1.  **Upload:** The user uploads a PDF.
2.  **Extract and chunk:** The application extracts the PDF text and
    splits it into smaller chunks.
3.  **Embed and store:** Sentence Transformers converts chunks into
    embeddings, which are stored in ChromaDB.
4.  **Retrieve:** The user's question is embedded and used to find
    relevant chunks.
5.  **Generate:** The retrieved context and question are sent to the
    Groq API.
6.  **Display:** The app displays the generated answer and available
    source information.

## Project structure

``` text
AI-RAG-Document-QA/
├── app.py
├── document_processor.py
├── vector_store.py
├── rag_qa.py
├── requirements.txt
├── README.md
└── .gitignore
```

The deployed app may create runtime data or a vector database directory.
Those generated files should not be committed unless the project
specifically requires them.

## Run locally

### 1. Clone the repository

``` bash
git clone https://github.com/janmanjaya2-jb/AI-RAG-Document-QA.git
cd AI-RAG-Document-QA
```

### 2. Create and activate a virtual environment

Windows PowerShell:

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

### 4. Configure the Groq API key

Create a `.streamlit/secrets.toml` file in the project folder:

``` toml
GROQ_API_KEY = "your_groq_api_key_here"
```

Replace the placeholder with your own Groq API key. **Never commit this
file or publish your API key.** Make sure `.streamlit/secrets.toml` is
excluded by `.gitignore`.

For Streamlit Community Cloud, add `GROQ_API_KEY` in the app's
**Settings → Secrets**.

### 5. Start the app

``` bash
streamlit run app.py
```

## Testing

Use a small PDF with clearly stated facts and test questions whose
answers are present in the document. Also test a question whose answer
is absent. Compare the generated response with the retrieved chunks and
check the displayed source.

Example test document questions: - How many solar panels were installed
in the Sunville project? - What is the rated capacity of each panel? -
What percentage did the project reduce electricity purchases by? - What
is the population of Sunville?

The last two questions test whether the system avoids inventing
information that the document does not provide.

## Security and limitations

-   Keep API keys in Streamlit Secrets or a local secrets file that is
    excluded from Git.
-   Answer quality depends on PDF text extraction, chunking, retrieval
    quality, and the language model's response.
-   Review the retrieved context and cited sources when verifying an
    answer.
-   API availability and rate limits may affect answer generation.

## Author

Janmanjaya Behera
