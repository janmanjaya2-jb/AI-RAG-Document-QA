
from pypdf import PdfReader

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF document.

    Args:
        pdf_path: Path of the PDF file.

    Returns:
        Extracted text from all pages.
    """

    reader = PdfReader(pdf_path)

    extracted_text = ""

    for page_number, page in enumerate(reader.pages, start=1):

        page_text = page.extract_text() or ""

        extracted_text += f"\n--- Page {page_number} ---\n"
        extracted_text += page_text

    return extracted_text

def chunk_text(text, chunk_size=100, overlap=20):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(words[start:end])

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks

if __name__ == "__main__":

    pdf_path = "data/rag_sample_text_document.pdf"

    # Step 1: Extract text from PDF
    text = extract_text_from_pdf(pdf_path)

    print("PDF TEXT EXTRACTION SUCCESSFUL!")

    # Step 2: Divide extracted text into chunks
    chunks = chunk_text(text)

    print("\nTEXT CHUNKING SUCCESSFUL!")
    print("-------------------------")

    # Step 3: Display total number of chunks
    print("Total chunks:", len(chunks))

    # Step 4: Display first chunk
    print("\nFIRST CHUNK:")
    print(chunks[0])

    # Step 5: Display second chunk if available
    if len(chunks) > 1:
        print("\nSECOND CHUNK:")
        print(chunks[1])