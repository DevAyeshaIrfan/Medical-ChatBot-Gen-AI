from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


def load_pdf_file(data: str):
    """
    Load all PDF files from a folder and return raw documents.
    """
    folder = Path(data).resolve()

    if not folder.exists():
        raise FileNotFoundError(f"Directory not found: {folder}")

    if not folder.is_dir():
        raise ValueError(f"Expected a directory, got: {folder}")

    pdf_files = list(folder.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {folder}")

    loader = DirectoryLoader(
        str(folder),
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = loader.load()
    return documents


def filter_to_minimal_docs(documents):
    """
    Keep only non-empty text and minimal metadata.
    """
    cleaned_docs = []

    for doc in documents:
        text = (doc.page_content or "").strip()
        if not text:
            continue

        metadata = dict(doc.metadata) if doc.metadata else {}

        minimal_metadata = {}
        if "source" in metadata:
            minimal_metadata["source"] = metadata["source"]
        if "page" in metadata:
            minimal_metadata["page"] = metadata["page"]

        cleaned_docs.append(
            Document(page_content=text, metadata=minimal_metadata)
        )

    return cleaned_docs


def text_split(extracted_data, chunk_size: int = 500, chunk_overlap: int = 20):
    """
    Split documents into chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(extracted_data)


def download_hugging_face_embeddings():
    """
    Load the embedding model used for indexing/retrieval.
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )