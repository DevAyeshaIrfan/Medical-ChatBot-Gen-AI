import os
from uuid import uuid4

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from src.helper import (
    load_pdf_file,
    filter_to_minimal_docs,
    text_split,
    download_hugging_face_embeddings,
)


def create_pinecone_index(
    data_path="Data/",
    index_name="medical-chatbot",
    namespace="book-1",
    dimension=384,
    metric="cosine",
    cloud="aws",
    region="us-east-1",
    batch_size=100
):
    load_dotenv()

    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise ValueError("PINECONE_API_KEY not found in .env")

    # 1) Load and preprocess documents
    extracted_data = load_pdf_file(data=data_path)
    filtered_data = filter_to_minimal_docs(extracted_data)
    text_chunks = text_split(filtered_data, chunk_size=500, chunk_overlap=20)

    if not text_chunks:
        raise ValueError("No text chunks were created.")

    print(f"Raw documents loaded: {len(extracted_data)}")
    print(f"Filtered documents: {len(filtered_data)}")
    print(f"Text chunks created: {len(text_chunks)}")

    # 2) Load embeddings
    embeddings = download_hugging_face_embeddings()

    # 3) Connect to Pinecone
    pc = Pinecone(api_key=pinecone_api_key)

    # 4) Create index if it does not exist
    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        print(f"Created index: {index_name}")
    else:
        print(f"Index already exists: {index_name}")

    # 5) Initialize vector store
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        namespace=namespace,
    )

    # 6) Add documents in batches
    total = len(text_chunks)

    for start in range(0, total, batch_size):
        batch_docs = text_chunks[start:start + batch_size]
        batch_ids = [str(uuid4()) for _ in batch_docs]

        vectorstore.add_documents(
            documents=batch_docs,
            ids=batch_ids
        )

        end = min(start + batch_size, total)
        print(f"Upserted chunks {start + 1} to {end} of {total}")

    print("\nIndexing completed successfully.")
    return vectorstore


if __name__ == "__main__":
    create_pinecone_index(
        data_path="Data/",
        index_name="medical-chatbot",
        namespace="book-1"
    )