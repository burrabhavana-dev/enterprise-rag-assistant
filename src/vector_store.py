from pathlib import Path

from langchain_community.vectorstores import FAISS

from ingestion import load_documents, split_documents
from embeddings import get_embedding_model


BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_STORE_DIR = BASE_DIR / "data" / "vector_store"


def create_vector_store():
    """
    Load documents, split them into chunks,
    create embeddings, and build a FAISS vector store.
    """

    print("Loading documents...")

    documents = load_documents()

    print(f"Documents loaded: {len(documents)}")

    print("Splitting documents...")

    chunks = split_documents(documents)

    print(f"Chunks created: {len(chunks)}")

    print("Loading embedding model...")

    embeddings = get_embedding_model()

    print("Creating FAISS vector store...")

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    VECTOR_STORE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    vector_store.save_local(
        str(VECTOR_STORE_DIR)
    )

    print("FAISS vector store created successfully.")
    print(f"Saved to: {VECTOR_STORE_DIR}")


if __name__ == "__main__":
    create_vector_store()