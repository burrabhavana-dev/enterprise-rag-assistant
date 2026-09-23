from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = BASE_DIR / "data" / "documents"


def load_documents():
    """
    Load all TXT documents from the data/documents directory.
    """

    documents = []

    for file_path in DOCUMENTS_DIR.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8")

        document = Document(
            page_content=text,
            metadata={
                "source": file_path.name,
                "file_path": str(file_path)
            }
        )

        documents.append(document)

    return documents


def split_documents(documents):
    """
    Split documents into smaller chunks suitable for retrieval.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)

    return chunks


if __name__ == "__main__":

    documents = load_documents()

    print(f"Documents loaded: {len(documents)}")

    chunks = split_documents(documents)

    print(f"Chunks created: {len(chunks)}")
    print()

    for i, chunk in enumerate(chunks[:5], start=1):
        print("=" * 60)
        print(f"CHUNK {i}")
        print(f"Source: {chunk.metadata['source']}")
        print("-" * 60)
        print(chunk.page_content)