from pathlib import Path
from functools import lru_cache

from langchain_community.vectorstores import FAISS
from embeddings import get_embedding_model

BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_STORE_DIR = BASE_DIR / "data" / "vector_store"


@lru_cache(maxsize=1)
def load_vector_store():
    embeddings = get_embedding_model()

    vector_store = FAISS.load_local(
        str(VECTOR_STORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vector_store

def search_documents(query, k=3, score_threshold=1.5):
    """
    Retrieve candidate document chunks using FAISS and rerank them
    using simple keyword overlap.
    """

    vector_store = load_vector_store()

    # Retrieve a larger candidate set from FAISS
    results = vector_store.similarity_search_with_score(
        query,
        k=5
    )

    # Keep candidates within the FAISS distance threshold
    filtered_results = [
        (document, score)
        for document, score in results
        if score <= score_threshold
    ]

    # Normalize query terms
    query_terms = set(
        word.lower().strip("?,.!:")
        for word in query.split()
        if len(word) > 2
    )

    reranked_results = []

    for document, score in filtered_results:

        document_words = set(
            word.lower().strip("?,.!:")
            for word in document.page_content.split()
            if len(word) > 2
        )

        keyword_overlap = len(query_terms.intersection(document_words))

        reranked_results.append(
            (
                document,
                score,
                keyword_overlap
            )
        )

    # First prioritize keyword relevance,
    # then use FAISS similarity as the tie-breaker.
    reranked_results.sort(
        key=lambda item: (-item[2], item[1])
    )

    return [
        (document, score)
        for document, score, keyword_overlap in reranked_results[:k]
    ]


if __name__ == "__main__":

    query = "How many days of annual leave does a full-time employee receive?"

    results = search_documents(query, k=3)

    print()
    print("=" * 70)
    print("USER QUERY")
    print("=" * 70)
    print(query)

    print()
    print("=" * 70)
    print("RETRIEVED DOCUMENTS")
    print("=" * 70)

    for i, (document, score) in enumerate(results, start=1):

        print()
        print(f"RESULT {i}")
        print(f"Source: {document.metadata['source']}")
        print(f"Score: {score}")
        print("-" * 70)
        print(document.page_content)