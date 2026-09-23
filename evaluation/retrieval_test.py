from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from retriever import load_vector_store


QUESTIONS = [
    ("How many annual leave days are provided?", "ANSWERABLE"),
    ("How far in advance should leave be requested?", "ANSWERABLE"),
    ("Can emergency leave be requested with less notice?", "ANSWERABLE"),
    ("Can employees work remotely?", "ANSWERABLE"),
    ("What is required for remote work from another state?", "ANSWERABLE"),
    ("When can new employees enroll in health insurance?", "ANSWERABLE"),
    ("What documentation is required for dependents?", "ANSWERABLE"),
    ("How soon must travel expenses be submitted?", "ANSWERABLE"),
    ("What expenses are not reimbursable?", "ANSWERABLE"),
    ("Who approves expense reports?", "ANSWERABLE"),
    ("What is parental leave policy?", "UNANSWERABLE"),
    ("What is dental insurance policy?", "UNANSWERABLE"),
    ("What is employee stock option policy?", "UNANSWERABLE"),
    ("How many sick days are provided?", "UNANSWERABLE"),
    ("What is maternity leave duration?", "UNANSWERABLE"),
]


def main():
    vector_store = load_vector_store()

    print("=" * 90)
    print("RAG RETRIEVAL EVALUATION")
    print("=" * 90)

    for number, (question, expected) in enumerate(QUESTIONS, start=1):

        results = vector_store.similarity_search_with_score(
            question,
            k=3
        )

        print()
        print("-" * 90)
        print(f"TEST {number}")
        print(f"Expected: {expected}")
        print(f"Question: {question}")
        print("-" * 90)

        for rank, (document, score) in enumerate(results, start=1):
            source = document.metadata["source"]

            print(
                f"{rank}. {source:<35} "
                f"Score: {score:.4f}"
            )


if __name__ == "__main__":
    main()