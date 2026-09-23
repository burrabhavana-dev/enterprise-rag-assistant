from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from rag_pipeline import generate_answer


TEST_CASES = [
    {
        "question": "How many annual leave days are provided?",
        "expected_source": "leave_policy.txt",
        "answerable": True,
    },
    {
        "question": "How far in advance should leave be requested?",
        "expected_source": "leave_policy.txt",
        "answerable": True,
    },
    {
        "question": "Can emergency leave be requested with less notice?",
        "expected_source": "leave_policy.txt",
        "answerable": True,
    },
    {
        "question": "Can employees work remotely?",
        "expected_source": "remote_work_policy.txt",
        "answerable": True,
    },
    {
        "question": "What is required for remote work from another state?",
        "expected_source": "remote_work_policy.txt",
        "answerable": True,
    },
    {
        "question": "When can new employees enroll in health insurance?",
        "expected_source": "health_insurance_policy.txt",
        "answerable": True,
    },
    {
        "question": "What documentation is required for dependents?",
        "expected_source": "health_insurance_policy.txt",
        "answerable": True,
    },
    {
        "question": "How soon must travel expenses be submitted?",
        "expected_source": "travel_policy.txt",
        "answerable": True,
    },
    {
        "question": "What expenses are not reimbursable?",
        "expected_source": "expense_policy.txt",
        "answerable": True,
    },
    {
        "question": "Who approves expense reports?",
        "expected_source": "expense_policy.txt",
        "answerable": True,
    },
    {
        "question": "What is parental leave policy?",
        "expected_source": None,
        "answerable": False,
    },
    {
        "question": "What is dental insurance policy?",
        "expected_source": None,
        "answerable": False,
    },
    {
        "question": "What is employee stock option policy?",
        "expected_source": None,
        "answerable": False,
    },
    {
        "question": "How many sick days are provided?",
        "expected_source": None,
        "answerable": False,
    },
    {
        "question": "What is maternity leave duration?",
        "expected_source": None,
        "answerable": False,
    },
]


REFUSAL_TEXT = "I could not find sufficient information in the knowledge base."


def main():
    total = len(TEST_CASES)
    answerable_total = sum(case["answerable"] for case in TEST_CASES)
    unanswerable_total = total - answerable_total

    retrieval_hits = 0
    answerable_correct = 0
    unsupported_rejections = 0

    print("=" * 90)
    print("ENTERPRISE RAG EVALUATION")
    print("=" * 90)

    for index, case in enumerate(TEST_CASES, start=1):

        question = case["question"]

        answer, results, is_supported = generate_answer(question)

        sources = []
        for document, score in results:
            source = document.metadata["source"]

            if source not in sources:
                sources.append(source)

        expected_source = case["expected_source"]

        if case["answerable"]:
            retrieval_correct = expected_source in sources
            answer_correct = is_supported and retrieval_correct

            if retrieval_correct:
                retrieval_hits += 1

            if answer_correct:
                answerable_correct += 1

        else:
            retrieval_correct = len(sources) == 0
            answer_correct = not is_supported

            if answer_correct:
                unsupported_rejections += 1

        print()
        print("-" * 90)
        print(f"TEST {index}")
        print(f"Question: {question}")
        print(f"Expected source: {expected_source}")
        print(f"Retrieved sources: {sources}")
        print(f"Supported: {is_supported}")
        print(f"Answer: {answer}")

    retrieval_hit_rate = (
        retrieval_hits / answerable_total * 100
        if answerable_total
        else 0
    )

    answerable_accuracy = (
        answerable_correct / answerable_total * 100
        if answerable_total
        else 0
    )

    rejection_rate = (
        unsupported_rejections / unanswerable_total * 100
        if unanswerable_total
        else 0
    )

    print()
    print("=" * 90)
    print("EVALUATION SUMMARY")
    print("=" * 90)

    print(f"Total questions: {total}")
    print(f"Answerable questions: {answerable_total}")
    print(f"Unanswerable questions: {unanswerable_total}")
    print(f"Retrieval hit rate: {retrieval_hit_rate:.1f}%")
    print(f"Answerable question accuracy: {answerable_accuracy:.1f}%")
    print(f"Unsupported-query rejection rate: {rejection_rate:.1f}%")
    print("=" * 90)


if __name__ == "__main__":
    main()