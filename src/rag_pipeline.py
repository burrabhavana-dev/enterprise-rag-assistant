from pathlib import Path

from langchain_ollama import OllamaLLM

from retriever import search_documents


def build_context(results):
    """
    Combine retrieved document chunks into a single context.
    """

    context_parts = []

    for i, (document, score) in enumerate(results, start=1):

        source = document.metadata["source"]

        context_parts.append(
            f"[Source {i}: {source}]\n"
            f"{document.page_content}"
        )

    return "\n\n".join(context_parts)


def generate_answer(query):
    """
    Retrieve relevant documents and generate
    a grounded answer using Phi-3.
    """

    results = search_documents(query, k=2, score_threshold=1.5)
    if not results:
        return (
            "I could not find sufficient information in the knowledge base.",
            [],
            False
        )

    context = build_context(results)

    prompt = f"""
You are an enterprise knowledge assistant.

Answer the user's question using ONLY information that directly
supports the question from the provided context.

IMPORTANT:
- Do not answer based only on a shared keyword.
- A document is relevant only if it contains information that
  directly answers the user's question.
- Ignore retrieved documents that are unrelated to the question.
- If none of the retrieved documents directly support the answer,
  say exactly:
  "I could not find sufficient information in the knowledge base."
- Do not invent, assume, or infer information that is not stated
  in the context.
- Do not include source names, citations, or references in your answer.
  The application will display the sources separately.

CONTEXT:
{context}

USER QUESTION:
{query}

ANSWER:
"""

    llm = OllamaLLM(
        model="phi3",
        temperature=0
    )

    response = llm.invoke(prompt)
    is_supported = "I could not find sufficient information" not in response

    return response, results, is_supported
    


if __name__ == "__main__":

    query = "What is the company's parental leave policy?"

    answer, results = generate_answer(query)

    print()
    print("=" * 70)
    print("QUESTION")
    print("=" * 70)
    print(query)

    print()
    print("=" * 70)
    print("GENERATED ANSWER")
    print("=" * 70)
    print(answer)

    print()
    print("=" * 70)
    print("SOURCES")
    print("=" * 70)

    seen_sources = set()

    for document, score in results:

        source = document.metadata["source"]

        if source not in seen_sources:
            print(f"- {source}")
            seen_sources.add(source)