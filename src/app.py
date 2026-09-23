import streamlit as st

from rag_pipeline import generate_answer


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Enterprise Knowledge Assistant",
    page_icon="📚",
    layout="wide"
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("📚 Enterprise Knowledge Assistant")

st.write(
    "Ask questions about the company knowledge base. "
    "Answers are generated using retrieved documents."
)


# --------------------------------------------------
# Question input
# --------------------------------------------------

query = st.text_input(
    "Ask a question",
    placeholder="Example: How many days of annual leave does a full-time employee receive?"
)


# --------------------------------------------------
# Ask button
# --------------------------------------------------

if st.button("Ask Question"):

    if not query.strip():

        st.warning("Please enter a question.")

    else:

        with st.spinner("Searching the knowledge base..."):

            answer, results, is_supported = generate_answer(query)

        # --------------------------------------------------
        # Answer
        # --------------------------------------------------

        st.subheader("Answer")

        st.write(answer)

        # --------------------------------------------------
        # Sources
        # --------------------------------------------------

        st.subheader("Sources")

        if is_supported and results:
            displayed_sources = set()

            for document, score in results:
                source = document.metadata["source"]

                if source not in displayed_sources:
                    st.write(f"📄 {source}")
                    displayed_sources.add(source)
        else:
            st.write("No supporting sources found.")