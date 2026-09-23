# Enterprise Knowledge RAG Assistant

An enterprise knowledge assistant that uses Retrieval-Augmented Generation (RAG) to answer questions from internal policy documents while rejecting questions that cannot be supported by the knowledge base.

## Overview

This project demonstrates an end-to-end local RAG pipeline for enterprise document question answering.

The system:

- Ingests enterprise policy documents
- Splits documents into retrieval-friendly chunks
- Generates semantic embeddings using Sentence Transformers
- Stores embeddings in a FAISS vector database
- Retrieves relevant document chunks for user questions
- Reranks retrieved candidates using keyword relevance
- Uses the local Phi-3 LLM through Ollama to generate grounded answers
- Displays supporting document sources
- Rejects unsupported questions instead of generating unsupported answers

## Architecture

User Question
    |
    v
FAISS Semantic Retrieval
    |
    v
Candidate Reranking
    |
    v
Relevant Context
    |
    v
Phi-3 Local LLM
    |
    v
Grounded Answer
    |
    v
Source Attribution

## Technology Stack

Python
LangChain
FAISS
Sentence Transformers
Hugging Face
Ollama
Phi-3
Streamlit
FastAPI
PyPDF
Pandas
NumPy

## Project Structure

enterprise-rag-assistant/
|
├── data/
│   ├── documents/
│   │   ├── leave_policy.txt
│   │   ├── remote_work_policy.txt
│   │   ├── health_insurance_policy.txt
│   │   ├── travel_policy.txt
│   │   └── expense_policy.txt
│   │
│   └── vector_store/
│       ├── index.faiss
│       └── index.pkl
|
├── src/
│   ├── __init__.py
│   ├── ingestion.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── rag_pipeline.py
│   └── app.py
|
├── evaluation/
│   ├── retrieval_test.py
│   └── evaluate_rag.py
|
├── .gitignore
├── README.md
└── requirements.txt

## Knowledge Base

The current demonstration knowledge base contains five enterprise policy documents:

- Employee Leave Policy
- Remote Work Policy
- Health Insurance Policy
- Business Travel Policy
- Employee Expense Policy

## Retrieval

Documents are converted into semantic embeddings using:

`sentence-transformers/all-MiniLM-L6-v2`

FAISS is used for vector similarity search.

The retrieval pipeline combines semantic similarity with lightweight keyword-based reranking to improve ordering of retrieved candidates.

## Generation

The application uses Phi-3 locally through Ollama.

The LLM is instructed to:

- Use only retrieved knowledge-base information
- Avoid unsupported assumptions
- Reject questions that cannot be answered from the knowledge base
- Avoid generating unsupported information

## Evaluation

The system was evaluated using a 15-question test set containing:

- 10 answerable questions
- 5 unanswerable questions

Evaluation results:

| Metric | Result |
|---|---:|
| Total questions | 15 |
| Answerable questions | 10 |
| Unanswerable questions | 5 |
| Retrieval hit rate | 100% |
| Answerable question accuracy | 100% |
| Unsupported-query rejection rate | 100% |

The evaluation includes questions covering leave, remote work, health insurance, travel, and expense policies, along with unsupported questions such as parental leave, dental insurance, stock option policies, sick days, and maternity leave.

## Example

Question:

"What documentation is required for dependents?"

Answer:

"Employees must provide the required enrollment information and dependent documentation when adding eligible dependents."

Source:

`health_insurance_policy.txt`

For unsupported questions such as:

"What is dental insurance policy?"

the system responds:

"I could not find sufficient information in the knowledge base."

## Running the Project

### 1. Create the virtual environment
python -m venv .venv

### 2. Activate the virtual environment
For Windows Command Prompt:
.venv\Scripts\activate

### 3. Install dependencies
pip freeze > requirements.txt
pip install -r requirements.txt

### 4. Install and configure Ollama
ollama pull phi3
ollama list

### 5. Prepare the knowledge base

The demonstration project includes:
data/documents/
├── leave_policy.txt
├── remote_work_policy.txt
├── health_insurance_policy.txt
├── travel_policy.txt
└── expense_policy.txt

### 6. Build the FAISS vector store
python src\vector_store.py

### 7. Run the Streamlit application
streamlit run src\app.py

What documentation is required for dependents?
Example supported questions:
How many days of annual leave does a full-time employee receive?
How far in advance should leave be requested?
Can employees work remotely?
What is required for remote work from another state?
What documentation is required for dependents?
The application retrieves relevant policy information and generates a grounded answer using the local Phi-3 model.

### 9. Test unsupported questions
Example unsupported questions:
What is parental leave policy?
What is dental insurance policy?
What is employee stock option policy?
How many sick days are provided?
What is maternity leave duration?

For unsupported questions, the application responds:
I could not find sufficient information in the knowledge base.
No supporting sources are displayed when the system determines that the retrieved context does not sufficiently support an answer.

### 10. Run the retrieval evaluation
To inspect the raw retrieval behavior, run:
python evaluation\retrieval_test.py

This displays:
Evaluation question
Expected question type
Top retrieved documents
FAISS similarity scores

### 11. Run the complete RAG evaluation
python evaluation\evaluate_rag.py
The evaluation script tests both answerable and unanswerable questions and reports:
Total questions
Answerable questions
Unanswerable questions
Retrieval hit rate
Answerable question accuracy
Unsupported-query rejection rate

### 12. Evaluation Results
The current evaluation uses a 15-question test set:

10 answerable questions
5 unanswerable questions

## Key Engineering Challenges

### Semantic Retrieval

Some questions contain terms that appear across multiple policy documents. The system combines FAISS semantic retrieval with lightweight keyword-based reranking to improve candidate ordering.

### Unsupported Questions

The system must distinguish between questions that can be answered from the knowledge base and questions outside its scope. The generation pipeline instructs the LLM to reject unsupported questions rather than generate unsupported information.

### Retrieval Evaluation

A separate evaluation pipeline was implemented to test retrieval and answer-generation behavior using both answerable and unanswerable questions.