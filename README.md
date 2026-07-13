# 📚 Research RAG Project
### Intelligent Research Assistant using Hybrid Retrieval, Google Gemini, Docling, Qdrant, and RAGAS

![Python](https://img.shields.io/badge/Python-3.12-blue)
![LangChain](https://img.shields.io/badge/LangChain-LCEL-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-red)
![Gemini](https://img.shields.io/badge/Google-Gemini-orange)
![Qdrant](https://img.shields.io/badge/Qdrant-Hybrid%20Search-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📖 Overview

Research RAG Project is an intelligent document question-answering system that enables users to upload research papers or PDF documents and interact with them through natural language.

The system combines **Retrieval-Augmented Generation (RAG)** with **Hybrid Search**, **Cross-Encoder Re-ranking**, and **Google Gemini 2.5 Flash** to generate highly accurate and context-aware answers.

Unlike traditional chatbots, the application retrieves only the most relevant information from uploaded documents before generating responses, significantly reducing hallucinations and improving answer quality.

---

## 🚀 Features

- 📄 Upload any PDF research paper
- 🧠 Intelligent document parsing using Docling
- 🔍 Hybrid Retrieval (Dense + Sparse Search)
- ⚡ Cross Encoder Re-ranking for better retrieval quality
- 🤖 Google Gemini 2.5 Flash for answer generation
- 💬 Interactive Streamlit Chat Interface
- 📊 Automated RAG Evaluation using RAGAS
- 📈 Faithfulness and Answer Relevancy Metrics
- 🔄 Dynamic Question Generation for Evaluation
- ⚙️ Modular Project Architecture
- 🚀 Production-ready Retrieval Pipeline

---

# 🏗️ System Architecture

```
                    PDF Document
                          │
                          ▼
                 Docling Document Parser
                          │
                          ▼
                  Clean Text Chunks
                          │
                          ▼
             Dense + Sparse Embeddings
                          │
                          ▼
              Qdrant Hybrid Vector Store
                          │
                Similarity Search
                          │
                          ▼
           Cross Encoder Re-ranking
                          │
                          ▼
            Top Relevant Context
                          │
                          ▼
           Google Gemini 2.5 Flash
                          │
                          ▼
              Final Generated Answer
```

---

# 🛠️ Technology Stack

## Programming

- Python 3.12

## Frameworks

- Streamlit
- LangChain
- LCEL

## LLM

- Google Gemini 2.5 Flash

## Embeddings

- sentence-transformers/all-MiniLM-L6-v2

## Vector Database

- Qdrant Vector Store

## Sparse Embedding

- FastEmbed BM25

## Document Parsing

- Docling

## Evaluation

- RAGAS

## Environment Management

- Python Dotenv

---

# ⚙️ Workflow

### Step 1

Upload a PDF document.

↓

### Step 2

Docling extracts structured document content.

↓

### Step 3

Text is cleaned and converted into chunks.

↓

### Step 4

Dense and Sparse embeddings are generated.

↓

### Step 5

Embeddings are stored in an in-memory Qdrant vector database.

↓

### Step 6

User asks a question.

↓

### Step 7

Hybrid Retrieval fetches the most relevant chunks.

↓

### Step 8

Cross Encoder reranks retrieved chunks.

↓

### Step 9

Top-ranked context is passed to Gemini.

↓

### Step 10

Gemini generates an accurate response grounded in the retrieved context.

---

# 📊 RAG Evaluation

The application includes an automated evaluation pipeline using **RAGAS**.

Metrics evaluated:

- Faithfulness
- Answer Relevancy

Evaluation Process

- Automatically generates evaluation questions
- Produces ground truth answers
- Retrieves context
- Generates answers
- Computes evaluation metrics
- Displays results in Streamlit

---

# 📷 Application Features

### 📄 Upload PDF

Upload any research paper or document.

### 💬 Chat Interface

Ask natural language questions.

Example:

```
Summarize the paper.

What methodology is used?

Explain the experimental setup.

What are the limitations?

What future work is suggested?
```

---

### 📊 Evaluation Dashboard

Generate automatic RAG evaluation reports with a single click.

---

# ⚡ Installation

Clone the repository

```bash
git clone https://github.com/anuguhruthikreddy03/research-RAG-project.git

cd research-RAG-project
```

Create virtual environment

```bash
python -m venv venv
```

Activate

Windows

```bash
venv\Scripts\activate
```

Linux/Mac

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file.

```env
GOOGLE_API_KEY=your_google_api_key

LANGFUSE_PUBLIC_KEY=optional

LANGFUSE_SECRET_KEY=optional
```

---

# ▶️ Run the Application

```bash
streamlit run app.py
```

---

# 📦 Main Libraries Used

- LangChain
- LangChain Community
- LangChain Google GenAI
- LangChain HuggingFace
- LangChain Docling
- Qdrant
- FastEmbed
- Streamlit
- RAGAS
- Pandas
- PyPDF
- Python Dotenv

---

# ✨ Key Highlights

- Hybrid Dense + Sparse Retrieval
- Cross Encoder Re-ranking
- Google Gemini Integration
- Modular RAG Architecture
- Automatic Evaluation Pipeline
- Interactive Chat Interface
- Production-ready Code Structure
- High Retrieval Accuracy
- Easy to Extend
- Clean UI

---

# 👨‍💻 Author

**Hruthik Reddy**

B.Tech – Computer Science & Engineering (AI & ML)

** GitHub ** https://github.com/anuguhruthikreddy03

** LinkedIn ** https://www.linkedin.com/in/hruthikrdy03/
---

# ⭐ If you found this project useful

Please consider giving this repository a ⭐ on GitHub.
