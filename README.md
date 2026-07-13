# 📚 Research RAG Project

An intelligent **Retrieval-Augmented Generation (RAG)** application that enables users to upload research papers (PDFs) and ask questions in natural language. The system combines **Hybrid Retrieval**, **Cross-Encoder Re-ranking**, and **Google Gemini 2.5 Flash** to generate accurate, context-aware answers grounded in the uploaded documents.

---

## 🚀 Features

- 📄 Upload and process research papers in PDF format
- 📑 Automatic document parsing using Docling
- 🔍 Hybrid Retrieval (Dense + Sparse Search)
- 🎯 Cross-Encoder Re-ranking for improved retrieval accuracy
- 🤖 Context-aware answer generation with Google Gemini 2.5 Flash
- 💬 Interactive Streamlit-based chat interface
- 📊 RAG evaluation using RAGAS metrics
- ⚡ Modular and scalable architecture

---

## 🏗️ Architecture

```
PDF Document
      │
      ▼
Docling Parser
      │
      ▼
Text Chunking
      │
      ▼
Embedding Generation
(Dense + Sparse)
      │
      ▼
Qdrant Vector Store
      │
      ▼
Hybrid Retrieval
      │
      ▼
Cross-Encoder Re-ranking
      │
      ▼
Google Gemini 2.5 Flash
      │
      ▼
Generated Response

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| Language | Python 3.12 |
| Framework | Streamlit, LangChain |
| LLM | Google Gemini 2.5 Flash |
| Vector Database | Qdrant |
| Embeddings | FastEmbed, Sentence Transformers |
| Document Parsing | Docling |
| Evaluation | RAGAS, Langfuse |
| Environment | Python Dotenv |

---

## ⚙️ How It Works

1. Upload a research paper (PDF).
2. Extract and clean document content using Docling.
3. Split the document into meaningful chunks.
4. Generate dense and sparse embeddings.
5. Store embeddings in the Qdrant vector database.
6. Retrieve the most relevant chunks using Hybrid Search.
7. Re-rank retrieved results using a Cross-Encoder.
8. Generate an answer using Google Gemini based on the retrieved context.
9. Evaluate the generated response using RAGAS metrics.

---

## 📊 Evaluation

The project uses **RAGAS** to measure the quality of generated responses.

### Metrics

- Faithfulness
- Answer Relevancy

These metrics help evaluate whether the generated answer is both accurate and relevant to the retrieved document context.

---

## 💻 Installation

### Clone the repository

```bash
git clone https://github.com/anuguhruthikreddy03/research-RAG-project.git
cd research-RAG-project
```

### Create a virtual environment

```bash
python -m venv venv
```

### Activate the environment

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file and add:

```env
GOOGLE_API_KEY=your_google_api_key
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

---

## 👨‍💻 Author

**Hruthik Reddy**

**GitHub:** https://github.com/anuguhruthikreddy03

**LinkedIn:** https://www.linkedin.com/in/hruthikrdy03/

---
## ⭐ Support

If you found this project useful, consider giving it a **⭐ Star** on GitHub.
