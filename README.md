# 🤖 RAG Document Chatbot

A chatbot that answers questions using the content of your own PDF documents — powered by Retrieval-Augmented Generation (RAG), so answers are grounded in real source material instead of relying purely on the model's general knowledge.

**🔗 Live Demo:** [rag-chatbot.streamlit.app](https://rag-chatbot-wwd6dkfckijt39qulmzm6u.streamlit.app/)

## What it does

Upload PDFs, and the chatbot lets you ask natural-language questions about their content. Instead of hallucinating answers, it retrieves the most relevant sections of the document and uses them as context for generating a response — and honestly says "I don't know" when the answer isn't in the source material.

## How it works

1. **Document loading** — PDFs are parsed and their text extracted using `pypdf`.
2. **Chunking** — Text is split into overlapping chunks so relevant sections can be retrieved independently of document length.
3. **Embeddings** — Each chunk is converted into a vector representation using Google's Gemini embedding model.
4. **Vector storage** — Chunks are stored in a local ChromaDB vector database for fast similarity search.
5. **Retrieval** — When a question is asked, the most semantically similar chunks are retrieved from ChromaDB.
6. **Generation** — Retrieved chunks are passed as context to Gemini, which generates a grounded, source-based answer.

## Tech Stack

- **Python**
- **LangChain** — orchestration for embeddings and retrieval
- **Google Gemini API** — embeddings + chat generation
- **ChromaDB** — local vector database
- **Streamlit** — web interface and deployment

## Running locally

```bash
git clone https://github.com/[your-username]/rag-chatbot.git
cd rag-chatbot
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file with:
