import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(page_title="DocuMind AI", page_icon="🧠", layout="wide")

st.markdown("""
<style>
.stChatMessage { border-radius: 12px; padding: 8px; }
[data-testid="stCaptionContainer"] { color: #9CA3AF; font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("🧠 DocuMind AI")
    st.markdown("An AI assistant that answers questions from your documents, with source citations.")
    st.divider()
    st.subheader("📄 Loaded Documents")
    for f in os.listdir("docs"):
        if f.endswith(".pdf"):
            st.markdown(f"- {f}")
    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

st.title("🧠 DocuMind AI")
st.caption("Ask questions and get answers grounded in your documents — with sources.")

@st.cache_resource
def load_chain():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = Chroma(embedding_function=embeddings, persist_directory="chroma_db")
    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
    return vectorstore, llm

vectorstore, llm = load_chain()

def answer_question(question):
    greetings = ["hi", "hello", "hey", "how are you", "good morning", "good evening"]
    if question.lower().strip("?! ") in greetings:
        return "Hello! Ask me anything — I'll check my documents first, and use my general knowledge if the answer isn't there.", set()

    results = vectorstore.similarity_search(question, k=6)
    context = "\n\n".join([doc.page_content for doc in results])

    sources = set()
    for doc in results:
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        sources.add(f"{src}, page {page}")

    prompt = f"""You have two sources of knowledge: the context below (from the user's documents) and your own general knowledge.

Instructions:
1. First, try to answer using the context below.
2. If the context contains a relevant answer, use it.
3. If the context does NOT contain a relevant answer, answer using your own general knowledge instead.
4. At the very end of your answer, on a new line, write exactly one of these tags: [SOURCE: DOCUMENT] or [SOURCE: GENERAL]

Context from documents:
{context}

Question: {question}

Answer:"""

    response = llm.invoke(prompt)
    answer_text = response.content[0]["text"] if isinstance(response.content, list) else response.content

    used_document = "[SOURCE: DOCUMENT]" in answer_text
    answer_text = answer_text.replace("[SOURCE: DOCUMENT]", "").replace("[SOURCE: GENERAL]", "").strip()

    if not used_document:
        sources = set()

    return answer_text, sources

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

question = st.chat_input("Ask a question about your documents...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, sources = answer_question(question)
            st.write(answer)
            if sources:
                st.caption("📚 Sources: " + ", ".join(sorted(sources)))

    st.session_state.messages.append({"role": "assistant", "content": answer})