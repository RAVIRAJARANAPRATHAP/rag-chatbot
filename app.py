import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="My AI Document Chatbot", page_icon="🤖")
st.title("🤖 Ask My Documents")
st.caption("A RAG chatbot that answers questions from my uploaded PDFs")

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
        return "Hello! Ask me anything about the documents I've been given, and I'll do my best to answer from them."

    results = vectorstore.similarity_search(question, k=6)
    context = "\n\n".join([doc.page_content for doc in results])

    prompt = f"""You are answering questions using the context below, which comes from the user's own documents.
Use the context to answer as helpfully as possible, even if the wording doesn't exactly match the question — use your judgment to connect related information.
Only say "I don't know" if the context truly has nothing relevant to the question.

Context:
{context}

Question: {question}

Answer:"""

    response = llm.invoke(prompt)
    if isinstance(response.content, list):
        return response.content[0]["text"]
    return response.content

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
            answer = answer_question(question)
            st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})