import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from tavily import TavilyClient
from dotenv import load_dotenv
from datetime import datetime
from zoneinfo import ZoneInfo
from pypdf import PdfReader
import docx
import base64
import os

load_dotenv()

st.set_page_config(page_title="Aria Assistant", page_icon="✨", layout="wide")

st.markdown("""
<style>
.stChatMessage { border-radius: 12px; padding: 8px; }
[data-testid="stCaptionContainer"] { color: #9CA3AF; font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("✨ Aria Assistant")
    st.markdown("Ask anything. Upload files or images for extra context.")
    st.divider()

    uploaded_files = st.file_uploader(
        "📎 Upload documents or images",
        type=["pdf", "docx", "txt", "png", "jpg", "jpeg"],
        accept_multiple_files=True
    )

    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

st.title("✨ Aria Assistant")
st.caption("General knowledge, live web search, and file/image understanding.")

@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(model="gemini-3.6-flash")

llm = load_llm()
tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

def friendly_error(e):
    msg = str(e).lower()
    if "rate" in msg or "quota" in msg or "429" in msg:
        return "I'm getting a lot of requests right now and hit a temporary rate limit (this app runs on a free tier). Please wait 30-60 seconds and try again."
    return "Something went wrong processing that request. Please try again in a moment."

def extract_text_from_upload(file):
    if file.type == "application/pdf":
        reader = PdfReader(file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    elif file.name.endswith(".docx"):
        doc = docx.Document(file)
        return "\n".join(p.text for p in doc.paragraphs)
    elif file.type == "text/plain":
        return file.read().decode("utf-8")
    return ""

def get_current_time():
    utc_now = datetime.now(ZoneInfo("UTC"))
    zones = {
        "India (IST)": "Asia/Kolkata",
        "New York (EST/EDT)": "America/New_York",
        "London (GMT/BST)": "Europe/London",
        "Tokyo (JST)": "Asia/Tokyo",
    }
    lines = [f"**UTC:** {utc_now.strftime('%A, %B %d, %Y - %I:%M %p')}"]
    for name, zone in zones.items():
        local_time = utc_now.astimezone(ZoneInfo(zone))
        lines.append(f"**{name}:** {local_time.strftime('%A, %B %d, %Y - %I:%M %p')}")
    return "\n\n".join(lines)

def needs_search(question):
    triggers = ["current", "latest", "today", "now", "recent", "price", "who is the", "score", "news", "weather"]
    return any(t in question.lower() for t in triggers)

def answer_question(question, uploaded_files):
    greetings = ["hi", "hello", "hey", "how are you", "good morning", "good evening"]
    if question.lower().strip("?! ") in greetings:
        return "Hello! Ask me anything, upload a file or image, or ask about current events.", []

    time_keywords = ["current time", "what time is it", "time right now", "current date", "today's date", "what day is it"]
    if any(kw in question.lower() for kw in time_keywords):
        return "Here's the current date and time:\n\n" + get_current_time(), []

    image_files = [f for f in (uploaded_files or []) if f.type.startswith("image/")]
    doc_files = [f for f in (uploaded_files or []) if not f.type.startswith("image/")]

    doc_context = ""
    for f in doc_files:
        f.seek(0)
        doc_context += f"\n\n--- Content from {f.name} ---\n{extract_text_from_upload(f)}"

    if image_files:
        content = [{"type": "text", "text": f"{doc_context}\n\nQuestion: {question}" if doc_context else question}]
        for img_file in image_files:
            img_file.seek(0)
            img_bytes = img_file.read()
            b64_img = base64.b64encode(img_bytes).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": f"data:{img_file.type};base64,{b64_img}"
            })
        message = HumanMessage(content=content)
        try:
            response = llm.invoke([message])
            answer_text = response.content[0]["text"] if isinstance(response.content, list) else response.content
        except Exception as e:
            answer_text = friendly_error(e)
        return answer_text, [f.name for f in image_files + doc_files]

    sources = []
    search_context = ""

    if needs_search(question) and not doc_context:
        try:
            search_results = tavily.search(query=question, max_results=4)
            for r in search_results.get("results", []):
                search_context += f"{r['content']}\n\n"
                sources.append(r["url"])
        except Exception:
            pass  # fall back to general knowledge if search fails

    combined_context = doc_context + "\n\n" + search_context

    if combined_context.strip():
        prompt = f"""Answer the question using the context below where relevant, combined with your own knowledge.

Context:
{combined_context}

Question: {question}

Answer clearly and concisely."""
    else:
        prompt = f"Answer this question using your general knowledge: {question}"

    try:
        response = llm.invoke(prompt)
        answer_text = response.content[0]["text"] if isinstance(response.content, list) else response.content
    except Exception as e:
        answer_text = friendly_error(e)

    if doc_files:
        sources = [f.name for f in doc_files] + sources

    return answer_text, sources

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

question = st.chat_input("Ask me anything, or ask about your uploaded files...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, sources = answer_question(question, uploaded_files)
            st.write(answer)
            if sources:
                st.caption("📎 Sources: " + ", ".join(sources))

    st.session_state.messages.append({"role": "assistant", "content": answer})