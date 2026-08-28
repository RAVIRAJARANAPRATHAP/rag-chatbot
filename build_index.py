from pypdf import PdfReader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv
import os
import time

load_dotenv()

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

docs_folder = "docs"
all_chunks = []

for filename in os.listdir(docs_folder):
    if filename.endswith(".pdf"):
        path = os.path.join(docs_folder, filename)
        reader = PdfReader(path)
        for page_num, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            text_chunks = chunk_text(page_text)
            for chunk in text_chunks:
                if chunk.strip():
                    all_chunks.append(Document(
                        page_content=chunk,
                        metadata={"source": filename, "page": page_num}
                    ))
        print(f"Loaded {filename}: {len(reader.pages)} pages")

print(f"\nTotal chunks: {len(all_chunks)}")

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory="chroma_db"
)

batch_size = 20
for i in range(0, len(all_chunks), batch_size):
    batch = all_chunks[i:i + batch_size]
    vectorstore.add_documents(batch)
    print(f"Embedded chunks {i + 1} to {i + len(batch)} of {len(all_chunks)}")
    time.sleep(15)

print("Done! Your chunks are embedded and saved in the chroma_db folder.")