from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory="chroma_db"
)

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

def answer_question(question):
    results = vectorstore.similarity_search(question, k=4)
    context = "\n\n".join([doc.page_content for doc in results])

    prompt = f"""Answer the question using only the context below. If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:"""

    response = llm.invoke(prompt)
    if isinstance(response.content, list):
        return response.content[0]["text"]
    return response.content

if __name__ == "__main__":
    while True:
        question = input("\nAsk a question (or type 'exit'): ")
        if question.lower() == "exit":
            break
        answer = answer_question(question)
        print(f"\nAnswer: {answer}")