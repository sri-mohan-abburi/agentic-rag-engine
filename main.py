import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA

load_dotenv()

app = FastAPI()

# --- 1. Setup Resources ---

llm = ChatGoogleGenerativeAI(model="gemini-flash-latest")

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# LOAD FAISS DATABASE
# Note: allow_dangerous_deserialization is required for local files in newer versions.
# In a real enterprise app, ensure you trust the file source (we built it, so we trust it).
vector_db = FAISS.load_local(
    "faiss_index", embeddings, allow_dangerous_deserialization=True
)

retriever = vector_db.as_retriever(search_kwargs={"k": 2})

qa_chain = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
)


# --- 2. Data Contract ---
class QueryRequest(BaseModel):
    query: str
    user_id: str = "guest"


# --- 3. Endpoints ---
@app.post("/chat")
async def chat_endpoint(request: QueryRequest):
    result = qa_chain.invoke({"query": request.query})

    return {
        "received_query": request.query,
        "ai_response": result["result"],
        "source_documents": [doc.page_content for doc in result["source_documents"]],
    }


@app.get("/")
def home():
    return {"message": "RAG Agent (FAISS Edition) is Online"}
