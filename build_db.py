import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# from langchain.schema import Documen
from langchain_core.documents import Document

# Load env to get API Key
load_dotenv()

# 1. The "Raw Data"
fake_company_data = [
    "At Jobless Corp, the remote work policy allows 3 days of WFH per week.",
    "The engineering team daily standup is at 10:00 AM EST.",
    "To reset your password, you must submit a ticket to IT-Support via Slack channel #help-desk.",
    "Jobless Corp was founded in 2025 with the mission to simplify AI agents.",
]

documents = [Document(page_content=text) for text in fake_company_data]

# 2. The Embedding Model
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# 3. Create FAISS Database
print("Creating FAISS vector database...")
vector_db = FAISS.from_documents(documents=documents, embedding=embeddings)

# 4. Save it locally
# This creates a folder named "faiss_index"
vector_db.save_local("faiss_index")

print("Database built! Saved to ./faiss_index folder.")
