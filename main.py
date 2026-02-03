import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# We import the Agent toolkit
from langchain_classic.agents import initialize_agent, AgentType
from langchain_core.tools import Tool

# Import the custom tool we just made
from tools import create_it_ticket

load_dotenv()

app = FastAPI()

# --- 1. Setup Resources ---
llm = ChatGoogleGenerativeAI(model="gemini-flash-latest")
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# Load DB
vector_db = FAISS.load_local(
    "faiss_index", embeddings, allow_dangerous_deserialization=True
)
retriever = vector_db.as_retriever()

# --- 2. Define the Toolkit ---
# We wrap our capabilities as "Tools" so the LLM understands how to use them.


# Tool 1: The RAG System
# Created a mini-function to wrap the retrieval
def search_knowledge_base(query: str):
    docs = retriever.invoke(query)
    return "\n\n".join([d.page_content for d in docs])


knowledge_tool = Tool(
    name="KnowledgeBase",
    func=search_knowledge_base,
    description="Use this to answer questions about company policies, holidays, or general info.",
)

# Tool 2: The Ticket System
ticket_tool = Tool(
    name="IT_Ticket_Creator",
    func=create_it_ticket,
    description="Use this ONLY when the user explicitly asks to create a ticket.",
    return_direct=True,
)

tools = [knowledge_tool, ticket_tool]

# --- 3. Initialize the Agent ---
# ZERO_SHOT_REACT_DESCRIPTION means:
# "Look at the tools, Look at the user query, Reason about which tool to pick."
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,  # This will print the AI's "thought process" in your terminal
    handle_parsing_errors=True,
)


# --- 4. API Endpoint ---
class QueryRequest(BaseModel):
    query: str
    user_id: str = "guest"


@app.post("/chat")
async def chat_endpoint(request: QueryRequest):
    # The agent handles the decision making now
    response = agent.invoke(request.query)

    return {
        "received_query": request.query,
        "ai_response": response["output"],
        "status": "completed",
    }


@app.get("/")
def home():
    return {"message": "Agentic Engine (RAG + Tools) is Online"}
