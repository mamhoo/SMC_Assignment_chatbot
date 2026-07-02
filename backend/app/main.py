from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from jose import jwt
from datetime import datetime, timedelta
from typing import TypedDict

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langgraph.graph import StateGraph, END

load_dotenv()

app = FastAPI(title="Financial Chatbot - LangGraph")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
SECRET_KEY = os.getenv("SECRET_KEY", "super-secret-key")

def create_access_token(data: dict):
    expire = datetime.utcnow() + timedelta(minutes=120)
    to_encode = data.copy()
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm="HS256")

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    if form_data.username == "demo" and form_data.password == "demo123":
        return {"access_token": create_access_token({"sub": form_data.username}), "token_type": "bearer"}
    raise HTTPException(status_code=401, detail="Invalid credentials")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=os.getenv("OPENAI_API_KEY"))

sql_db = SQLDatabase.from_uri("postgresql://postgres:postgres@localhost:5432/financial_db")
toolkit = SQLDatabaseToolkit(db=sql_db, llm=llm)
sql_agent = create_sql_agent(llm=llm, toolkit=toolkit, verbose=True)

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
if "financial-10k" not in pc.list_indexes().names():
    pc.create_index(
        name="financial-10k",
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

vectorstore = PineconeVectorStore(
    index_name="financial-10k",
    embedding=embeddings,
    pinecone_api_key=os.getenv("PINECONE_API_KEY")
)

class AgentState(TypedDict):
    question: str
    route: str
    answer: str
    source: str

def router_node(state: AgentState):
    prompt = f"Classify as SQL or RAG: {state['question']}"
    cls = llm.invoke(prompt).content.strip().upper()
    return {"route": "sql" if "SQL" in cls else "rag"}

def sql_node(state: AgentState):
    result = sql_agent.invoke({"input": state["question"]})
    return {"answer": result["output"], "source": "SQL Database"}

def rag_node(state: AgentState):
    docs = vectorstore.similarity_search(state["question"], k=4)
    context = "\n".join([doc.page_content for doc in docs])
    result = llm.invoke(f"Context: {context}\nQuestion: {state['question']}\nAnswer:")
    return {"answer": result.content, "source": "Pinecone 10-K RAG"}

workflow = StateGraph(AgentState)
workflow.add_node("router", router_node)
workflow.add_node("sql", sql_node)
workflow.add_node("rag", rag_node)

workflow.set_entry_point("router")
workflow.add_conditional_edges("router", lambda x: x["route"], {"sql": "sql", "rag": "rag"})
workflow.add_edge("sql", END)
workflow.add_edge("rag", END)

agent_graph = workflow.compile()

class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
async def chat(request: ChatRequest, token: str = Depends(oauth2_scheme)):
    try:
        jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    except:
        raise HTTPException(status_code=401)
    
    try:
        result = agent_graph.invoke({"question": request.question})
        return {"answer": result["answer"], "source": result["source"]}
    except Exception as e:
        return {"answer": f"Error processing question: {str(e)}", "source": "Error"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
