from fastapi import FastAPI, HTTPException, Request, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional
import os
from datetime import datetime
from pymongo import MongoClient
import redis
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
import tempfile
import subprocess

# Initializing FastAPI
app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Creating directories
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Mounting static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
OLLAMA_MODEL = "llama3"  # or "mistral", "gemma"

# Database connections
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["chat_db"]
messages_collection = db["messages"]
redis_client = redis.Redis.from_url(REDIS_URL)

# Starting Ollama service if not running
try:
    redis_client.ping()
except redis.ConnectionError:
    print("Redis not running, using in-memory cache")

try:
    # Checking if Ollama is running
    subprocess.run(["ollama", "list"], check=True, capture_output=True)
except (subprocess.CalledProcessError, FileNotFoundError):
    print("Starting Ollama service...")
    subprocess.Popen(["ollama", "serve"])

# RAG setup
embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)
llm = Ollama(model=OLLAMA_MODEL, temperature=0.7)
vectorstore = None
qa_chain = None

# Models
class Message(BaseModel):
    text: str
    sender: str  # "user" or "ai"
    timestamp: Optional[datetime] = None

class ChatResponse(BaseModel):
    messages: List[Message]
    ai_response: Optional[str] = None

def initialize_rag():
    global vectorstore, qa_chain
    try:
        # Loading a sample PDF
        loader = PyPDFLoader("https://arxiv.org/pdf/1706.03762.pdf")
        documents = loader.load()
        
        # Spliting documents
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        
        # Creating a vectorstore
        vectorstore = Chroma.from_documents(texts, embeddings)
        
        # Creating a QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )
        print("RAG initialized successfully with Ollama")
    except Exception as e:
        print(f"Error initializing RAG: {str(e)}")
        qa_chain = None

initialize_rag()

# API Endpoints
@app.post("/chat", response_model=ChatResponse)
async def chat(message: Message):
    try:
        message.timestamp = datetime.utcnow()
        message_dict = message.dict()
        messages_collection.insert_one(message_dict)
        
        redis_key = f"recent_messages:{message.sender}"
        redis_client.lpush(redis_key, str(message_dict))
        redis_client.ltrim(redis_key, 0, 4)
        
        ai_response = None
        if message.sender == "user" and qa_chain:
            try:
                result = qa_chain({"query": message.text})
                ai_response = result.get("result", "I couldn't generate a response")
            except Exception as e:
                ai_response = f"AI service error: {str(e)}"
            
            ai_message = Message(
                text=ai_response,
                sender="ai",
                timestamp=datetime.utcnow()
            )
            messages_collection.insert_one(ai_message.dict())
            redis_client.lpush(redis_key, str(ai_message.dict()))
            redis_client.ltrim(redis_key, 0, 4)
        
        recent_messages = [
            eval(msg.decode("utf-8"))
            for msg in redis_client.lrange(redis_key, 0, -1)
        ]
        
        return {"messages": recent_messages, "ai_response": ai_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/messages", response_model=List[Message])
async def get_messages(limit: int = 10):
    try:
        messages = list(messages_collection.find().sort("timestamp", -1).limit(limit))
        for msg in messages:
            msg["_id"] = str(msg["_id"])
        return messages
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        
        global vectorstore, qa_chain
        vectorstore = Chroma.from_documents(texts, embeddings)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )
        
        os.unlink(temp_file_path)
        return {"message": "Document processed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    return {"status": "healthy"}