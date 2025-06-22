from fastapi import FastAPI, HTTPException, Request, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, validator
from typing import List, Optional
import os
from datetime import datetime
from pymongo import MongoClient
import redis
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.chat_models import ChatOllama
import tempfile
import subprocess
import json
from bson import ObjectId
from bson.json_util import loads, dumps

# Initialize FastAPI
app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup directories
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
OLLAMA_MODEL = "llama3"

# Database connections
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["chat_db"]
messages_collection = db["messages"]

# Redis connection
try:
    redis_client = redis.Redis.from_url(REDIS_URL)
    redis_client.ping()
except redis.ConnectionError:
    print("Using in-memory cache instead of Redis")
    from fakeredis import FakeRedis
    redis_client = FakeRedis()

# RAG setup
embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)
llm = ChatOllama(model="llama3", temperature=0.7)
vectorstore = None
qa_chain = None

# Models with complete serialization handling
class Message(BaseModel):
    text: str
    sender: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    id: Optional[str] = Field(None, alias="_id")

    @validator('timestamp', pre=True)
    def parse_timestamp(cls, v):
        if isinstance(v, dict) and '$date' in v:
            return datetime.fromisoformat(v['$date'].replace('Z', '+00:00'))
        return v

    @validator('id', pre=True)
    def parse_id(cls, v):
        if isinstance(v, dict) and '$oid' in v:
            return v['$oid']
        if isinstance(v, ObjectId):
            return str(v)
        return v

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            ObjectId: lambda v: str(v)
        }
        allow_population_by_field_name = True

class ChatResponse(BaseModel):
    messages: List[Message]
    ai_response: Optional[str] = None

def initialize_rag():
    global vectorstore, qa_chain
    try:
        loader = PyPDFLoader("https://arxiv.org/pdf/1706.03762.pdf")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(texts, embeddings)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )
        print("RAG initialized successfully")
    except Exception as e:
        print(f"RAG initialization error: {str(e)}")
        qa_chain = None

initialize_rag()

def serialize_for_storage(data: dict) -> str:
    """Convert MongoDB document to JSON string with proper serialization"""
    return dumps(data)

def deserialize_from_storage(data: bytes) -> dict:
    """Convert stored JSON back to dict with MongoDB format handling"""
    try:
        return loads(data.decode('utf-8'))
    except Exception:
        return {"text": "Error decoding message", "sender": "system"}

@app.post("/chat", response_model=ChatResponse)
async def chat(message: Message):
    try:
        # Prepare message without id for MongoDB
        message_dict = message.dict(exclude={'id'})
        
        # MongoDB insertion
        result = messages_collection.insert_one(message_dict)
        message_dict['_id'] = result.inserted_id
        
        # Redis storage with proper serialization
        redis_key = f"recent_messages:{message.sender}"
        redis_client.lpush(redis_key, serialize_for_storage(message_dict))
        redis_client.ltrim(redis_key, 0, 4)
        
        # Generate AI response if needed
        ai_response = None
        if message.sender == "user" and qa_chain:
            try:
                result = qa_chain({"query": message.text})
                ai_response = result.get("result", "I couldn't generate a response")
                
                # Store AI response
                ai_message = Message(
                    text=ai_response,
                    sender="ai"
                )
                ai_dict = ai_message.dict(exclude={'id'})
                messages_collection.insert_one(ai_dict)
                redis_client.lpush(redis_key, serialize_for_storage(ai_dict))
                
            except Exception as e:
                ai_response = f"AI service error: {str(e)}"
        
        # Retrieve and format messages
        recent_messages = [
            Message.parse_obj(deserialize_from_storage(msg))
            for msg in redis_client.lrange(redis_key, 0, -1)
        ]
        
        return ChatResponse(
            messages=recent_messages,
            ai_response=ai_response
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/messages", response_model=List[Message])
async def get_messages(limit: int = 10):
    try:
        messages = list(messages_collection.find().sort("timestamp", -1).limit(limit))
        return [Message.parse_obj(msg) for msg in messages]
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)