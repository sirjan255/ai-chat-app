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
from fastapi import Query
from fastapi.responses import StreamingResponse
from fastapi import Query

# ================== FastAPI App Initialization ==================
# Initialize FastAPI application instance
app = FastAPI()

# ================== CORS Middleware Setup ==================
# Add CORS middleware to allow frontend/backend communication across origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# ================== Static & Template Directories ==================
# Ensure "static" and "templates" directories exist for serving static files and templates
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
# Mount the static directory to serve static files at /static
app.mount("/static", StaticFiles(directory="static"), name="static")
# Setup Jinja2 template rendering for HTML responses
templates = Jinja2Templates(directory="templates")

# ================== Configuration ==================
# Get environment variables or use defaults for MongoDB and Redis connections
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
OLLAMA_MODEL = "llama3"  # Name of LLM model to use with Ollama

# ================== Database Connections ==================
# Connect to MongoDB using pymongo
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["chat_db"]  # Database name: chat_db
messages_collection = db["messages"]  # Collection: messages

# ================== Redis Connection ==================
# Attempt to connect to Redis (for caching/recent messages)
try:
    redis_client = redis.Redis.from_url(REDIS_URL)
    redis_client.ping()  # Test the connection
except redis.ConnectionError:
    # If Redis is not available, fallback to in-memory fake Redis (not persistent)
    print("Using in-memory cache instead of Redis")
    from fakeredis import FakeRedis
    redis_client = FakeRedis()

# ================== RAG (Retrieval-Augmented Generation) Setup ==================
# Embeddings and LLM (Language Model) objects using Ollama
embeddings = OllamaEmbeddings(model=OLLAMA_MODEL)
llm = ChatOllama(model="llama3", temperature=0.7)  # LLM for QA chain
vectorstore = None  # Will hold vector DB for retrieval
qa_chain = None     # Will hold the QA retrieval chain

# ================== Pydantic Models (with Serialization) ==================
class Message(BaseModel):
    text: str  # Message text content
    sender: str  # "user" or "ai"
    timestamp: datetime = Field(default_factory=datetime.utcnow)  # Creation time
    id: Optional[str] = Field(None, alias="_id")  # MongoDB ObjectId as string

    # Validator for timestamp: handles BSON datetime decoding
    @validator('timestamp', pre=True)
    def parse_timestamp(cls, v):
        if isinstance(v, dict) and '$date' in v:
            return datetime.fromisoformat(v['$date'].replace('Z', '+00:00'))
        return v

    # Validator for id: handles MongoDB ObjectId decoding
    @validator('id', pre=True)
    def parse_id(cls, v):
        if isinstance(v, dict) and '$oid' in v:
            return v['$oid']
        if isinstance(v, ObjectId):
            return str(v)
        return v

    class Config:
        # JSON encoders for serialization
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            ObjectId: lambda v: str(v)
        }
        allow_population_by_field_name = True  # Allow using "id" instead of "_id"

# Response model for chat endpoint (list of messages and AI response)
class ChatResponse(BaseModel):
    messages: List[Message]
    ai_response: Optional[str] = None

# ================== RAG Initialization Function ==================
def initialize_rag():
    global vectorstore, qa_chain
    try:
        # Load a public PDF (Google's transformer paper) using PyPDFLoader
        loader = PyPDFLoader("https://arxiv.org/pdf/1706.03762.pdf")
        documents = loader.load()
        # Split the loaded documents into manageable text chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        # Create a Chroma vectorstore for document retrieval
        vectorstore = Chroma.from_documents(texts, embeddings)
        # Setup RetrievalQA chain using the LLM and retriever
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )
        print("RAG initialized successfully")
    except Exception as e:
        print(f"RAG initialization error: {str(e)}")
        qa_chain = None  # Disable QA if initialization fails

# Call RAG initialization at startup
initialize_rag()

# ================== Serialization Utilities ==================
def serialize_for_storage(data: dict) -> str:
    """Convert MongoDB document to JSON string with proper serialization"""
    return dumps(data)  # Use bson.json_util.dumps for MongoDB compatibility

def deserialize_from_storage(data: bytes) -> dict:
    """Convert stored JSON back to dict with MongoDB format handling"""
    try:
        return loads(data.decode('utf-8'))
    except Exception:
        return {"text": "Error decoding message", "sender": "system"}

# ================== API Endpoints ==================

@app.post("/chat", response_model=ChatResponse)
async def chat(message: Message):
    """
    Handles chat messages from the user, stores them in MongoDB and Redis,
    generates an AI response using the QA chain if sender is 'user',
    and returns recent chat history and AI response.
    """
    try:
        # Prepare message dict without 'id' for MongoDB insertion
        message_dict = message.dict(exclude={'id'})
        
        # Insert user's message into MongoDB
        result = messages_collection.insert_one(message_dict)
        message_dict['_id'] = result.inserted_id  # Store inserted ObjectId
        
        # Store the message in Redis (recent history cache)
        redis_key = f"recent_messages:{message.sender}"
        redis_client.lpush(redis_key, serialize_for_storage(message_dict))
        redis_client.ltrim(redis_key, 0, 4)  # Keep only the 5 most recent messages
        
        # AI response (only if sender is user and RAG is available)
        ai_response = None
        if message.sender == "user" and qa_chain:
            try:
                # Generate AI response using RetrievalQA chain
                result = qa_chain({"query": message.text})
                ai_response = result.get("result", "I couldn't generate a response")
                
                # Store AI response as a new message
                ai_message = Message(
                    text=ai_response,
                    sender="ai"
                )
                ai_dict = ai_message.dict(exclude={'id'})
                messages_collection.insert_one(ai_dict)  # Save to MongoDB
                redis_client.lpush(redis_key, serialize_for_storage(ai_dict))  # Save to Redis
                
            except Exception as e:
                ai_response = f"AI service error: {str(e)}"
        
        # Retrieve the recent messages from Redis, deserialize and parse into Message objects
        recent_messages = [
            Message.parse_obj(deserialize_from_storage(msg))
            for msg in redis_client.lrange(redis_key, 0, -1)
        ]
        
        # Return the API response (recent chat history + AI reply)
        return ChatResponse(
            messages=recent_messages,
            ai_response=ai_response
        )
        
    except Exception as e:
        # Catch-all error handling: returns 500 with error message
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/messages", response_model=List[Message])
async def get_messages(limit: int = 10):
    """
    Returns the most recent messages from the MongoDB collection, sorted by timestamp (descending).
    """
    try:
        # Query the last N messages, sort by timestamp (most recent first)
        messages = list(messages_collection.find().sort("timestamp", -1).limit(limit))
        return [Message.parse_obj(msg) for msg in messages]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...)):
    """
    Accepts a PDF upload, processes and splits it, recreates the vectorstore and QA chain for new document context.
    """
    try:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Load and process the PDF as new document context for RAG
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        
        # Update global vectorstore and QA chain to use the uploaded document
        global vectorstore, qa_chain
        vectorstore = Chroma.from_documents(texts, embeddings)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )
        
        # Remove temporary file after processing
        os.unlink(temp_file_path)
        return {"message": "Document processed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def read_root(request: Request):
    """
    Serves the root HTML page using Jinja2 templates (index.html).
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    """
    Health check endpoint for readiness/liveness check.
    """
    return {"status": "healthy"}

@app.get("/chat-summary")
async def chat_summary(limit: int = Query(20, description="Number of messages to summarize")):
    """
    Summarize the most recent N chat messages using the LLM.
    """
    try:
        # Fetch the most recent N messages (sorted by timestamp, oldest to newest)
        messages = list(messages_collection.find().sort("timestamp", -1).limit(limit))
        messages = list(reversed(messages))  # Oldest first

        # Concatenate messages as a single string for summarization
        chat_text = ""
        for msg in messages:
            sender = msg.get("sender", "user")
            text = msg.get("text", "")
            chat_text += f"{sender}: {text}\n"

        if not chat_text.strip():
            return {"summary": "No chat messages available to summarize."}

        # Prompt for the LLM
        prompt = (
            "Summarize the following conversation between a user and an AI assistant:\n\n"
            f"{chat_text}\n\nSummary:"
        )

        # Generate summary using the Ollama LLM directly (using your llm object)
        summary_response = llm.invoke(prompt)
        if isinstance(summary_response, dict) and "content" in summary_response:
            summary = summary_response["content"]
        else:
            summary = str(summary_response)

        return {"summary": summary.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

        
# ================== Export Conversation Endpoint ==================
# Endpoint to export the most recent chat messages as a downloadable text file
@app.get("/export-conversation")
async def export_conversation(
    limit: int = Query(50, description="Number of recent messages to export (default: 50)")
):
    """
    Export the most recent N chat messages as a downloadable text file.
    """
    try:
        # Fetch messages, newest first then reverse for natural order
        messages = list(messages_collection.find().sort("timestamp", -1).limit(limit))
        messages = list(reversed(messages))

        # Build the export content
        export_lines = []
        for msg in messages:
            time_str = msg.get("timestamp")
            if isinstance(time_str, dict) and "$date" in time_str:
                time_str = time_str["$date"]
            export_lines.append(f"[{time_str}] {msg.get('sender', '')}: {msg.get('text', '')}")
        export_text = "\n".join(export_lines)

        # Provide as a downloadable text file
        def file_iterator():
            yield export_text

        return StreamingResponse(
            file_iterator(),
            media_type="text/plain",
            headers={"Content-Disposition": "attachment; filename=conversation_export.txt"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =============== Entrypoint for running with 'python main.py' ===============
if __name__ == "__main__":
    import uvicorn
    # Run the FastAPI app with Uvicorn server, listening on all interfaces at port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)