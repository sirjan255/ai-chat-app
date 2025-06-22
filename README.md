# AI Chat Application with FastAPI and Ollama

## 🚀 Complete Setup Guide

A step-by-step guide to set up and run the AI Chat Application with MongoDB, Redis, and Ollama (LLM).

---

## 🔧 Prerequisites

- **Python 3.9+** ([Download Python](https://www.python.org/downloads/))
- **MongoDB Atlas Account** ([Sign up here](https://www.mongodb.com/cloud/atlas/register))
- **Redis** (Cloud or local)
- **Ollama** ([Installation guide](#-ollama-setup))
- **Git** ([Download Git](https://git-scm.com/downloads)) (optional)

---

## 🖥️ Local Development Setup

### 1. Clone the Repository

```bash
# (venv not needed)
git clone https://github.com/yourusername/ai-chat-app.git
cd ai-chat-app
```

### 2. Set Up Python Virtual Environment

```bash
# (venv not needed)
python -m venv venv
```

#### Activate Virtual Environment

- **Windows:**
    ```bash
    venv\Scripts\activate
    ```
- **MacOS/Linux:**
    ```bash
    source venv/bin/activate
    ```

**All following commands should be run with your virtual environment activated!**

### 3. Install Python Dependencies

```bash
# Inside activated venv
pip install -r requirements.txt
```

---

## 🗄️ MongoDB Setup

### 1. Get MongoDB URI from Atlas

1. Go to [MongoDB Atlas](https://www.mongodb.com/cloud/atlas)
2. Create or use an existing project
3. Build a new cluster (Free Tier available)
4. Under "Database Access," create a user with read/write permissions
5. Under "Network Access," add your IP (use `0.0.0.0/0` to allow all IPs for testing)
6. Click "Connect" → "Connect your application"
7. Copy the connection string (URI), it looks like:

   ```
   mongodb+srv://<username>:<password>@cluster0.abc123.mongodb.net/?retryWrites=true&w=majority
   ```

### 2. Set MongoDB URI as Environment Variable

```bash
# Windows (inside venv):
set MONGO_URI=your_mongodb_uri_here

# MacOS/Linux (inside venv):
export MONGO_URI="your_mongodb_uri_here"
```

---

## 🔴 Redis Setup

### Option A: Redis Cloud (Recommended)

1. Sign up at [Redis Cloud](https://redis.com/try-free/)
2. Create a free database
3. Copy the Redis URL (format: `redis://default:<password>@<host>:<port>`)

### Option B: Local Redis

#### Install Redis

- **MacOS (Homebrew):**
    ```bash
    # (venv not needed)
    brew install redis
    brew services start redis
    ```
- **Linux:**
    ```bash
    # (venv not needed)
    sudo apt install redis-server
    sudo systemctl start redis-server
    ```
- **Windows:**
    - Download [Redis for Windows](https://github.com/microsoftarchive/redis/releases)
    - Extract and run in terminal:
        ```bash
        # (venv not needed)
        "C:\Program Files\Redis\redis-server.exe"
        ```
    - Leave this terminal open to keep Redis running.

#### Default Local Redis URL

```
redis://localhost:6379
```

If using default settings, you don't need a username or password. If you set a password, the URL would look like:
```
redis://default:<yourpassword>@localhost:6379
```

#### Set Redis URL as Environment Variable

```bash
# Windows (inside venv):
set REDIS_URL=redis://localhost:6379

# MacOS/Linux (inside venv):
export REDIS_URL="redis://localhost:6379"
```

#### Test Redis is Running

```bash
# (venv not needed)
redis-cli ping
# Should respond with "PONG"
```

---

## 🤖 Ollama Setup

### 1. Install Ollama

- Go to [https://ollama.com/download](https://ollama.com/download) and download the installer for your operating system (Mac, Windows, or Linux).

- Run the downloaded installer and follow the on-screen instructions to complete the installation.


### 2. Download LLM Model

```bash
# (venv not needed, but recommended)
ollama pull llama3
```

### 3. Start Ollama Service

```bash
# (venv not needed, but can be run from anywhere)
ollama serve
```
_Run this on another terminal. Keep this terminal running. If you close it, Ollama will stop._

---

## ⚙️ Application Configuration

### Create `.env` File

In the root folder, create a `.env` file with your config:

```ini
MONGO_URI=your_mongodb_uri_from_atlas
REDIS_URL=redis://localhost:6379
OLLAMA_MODEL=llama3
```

---

## 🚦 Running the Application

### 1. Activate Your Virtual Environment

- **Windows:**
    ```bash
    venv\Scripts\activate
    ```
- **MacOS/Linux:**
    ```bash
    source venv/bin/activate
    ```

### 2. Run the Server with Uvicorn

```bash
# Inside activated venv
uvicorn main:app --reload
```
- The app will be available at: [http://localhost:8000](http://localhost:8000)

---

## 🌐 API Endpoints

- `POST /chat` — Send and receive chat messages
- `GET /messages` — Get message history
- `POST /upload-document` — Upload PDF for RAG
- `GET /health` — Service health check

---

## 🚨 Troubleshooting

### MongoDB Connection Failed

- Ensure your IP is allowed in Atlas
- Double-check user/password in the URI

### Ollama Not Responding

```bash
# (venv not needed)
ollama list
```
- If you see models listed, Ollama is running

### Redis Connection Issues

- For local Redis, ensure the server is running:
    ```bash
    # Windows: check that "redis-server.exe" is running
    # Mac/Linux:
    redis-cli ping
    # Should reply: PONG
    ```

---

## 📝 Notes

- Commands for activating the virtual environment and running the app **must** be executed inside the venv.
- Redis must be running (either in a separate terminal or as a service) before starting the app.
- Ollama must be running and the model pulled before starting the app.

