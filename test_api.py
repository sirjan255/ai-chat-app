import requests

# Test /health endpoint
health_response = requests.get("http://localhost:8000/health")
print("Health Check:", health_response.json())

# Test /chat endpoint
chat_response = requests.post(
    "http://localhost:8000/chat",
    json={"text": "Hello, how are you?", "sender": "user"}
)
print("Chat Response:", chat_response.json())

# Test /messages endpoint
messages_response = requests.get("http://localhost:8000/messages")
print("Messages:", messages_response.json())