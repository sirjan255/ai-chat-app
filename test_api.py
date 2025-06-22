import requests
import time
import os

BASE_URL = "http://localhost:8000"

def wait_for_server(timeout=10):
    print("Waiting for FastAPI server to be available...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            response = requests.get(f"{BASE_URL}/health")
            if response.ok:
                print("✅ Server is up.")
                return True
        except Exception:
            pass
        time.sleep(1)
    print("❌ Server did not respond within timeout.")
    return False

def test_health():
    print("\n[TEST] /health")
    resp = requests.get(f"{BASE_URL}/health")
    print("Status code:", resp.status_code)
    print("Response:", resp.json())

def test_chat():
    print("\n[TEST] /chat")
    payload = {"text": "What is a transformer?", "sender": "user"}
    resp = requests.post(f"{BASE_URL}/chat", json=payload)
    print("Status code:", resp.status_code)
    try:
        data = resp.json()
        print("AI Response:", data.get("ai_response"))
        print("Messages returned:", len(data.get("messages", [])))
        for msg in data.get("messages", []):
            print(f"- [{msg['sender']}] {msg['text']} ({msg['timestamp']})")
    except Exception as e:
        print("Error decoding JSON:", e)
        print("Raw content:", resp.content)

def test_messages():
    print("\n[TEST] /messages")
    resp = requests.get(f"{BASE_URL}/messages")
    print("Status code:", resp.status_code)
    try:
        messages = resp.json()
        print(f"Returned {len(messages)} messages.")
        for msg in messages:
            print(f"- [{msg['sender']}] {msg['text']} ({msg['timestamp']})")
    except Exception as e:
        print("Error decoding JSON:", e)
        print("Raw content:", resp.content)

def test_upload_document():
    print("\n[TEST] /upload-document")
    # You can use any small PDF for testing
    sample_pdf = "sample.pdf"
    if not os.path.exists(sample_pdf):
        print("No 'sample.pdf' found. Skipping upload-document test.")
        return
    with open(sample_pdf, "rb") as f:
        files = {"file": (sample_pdf, f, "application/pdf")}
        resp = requests.post(f"{BASE_URL}/upload-document", files=files)
        print("Status code:", resp.status_code)
        try:
            print("Response:", resp.json())
        except Exception as e:
            print("Error decoding JSON:", e)
            print("Raw content:", resp.content)

if __name__ == "__main__":
    if wait_for_server():
        test_health()
        test_chat()
        test_messages()
        test_upload_document()
    else:
        print("Exiting: could not reach FastAPI server.")