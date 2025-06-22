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
    
    # URL of the PDF you want to use for testing
    pdf_url = "https://study.iitm.ac.in/ds/assets/pdf/Brochure.pdf"
    
    try:
        # Download the PDF directly
        print(f"Downloading test PDF from {pdf_url}...")
        response = requests.get(pdf_url)
        response.raise_for_status()  # Raise exception for bad status codes
        
        # Prepare the file for upload
        files = {
            "file": ("iitm_brochure.pdf", response.content, "application/pdf")
        }
        
        # Send to your endpoint
        resp = requests.post(f"{BASE_URL}/upload-document", files=files)
        print("Status code:", resp.status_code)
        
        try:
            print("Response:", resp.json())
        except Exception as e:
            print("Error decoding JSON:", e)
            print("Raw content:", resp.content)
            
    except Exception as e:
        print(f"Failed to download or process PDF: {str(e)}")

def test_chat_summary():
    print("\n[TEST] /chat-summary")
    try:
        resp = requests.get(f"{BASE_URL}/chat-summary?limit=20")
        print("Status code:", resp.status_code)
        try:
            print("Response:", resp.json())
        except Exception as e:
            print("Error decoding JSON:", e)
            print("Raw content:", resp.content)
    except Exception as e:
        print("Failed to test /chat-summary:", str(e))

def test_export_conversation():
    print("\n[TEST] /export-conversation")
    try:
        resp = requests.get(f"{BASE_URL}/export-conversation?limit=20", stream=True)
        print("Status code:", resp.status_code)
        content_disposition = resp.headers.get("Content-Disposition", "")
        if resp.status_code == 200 and "attachment" in content_disposition:
            filename = "conversation_export_test.txt"
            with open(filename, "wb") as f:
                for chunk in resp.iter_content(chunk_size=4096):
                    f.write(chunk)
            print(f"Exported conversation saved to '{filename}'")
        else:
            try:
                print("Response:", resp.json())
            except Exception as e:
                print("Error decoding JSON:", e)
                print("Raw content:", resp.content)
    except Exception as e:
        print("Failed to test /export-conversation:", str(e))

if __name__ == "__main__":
    if wait_for_server():
        test_health()
        test_chat()
        test_messages()
        test_upload_document()
        test_chat_summary()
        test_export_conversation()
    else:
        print("Exiting: could not reach FastAPI server.")