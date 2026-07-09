import os
import requests
import json
import sseclient

def test_sse():
    print("Testing SSE stream from backend...")
    
    # 1. Login to get token (as guest)
    login_resp = requests.post("http://127.0.0.1:8000/api/auth/guest")
    if login_resp.status_code != 200:
        print("Failed to login")
        return
    token = login_resp.json().get("access_token")
    
    # 2. Send chat request
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    data = {
        "question": "What is paracetamol?",
        "top_k": 3
    }
    
    response = requests.post("http://127.0.0.1:8000/api/chat", headers=headers, json=data, stream=True)
    
    client = sseclient.SSEClient(response)
    
    full_text = ""
    for event in client.events():
        if event.event == "metadata":
            print("Received metadata:", event.data)
        elif event.event == "message":
            # parse the JSON chunk
            chunk = json.loads(event.data)
            full_text += chunk
            print(f"Token: '{chunk}'")
        elif event.event == "final_answer":
            print("\nReceived final_answer:", event.data)
        elif event.event == "error":
            print("Received error:", event.data)
            
    print("\n--- FINAL CONSTRUCTED TEXT ---")
    print(full_text)
    print("------------------------------")
    
    if "TheThe" in full_text or "is is" in full_text:
        print("Backend sent duplicated tokens!")
    else:
        print("Backend sent tokens normally. The duplication is purely in the frontend React Strict Mode!")

if __name__ == "__main__":
    test_sse()
