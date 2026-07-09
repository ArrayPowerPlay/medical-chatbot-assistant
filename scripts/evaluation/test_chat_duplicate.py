import requests
import json
import uuid

def test_chat():
    print("Logging in as guest...")
    resp = requests.post("http://127.0.0.1:8000/api/auth/guest")
    token = resp.json()["access_token"]
    
    print("Fetching initial conversations...")
    headers = {"Authorization": f"Bearer {token}"}
    convs = requests.get("http://127.0.0.1:8000/api/conversations", headers=headers).json()
    print(f"Initial conversations: {len(convs)}")
    
    print("Sending chat request...")
    body = {
        "question": "test question",
        "conversation_id": None,
        "top_k": 5
    }
    
    # We use stream=True to mimic SSE
    with requests.post("http://127.0.0.1:8000/api/chat", headers=headers, json=body, stream=True) as r:
        for line in r.iter_lines():
            if line:
                print(line.decode('utf-8'))
                
    print("Fetching final conversations...")
    convs_after = requests.get("http://127.0.0.1:8000/api/conversations", headers=headers).json()
    print(f"Final conversations: {len(convs_after)}")
    for c in convs_after:
        print(f" - {c['id']}: {c['title']}")

if __name__ == "__main__":
    test_chat()
