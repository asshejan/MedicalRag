import requests
import time
import uuid
from pprint import pprint

# Configuration
BASE_URL = "http://localhost:8000"  # Update if your server runs on a different port
USER_ID = f"test_user_{uuid.uuid4().hex[:8]}"  # Generate a random user ID for testing

def test_conversation_memory():
    print(f"\n=== Testing Conversation Memory with User ID: {USER_ID} ===")
    
    # Step 1: Start a new conversation (should get a greeting)
    print("\n1. Starting a new conversation (should include greeting)")
    response = requests.post(
        f"{BASE_URL}/tutor/ask",
        json={"question": "What are the symptoms of diabetes?", "user_id": USER_ID, "session_id": None}
    )
    data = response.json()
    print("Response:")
    pprint(data)
    session_id = data['session_id']
    print(f"\nSession ID: {session_id}")
    print(f"Is new session: {data.get('is_new_session', 'N/A')}")
    
    # Step 2: Continue the conversation (should maintain context)
    print("\n2. Continuing conversation with follow-up question (should maintain context)")
    time.sleep(1)  # Small delay between requests
    response = requests.post(
        f"{BASE_URL}/tutor/ask",
        json={"question": "What about treatment options?", "user_id": USER_ID, "session_id": session_id}
    )
    data = response.json()
    print("Response:")
    pprint(data)
    print(f"\nIs new session: {data.get('is_new_session', 'N/A')}")
    
    # Step 3: Ask another follow-up question
    print("\n3. Asking another follow-up question")
    time.sleep(1)  # Small delay between requests
    response = requests.post(
        f"{BASE_URL}/tutor/ask",
        json={"question": "Are there any lifestyle changes that can help?", "user_id": USER_ID, "session_id": session_id}
    )
    data = response.json()
    print("Response:")
    pprint(data)
    print(f"\nIs new session: {data.get('is_new_session', 'N/A')}")
    
    # Step 4: Get conversation history
    print("\n4. Getting conversation history")
    response = requests.get(
        f"{BASE_URL}/tutor/conversation-history",
        params={"session_id": session_id}
    )
    data = response.json()
    print("Conversation History:")
    pprint(data)
    
    # Step 5: End the session and generate summary
    print("\n5. Ending session and generating summary")
    response = requests.post(
        f"{BASE_URL}/tutor/end-session",
        params={"session_id": session_id}
    )
    data = response.json()
    print("Session End Response:")
    pprint(data)
    
    # Step 6: Start a new session (should include personalized greeting with previous summary)
    print("\n6. Starting a new session (should include personalized greeting with previous summary)")
    time.sleep(2)  # Give time for the summary to be stored
    response = requests.post(
        f"{BASE_URL}/tutor/ask",
        json={"question": "Tell me about heart disease", "user_id": USER_ID, "session_id": None}
    )
    data = response.json()
    print("Response (should include personalized greeting):")
    pprint(data)
    new_session_id = data['session_id']
    print(f"\nNew Session ID: {new_session_id}")
    print(f"Is new session: {data.get('is_new_session', 'N/A')}")
    
    # Step 7: Get session summaries
    print("\n7. Getting session summaries")
    response = requests.get(
        f"{BASE_URL}/tutor/session-summaries",
        params={"user_id": USER_ID, "limit": 5}
    )
    data = response.json()
    print("Session Summaries:")
    pprint(data)

if __name__ == "__main__":
    test_conversation_memory()