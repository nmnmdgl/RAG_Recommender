import requests
import json
import time

API_URL = "http://127.0.0.1:8000/chat"

def test_conversational_flow():
    print("🚀 Starting API Test Simulation...\n")
    
    # Simulating Turn 1 (Vague request, expecting a clarifying question, no recs)
    history = [
        {"role": "user", "content": "We need a solution for senior leadership."}
    ]
    
    print("👤 USER (Turn 1):", history[0]['content'])
    try:
        response = requests.post(API_URL, json={"messages": history})
        data = response.json()
        print("🤖 AGENT:", data.get('reply'))
        print("📊 REC COUNT:", len(data.get('recommendations', [])))
        print("🏁 END CONVO:", data.get('end_of_conversation'))
        
        # Append agent's reply to history
        history.append({"role": "assistant", "content": data.get('reply')})
        
    except Exception as e:
        print("❌ Error connecting to server. Is FastAPI running?", e)
        return

    print("\n---\n")
    time.sleep(2)

    # Simulating Turn 2 (Specific request, expecting recommendations)
    user_turn_2 = "The pool consists of CXOs, director-level positions; people with more than 15 years of experience."
    print("👤 USER (Turn 2):", user_turn_2)
    history.append({"role": "user", "content": user_turn_2})
    
    try:
        response = requests.post(API_URL, json={"messages": history})
        data = response.json()
        print("🤖 AGENT:", data.get('reply'))
        print("📊 REC COUNT:", len(data.get('recommendations', [])))
        print("🏁 END CONVO:", data.get('end_of_conversation'))
        if data.get('recommendations'):
            print("📝 SHORTLIST:")
            for item in data['recommendations']:
                print(f"  - [{item['test_type']}] {item['name']}")
                
    except Exception as e:
        print("❌ Error on Turn 2:", e)

if __name__ == "__main__":
    test_conversational_flow()