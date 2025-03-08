import argparse
import requests
import json
from datetime import datetime
from typing import Any, Dict

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test the Flare AI Consensus API with different queries."
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="You are a helpful, knowledgeable AI assistant. Provide accurate and thorough answers.",
        help="System prompt to use for the query."
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="The question or query to send to the consensus system.",
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8080/api/routes/chat/",  # Note the trailing slash
        help="The API endpoint URL.",
    )
    return parser.parse_args()

def save_response(response_data: Dict[str, Any], query: str) -> None:
    # Create a simple filename based on the query
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"test_results/api_response_{timestamp}.json"
    
    # Save the response
    with open(filename, "w") as file:
        json.dump({
            "query": query,
            "timestamp": timestamp,
            "response": response_data
        }, file, indent=2)
    
    print(f"Response saved to {filename}")

def test_api(url: str, system_prompt: str, query: str) -> None:
    print(f"\nSending query to {url}:")
    print(f"System prompt: {system_prompt}")
    print(f"Query: {query}")
    
    # Prepare payload
    payload = {
        "system_message": system_prompt,
        "user_message": query
    }
    
    try:
        # Send request to API
        response = requests.post(url, json=payload)
        
        # Check if successful
        if response.status_code == 200:
            result = response.json()
            
            # Save the full response
            save_response(result, query)
            
            # Print the answer
            print("\n" + "="*80)
            print("CONSENSUS RESPONSE:")
            print("="*80)
            print(result["response"])
            print("="*80)
        else:
            print(f"Error: Received status code {response.status_code}")
            print(response.text)
    
    except Exception as e:
        print(f"Error: {str(e)}")

def main() -> None:
    args = parse_arguments()
    test_api(args.url, args.system_prompt, args.query)

if __name__ == "__main__":
    main() 