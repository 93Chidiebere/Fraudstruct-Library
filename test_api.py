import time
import requests

# Base URL where your FastAPI local server is running
BASE_URL = "http://127.0.0.1:8000"


def test_evaluate_endpoint():
    """Simulates sending various transaction scenarios to the evaluate endpoint."""
    print("--- Testing /v1/evaluate Endpoint ---")

    # Scenario 1: A normal transaction
    normal_tx = {
        "transaction_id": "TX-1001",
        "source_account": "ACC-ALICE",
        "destination_account": "ACC-BOB",
        "amount": 25000.0,
        "channel": "NIP",
    }

    # Scenario 2: A transaction exceeding the single cumulative limit
    whale_tx = {
        "transaction_id": "TX-1002",
        "source_account": "ACC-CHARLIE",
        "destination_account": "ACC-BOB",
        "amount": 2500000.0,  # 2.5 Million (Triggers BLOCK rule >= 2,000,000)
        "channel": "NIP",
    }

    # Send Scenario 1
    print("\nSending Normal Transaction...")
    response = requests.post(f"{BASE_URL}/v1/evaluate", json=normal_tx)
    print(f"Status Code: {response.status_code}")
    print("Response JSON:", response.json())

    # Send Scenario 2
    print("\nSending High-Value Transaction...")
    response = requests.post(f"{BASE_URL}/v1/evaluate", json=whale_tx)
    print(f"Status Code: {response.status_code}")
    print("Response JSON:", response.json())

    # Scenario 3: Simulating Velocity Structuring (Rapid consecutive transfers)
    print("\nSimulating Velocity Structuring (5 quick 100k transactions)...")
    for i in range(5):
        structuring_tx = {
            "transaction_id": f"TX-STRUCT-{i}",
            "source_account": "ACC-SUSPECT",
            "destination_account": "ACC-MULE",
            "amount": 100000.0,  # 5 * 100k = 500k total
            "channel": "NIP",
        }
        response = requests.post(f"{BASE_URL}/v1/evaluate", json=structuring_tx)
        print(
            f"Tx {i+1} - Decision: {response.json().get('decision')}, Reasons: {response.json().get('reasons')}"
        )


def test_train_endpoint():
    """Simulates sending account fraud labels to retrain the GNN model."""
    print("\n--- Testing /v1/train Endpoint ---")

    # Mock historical data labels for training
    training_data = {
        "labels": [
            {"account_id": "ACC-ALICE", "is_fraud": 0},
            {"account_id": "ACC-BOB", "is_fraud": 0},
            {"account_id": "ACC-SUSPECT", "is_fraud": 1},
            {"account_id": "ACC-MULE", "is_fraud": 1},
        ],
        "epochs": 5,  # Kept low for a fast test run
    }

    print("Sending GNN Training Request...")
    response = requests.post(f"{BASE_URL}/v1/train", json=training_data)

    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print("Success JSON:", response.json())
    else:
        print("Failure JSON:", response.json())


if __name__ == "__main__":
    # 1. Populate the engine graph first by running evaluations
    test_evaluate_endpoint()

    # 2. Wait a brief second for streaming graph updates to settle
    time.sleep(1)

    # 3. Test the retraining path now that the graph has node data
    test_train_endpoint()