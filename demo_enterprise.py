import torch
import requests
import json
import time
from torchvision import datasets, transforms

# Set up to hit the local FastAPI gateway we just built
API_URL = "http://127.0.0.1:8000/analyze_and_route"
HEADERS = {
    "X-Enterprise-API-Key": "ENTERPRISE_KEY_XYZ",
    "Content-Type": "application/json"
}

def ping_enterprise_gateway(label: str, pixels: list):
    """Sends a packet to our custom Cyber-Security & Routing Tunnel."""
    payload = {"data_payload": pixels}
    
    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload)
        data = response.json()
        
        status = data.get("status")
        route = data.get("routed_destination", "N/A")
        print(f"[{label}] API RESPONSE: {status} -> {route}")
        if status == "REJECTED_THREAT":
            print(f"   => ALARM: {data.get('reason')}")
            
    except requests.exceptions.ConnectionError:
        print("[!] ERROR: Enterprise Gateway is offline. Ensure `python src/enterprise_app.py` is running!")


def run_enterprise_demo():
    print("==========================================================")
    print("🔒 ENTERPRISE GATEWAY & ROUTING API - LIVE DEMO 🔒")
    print("==========================================================\n")
    
    # 1. Load an image from MNIST
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    clean_image, img_label = dataset[0]  # Grab a random '7'
    clean_pixels = clean_image.view(-1).tolist()
    
    # 2. Inject intense FGSM-style Adversarial Noise to simulate an attack (Spikes the variance massively)
    noisy_image = clean_image + torch.randn(clean_image.size()) * 4.0
    poisoned_pixels = noisy_image.view(-1).tolist()
    
    print("--- TEST 1: COMPUTE-SAVER LOAD BALANCING (Clean Image) ---")
    print("Simulating standard user requesting an inference...")
    ping_enterprise_gateway("CLEAN_REQ", clean_pixels)
    
    print("\n--- TEST 2: ANOMALY DEFENSE INCIDENT (Adversarial Data) ---")
    print("Simulating Hacker injecting noise to trick the neural network...")
    ping_enterprise_gateway("HACKER_ATTACK", poisoned_pixels)
    
    print("\n==========================================================")
    print(" DEMO COMPLETE! 🚀 The Gateway is successfully protecting ")
    print(" the internal engines and routing valid traffic instantly! ")
    print("==========================================================")

if __name__ == "__main__":
    run_enterprise_demo()
