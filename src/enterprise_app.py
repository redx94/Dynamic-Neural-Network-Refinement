from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
import torch
import yaml
import os
from dotenv import load_dotenv

from src.analyzer import Analyzer
from src.enterprise.compute_proxy import ComputeProxy
from src.enterprise.anomaly_defense import AnomalyDefense
from src.enterprise.green_ai import GreenAI

load_dotenv()

API_KEY = os.getenv("API_KEY", "ENTERPRISE_KEY_XYZ")
API_KEY_NAME = "X-Enterprise-API-Key"

async def verify_api_key(api_key_header: str = Header(None, alias=API_KEY_NAME)):
    if api_key_header != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid Enterprise API Key")
    return True

app = FastAPI(
    title="DNN Refinement - Enterprise Suite",
    description="Showcasing dynamic load balancing, cyber-defense, and green-AI battery shifting.",
    version="1.0.0"
)

# Load config to grab initial baseline thresholds
with open('config/train_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

analyzer = Analyzer()

# Initialize Enterprise Modules
proxy = ComputeProxy(thresholds=config['thresholds'])
defense = AnomalyDefense(max_variance=2.5, max_entropy=6.0)
green_ai = GreenAI(critical_battery_level=20)


class PayloadRequest(BaseModel):
    data_payload: list  # e.g. 784 flattened pixels or raw numerical arrays
    
@app.get("/")
def health_check():
    return {"status": "Enterprise Security and Routing API Online"}


@app.post("/analyze_and_route", dependencies=[Depends(verify_api_key)])
async def analyze_and_route(request: PayloadRequest):
    """
    1) Cyber-Defense Check: Instantly measures payload for adversarial noise limits.
    2) Green-AI Check: Reads Server/Edge device power levels and throttles thresholds if low power.
    3) Resource Proxy: Decides whether to bounce the payload to the Cloud or process locally.
    """
    try:
        # Convert incoming payload to tensor
        input_tensor = torch.tensor(request.data_payload, dtype=torch.float32)
        if len(input_tensor.shape) == 1:
            input_tensor = input_tensor.view(1, -1)

        # Mathematical Complexity Analysis
        complexities = analyzer.analyze(input_tensor)

        # --- MODULE 1: ANOMALY DEFENSE ---
        if defense.analyze_for_threat(complexities):
            return {
                "status": "REJECTED_THREAT", 
                "reason": "Input exceeded strict mathematical bounds. Potential Cyber-Security (Adversarial) attack detected.",
                "metrics": {k: v.item() for k,v in complexities.items()}
            }

        # --- MODULE 2: GREEN-AI THROTTLING ---
        # Modify the thresholds based on the OS battery (if running on edge devices)
        current_thresholds = green_ai.adjust_thresholds(config['thresholds'])
        
        # We temporarily load those adjusted thresholds into the proxy
        proxy.variance_threshold = current_thresholds.get("variance", 0.5)
        proxy.entropy_threshold = current_thresholds.get("entropy", 0.5)
        proxy.sparsity_threshold  = current_thresholds.get("sparsity", 0.5)

        # --- MODULE 3: COMPUTE PROXY ROUTING ---
        # Based on complexity vs thresholds, where do we send this data for actual processing?
        destination = proxy.route_request(complexities)

        return {
            "status": "APPROVED",
            "routed_destination": destination,
            "metrics": {k: v.item() for k,v in complexities.items()}
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    # Start the Enterprise Dashboard
    print("[*] Booting Dynamic Enterprise Security + Routing Gateway...")
    uvicorn.run("enterprise_app:app", host="0.0.0.0", port=8000, reload=True)
