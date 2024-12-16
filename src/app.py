# src/app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from src.model import DynamicNeuralNetwork
from src.hybrid_thresholds import HybridThresholds
from src.analyzer import Analyzer
from scripts.utils import load_model
import yaml

app = FastAPI(title="Dynamic Neural Network Refinement API")

# Define request schema
class InferenceRequest(BaseModel):
    input_data: list
    current_epoch: int = 1

# Load configuration
with open('config/train_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Initialize components
device = 'cuda' if torch.cuda.is_available() else 'cpu'
hybrid_thresholds = HybridThresholds(
    initial_thresholds=config['thresholds'],
    annealing_start_epoch=config['thresholds']['annealing_start_epoch'],
    total_epochs=config['thresholds']['total_epochs']
)
model = DynamicNeuralNetwork(hybrid_thresholds=hybrid_thresholds).to(device)
model = load_model(model, config['output']['final_model_path'], device)
analyzer = Analyzer()

@app.post("/predict")
def predict(request: InferenceRequest):
    try:
        input_tensor = torch.tensor(request.input_data, dtype=torch.float32).to(device)
        complexities = analyzer.analyze(input_tensor)
        complexities = hybrid_thresholds(
            complexities['variance'],
            complexities['entropy'],
            complexities['sparsity'],
            current_epoch=request.current_epoch
        )
        outputs = model(input_tensor, complexities)
        predictions = outputs.argmax(dim=1).tolist()
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
import streamlit as st
from dashboard import Dashboard
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
dashboard = Dashboard()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Dynamic Neural Network Refinement API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
