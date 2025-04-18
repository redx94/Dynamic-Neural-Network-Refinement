from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
import torch
import yaml
from src.model import DynamicNeuralNetwork
from src.hybrid_thresholds import HybridThresholds
from src.analyzer import Analyzer
from scripts.utils import load_model
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("API_KEY")
API_KEY_NAME = "X-API-Key"

async def verify_api_key(api_key_header: str = Header(None, alias=API_KEY_NAME)):
    if api_key_header != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return True

app = FastAPI(title="Dynamic Neural Network Refinement API")


class InferenceRequest(BaseModel):
    input_data: list
    current_epoch: int = 1


# Load configuration
with open('config/train_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Initialize model components
device = 'cuda' if torch.cuda.is_available() else 'cpu'
hybrid_thresholds = HybridThresholds(
    initial_thresholds=config['thresholds'],
    annealing_start_epoch=config['thresholds']['annealing_start_epoch'],
    total_epochs=config['thresholds']['total_epochs']
)

model = DynamicNeuralNetwork(hybrid_thresholds=hybrid_thresholds).to(device)
model = load_model(model, config['output']['final_model_path'], device)
analyzer = Analyzer()


@app.post("/predict", dependencies=[Depends(verify_api_key)])
async def predict(request: InferenceRequest):
    """
    Endpoint for model inference.
    """
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
