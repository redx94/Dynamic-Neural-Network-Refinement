import torch.nn.utils.prune as prune
import torch
from src.model import DynamicNeuralNetwork

def structured_pruning(model, amount=0.2):
    """
    Apply structured pruning to convolutional layers based on L1-norm.
    """
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.ln_structured(module, name='weight', amount=amount, n=1, dim=0)  # Prune filters
    
    # Remove pruning reparameterization to make pruning permanent
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.remove(module, 'weight')

def main():
    model = DynamicNeuralNetwork(None)  # Pass appropriate thresholds or handle within the model
    model.load_state_dict(torch.load('models/checkpoints/final_model.pth'))
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    structured_pruning(model, amount=0.2)
    
    torch.save(model.state_dict(), 'models/pruned/pruned_model.pth')

if __name__ == "__main__":
    main()
