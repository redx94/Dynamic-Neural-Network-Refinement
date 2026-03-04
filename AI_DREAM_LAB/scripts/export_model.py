
import torch
import argparse
from src.models import DynamicNeuralNetwork

def export_model(checkpoint_path: str, output_path: str):
    """Export model to ONNX format."""
    model = DynamicNeuralNetwork()
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(model, dummy_input, output_path, 
                     input_names=['input'],
                     output_names=['output'],
                     dynamic_axes={'input': {0: 'batch_size'},
                                 'output': {0: 'batch_size'}})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    
    export_model(args.checkpoint, args.output)
