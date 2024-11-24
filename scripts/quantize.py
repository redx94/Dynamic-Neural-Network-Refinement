import torch.quantization as quant
import torch
from src.model import DynamicNeuralNetwork

def apply_quantization(model, calibration_loader):
    """
    Apply post-training static quantization to the model.
    """
    model.eval()
    model.qconfig = quant.get_default_qconfig('fbgemm')
    
    # Fuse modules if necessary
    # Example:
    # model = torch.quantization.fuse_modules(model, [['conv1', 'bn1', 'relu1'], ['conv2', 'bn2', 'relu2']])
    
    # Prepare the model for static quantization
    quantized_model = quant.prepare(model, inplace=False)
    
    # Calibrate with calibration data
    with torch.no_grad():
        for data, _ in calibration_loader:
            data = data.to('cuda' if torch.cuda.is_available() else 'cpu')
            outputs = quantized_model(data)
    
    # Convert to quantized model
    quantized_model = quant.convert(quantized_model, inplace=False)
    
    return quantized_model

def main():
    model = DynamicNeuralNetwork(None)  # Pass appropriate thresholds or handle within the model
    model.load_state_dict(torch.load('models/pruned/pruned_model.pth'))
    model.to('cpu')  # Quantization is typically done on CPU
    
    # Assume calibration_loader is defined
    calibration_loader = None  # Define your calibration data loader here
    
    if calibration_loader is None:
        print("Please define the calibration_loader in the script.")
        return
    
    quantized_model = apply_quantization(model, calibration_loader)
    
    torch.save(quantized_model.state_dict(), 'models/quantized/quantized_model.pth')

if __name__ == "__main__":
    main()
