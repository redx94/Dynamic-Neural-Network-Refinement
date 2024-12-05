
import torch.nn.utils.prune as prune

def structured_pruning(model, amount=0.2):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(module, name='weight', amount=amount, n=1, dim=0)
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.remove(module, 'weight')

def apply_quantization(model, train_loader_calibration):
    model.eval()
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    quantized_model = torch.quantization.prepare(model, inplace=False)
    with torch.no_grad():
        for data, _ in train_loader_calibration:
            data = data.to(next(model.parameters()).device)
            quantized_model(data)
    quantized_model = torch.quantization.convert(quantized_model, inplace=False)
    return quantized_model
    