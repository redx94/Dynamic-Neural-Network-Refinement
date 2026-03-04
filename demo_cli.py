import torch
from torchvision import datasets, transforms
from src.model import DynamicNeuralNetwork
from src.analyzer import Analyzer
import random

def run_demo():
    print("==================================================")
    print("🚀 DYNAMIC NEURAL NETWORK REFINEMENT: LIVE DEMO 🚀")
    print("==================================================\n")
    
    device = torch.device("cpu")
    
    # Initialize Model and Analyzer
    print("[*] Initializing Analyzer and Dynamic NN...")
    analyzer = Analyzer()
    model = DynamicNeuralNetwork(hybrid_thresholds=None)
    
    # Try to load best_model.pth if it exists
    try:
        model.load_state_dict(torch.load("best_model.pth", map_location=device, weights_only=True))
        print("[*] Loaded trained weights from 'best_model.pth'.")
    except FileNotFoundError:
        print("[!] Warning: 'best_model.pth' not found. Using untrained weights for routing demo.")
        
    model.eval()

    # Load a few MNIST samples
    print("[*] Fetching MNIST test samples...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    
    # Pick 3 random images
    indices = [random.randint(0, len(test_dataset)-1) for _ in range(3)]
    
    print("\n--------------------------------------------------")
    print(" TEST 1: STANDARD CLEAN IMAGES (Low Complexity) ")
    print("--------------------------------------------------")
    for i, idx in enumerate(indices):
        image, label = test_dataset[idx]
        img_flat = image.view(1, -1)
        
        complexities = analyzer.analyze(img_flat)
        uses_deep = model._should_use_deep_path(complexities)
        path = "DEEP PATH (Max Accuracy)" if uses_deep else "SHALLOW PATH (Max Speed)"
        
        with torch.no_grad():
            output = model(img_flat, complexities)
            pred = output.argmax(dim=1).item()
            
        print(f"Sample {i+1} (True Label: {label})")
        print(f"  - Variance: {complexities['variance'].item():.4f}")
        print(f"  - Entropy:  {complexities['entropy'].item():.4f}")
        print(f"  - Sparsity: {complexities['sparsity'].item():.4f}")
        print(f"  => ROUTING: {path} | Prediction: {pred}\n")
        
    print("--------------------------------------------------")
    print(" TEST 2: SYNTHETIC NOISY IMAGES (High Complexity) ")
    print("--------------------------------------------------")
    for i, idx in enumerate(indices):
        image, label = test_dataset[idx]
        
        # Inject intense random noise to spike variance and entropy, ruining sparsity
        noisy_image = image + torch.randn(image.size()) * 1.5
        img_flat = noisy_image.view(1, -1)
        
        complexities = analyzer.analyze(img_flat)
        uses_deep = model._should_use_deep_path(complexities)
        path = "DEEP PATH (Max Accuracy)" if uses_deep else "SHALLOW PATH (Max Speed)"
        
        with torch.no_grad():
            output = model(img_flat, complexities)
            pred = output.argmax(dim=1).item()
            
        print(f"Noisy Sample {i+1} (True Label: {label})")
        print(f"  - Variance: {complexities['variance'].item():.4f}")
        print(f"  - Entropy:  {complexities['entropy'].item():.4f}")
        print(f"  - Sparsity: {complexities['sparsity'].item():.4f}")
        print(f"  => ROUTING: {path} | Prediction: {pred}\n")
        
    print("==================================================")
    print("DEMO COMPLETE: The network dynamically scales its compute!")
    print("==================================================")

if __name__ == "__main__":
    run_demo()
