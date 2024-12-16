
import torch
import torchvision
from torch.utils.data import DataLoader
from src.adaptive_thresholds import AdaptiveThresholds
from src.per_sample_complexity import ComplexityAnalyzer
from src.models.dynamic_nn import DynamicNeuralNetwork
from src.dataset_augmentation import ConditionalGAN
from src.neural_architecture_search import NeuralArchitectureSearch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model(model, train_loader, val_loader, epochs=100):
    import streamlit as st
    if not st.runtime.exists():
        st.runtime.get_instance()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Initialize metrics in session state if not present
    if 'metrics' not in st.session_state:
        st.session_state.metrics = {
            'training_loss': [],
            'validation_loss': [],
            'accuracy': [],
            'memory_usage': [],
            'learning_rates': [],
            'layer_activations': [],
            'gradient_norms': []
        }
    
    thresholds = AdaptiveThresholds().to(device)
    complexity_analyzer = ComplexityAnalyzer()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Calculate complexities
            complexities = complexity_analyzer.calculate_complexities(data)
            
            # Get adaptive thresholds
            adapted_thresholds = thresholds(complexities)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if batch_idx % 10 == 0:
                logger.info(f'Epoch: {epoch} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}')
                st.session_state.metrics['training_loss'].append(loss.item())
                st.session_state.metrics['memory_usage'].append(torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0)
                
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                
        val_loss /= len(val_loader)
        accuracy = 100. * correct / len(val_loader.dataset)
        logger.info(f'Validation - Average loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%')
        st.session_state.metrics['validation_loss'].append(val_loss)
        st.session_state.metrics['accuracy'].append(accuracy / 100.0)

if __name__ == '__main__':
    # Initialize model
    model = DynamicNeuralNetwork(input_dim=784, hidden_dims=[256, 128], output_dim=10)
    
    # Initialize data loaders (assuming MNIST data)
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                             transform=torchvision.transforms.ToTensor())
    val_dataset = torchvision.datasets.MNIST(root='./data', train=False,
                                           transform=torchvision.transforms.ToTensor())
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Train the model
    train_model(model, train_loader, val_loader)
