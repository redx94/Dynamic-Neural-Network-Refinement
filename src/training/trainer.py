import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any
from loguru import logger
from models.neural_network import DynamicNeuralNetwork
from models.hybrid_thresholds import HybridThresholds
from models.analyzer import Analyzer

class Trainer:
    """
    Trainer class for the Dynamic Neural Network.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.model = DynamicNeuralNetwork(
            input_dim=config['model']['input_dim'],
            hidden_dims=config['model']['hidden_dims'],
            output_dim=config['model']['output_dim']
        ).to(self.device)
        
        self.hybrid_thresholds = HybridThresholds(
            initial_thresholds=config['thresholds'],
            annealing_start_epoch=config['training']['annealing_start_epoch'],
            total_epochs=config['training']['epochs']
        )
        
        self.analyzer = Analyzer()
        
        # Setup loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate']
        )
        
    def train_epoch(self, dataloader: torch.utils.data.DataLoader, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(dataloader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Compute complexity metrics
            complexities = self.analyzer.analyze(data)
            thresholded = self.hybrid_thresholds(
                complexities['variance'],
                complexities['entropy'],
                complexities['sparsity'],
                current_epoch=epoch
            )
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(data, thresholded)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if batch_idx % self.config['training']['log_interval'] == 0:
                logger.info(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
        
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': 100. * correct / total
        }
    
    def validate(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Validate the model.
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in dataloader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                # Compute complexity metrics
                complexities = self.analyzer.analyze(data)
                thresholded = self.hybrid_thresholds(
                    complexities['variance'],
                    complexities['entropy'],
                    complexities['sparsity'],
                    current_epoch=self.config['training']['epochs']  # Use final thresholds
                )
                
                outputs = self.model(data, thresholded)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        return {
            'val_loss': total_loss / len(dataloader),
            'val_accuracy': 100. * correct / total
        }
    
    def train(self, train_loader: torch.utils.data.DataLoader, 
             val_loader: torch.utils.data.DataLoader) -> Dict[str, list]:
        """
        Complete training process.
        """
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
        for epoch in range(1, self.config['training']['epochs'] + 1):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Log metrics
            logger.info(
                f"Epoch {epoch}/{self.config['training']['epochs']} - "
                f"Loss: {train_metrics['loss']:.4f} - "
                f"Acc: {train_metrics['accuracy']:.2f}% - "
                f"Val Loss: {val_metrics['val_loss']:.4f} - "
                f"Val Acc: {val_metrics['val_accuracy']:.2f}%"
            )
            
            # Store metrics
            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])
            history['val_loss'].append(val_metrics['val_loss'])
            history['val_acc'].append(val_metrics['val_accuracy'])
        
        return history
