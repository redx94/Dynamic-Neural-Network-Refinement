import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
import numpy as np
from src.models.neural_network import DynamicNeuralNetwork
from src.models.refinement import NetworkRefinement

class Trainer:
    def __init__(self, 
                 model: DynamicNeuralNetwork,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 criterion: Optional[nn.Module] = None,
                 device: str = 'cpu'):
        """
        Initialize the trainer
        
        Args:
            model: The neural network model to train
            optimizer: Optional optimizer (default: Adam)
            criterion: Optional loss function (default: MSELoss)
            device: Device to train on (default: 'cpu')
        """
        self.model = model
        self.device = device
        self.model.to(device)
        
        # Default optimizer and loss function if not provided
        self.optimizer = optimizer or optim.Adam(model.parameters())
        self.criterion = criterion or nn.MSELoss()
        
        # Initialize refinement module
        self.refinement = NetworkRefinement(model)
        
        # Training history
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()
                
        return total_loss / len(val_loader)
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              epochs: int = 100,
              refinement_frequency: int = 5) -> Dict[str, List[float]]:
        """
        Train the model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
            refinement_frequency: How often to apply refinement
            
        Returns:
            Dictionary containing training history
        """
        for epoch in range(epochs):
            # Training phase
            train_loss = self.train_epoch(train_loader)
            
            # Validation phase
            val_loss = self.validate(val_loader)
            
            # Save metrics
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            # Refinement phase
            if epoch > 0 and epoch % refinement_frequency == 0:
                made_changes = self.refinement.refine_architecture()
                if made_changes:
                    # Reinitialize optimizer if architecture changed
                    self.optimizer = optim.Adam(self.model.parameters())
            
            # Print progress
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'Train Loss: {train_loss:.4f}')
            print(f'Val Loss: {val_loss:.4f}')
            
        return self.history
    
    def get_learning_curves(self) -> Tuple[List[float], List[float]]:
        """Return learning curves data"""
        return (self.history['train_loss'], self.history['val_loss'])
