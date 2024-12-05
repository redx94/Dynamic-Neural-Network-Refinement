import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional

class DynamicDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, transform: Optional[callable] = None):
        """
        Initialize the dataset with features and labels
        
        Args:
            X: Input features
            y: Target labels
            transform: Optional transform to apply to features
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]
        y = self.y[idx]
        
        if self.transform:
            x = self.transform(x)
            
        return x, y

def create_data_loaders(X_train: np.ndarray, 
                       y_train: np.ndarray,
                       X_val: np.ndarray,
                       y_val: np.ndarray,
                       batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        batch_size: Batch size for data loaders
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = DynamicDataset(X_train, y_train)
    val_dataset = DynamicDataset(X_val, y_val)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, val_loader

def generate_synthetic_data(n_samples: int = 1000,
                          n_features: int = 10,
                          n_classes: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data for testing
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of input features
        n_classes: Number of output classes
        
    Returns:
        Tuple of (features, labels)
    """
    # Generate random features
    X = np.random.randn(n_samples, n_features)
    
    # Generate labels using a non-linear function
    weights = np.random.randn(n_features)
    y = np.sin(X.dot(weights)) + np.random.randn(n_samples) * 0.1
    
    # Normalize features and labels
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    y = (y - y.mean()) / y.std()
    
    return X, y
