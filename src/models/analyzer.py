import torch
import torch.nn.functional as F
from typing import Dict

class Analyzer:
    """
    Computes complexity metrics for input data to guide dynamic routing decisions.
    """
    def __init__(self, eps: float = 1e-8):
        """
        Initialize analyzer with numerical stability parameter.
        
        Args:
            eps: Small constant for numerical stability
        """
        self.eps = eps
        
    def compute_variance(self, data: torch.Tensor) -> torch.Tensor:
        """
        Compute variance of input features.
        Args:
            data: Input tensor of shape (batch_size, feature_dim)
        Returns:
            Variance tensor of shape (batch_size,)
        """
        # Ensure the input is at least 2D and flatten all dimensions after the first
        if data.dim() > 2:
            data = data.view(data.size(0), -1)
        # Compute variance along feature dimension with unbiased estimation
        return torch.var(data, dim=1, unbiased=True)
        
    def compute_entropy(self, data: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of normalized input features.
        """
        # Normalize data to probability-like values
        probs = F.softmax(data, dim=1)
        # Compute entropy
        entropy = -torch.sum(probs * torch.log(probs + self.eps), dim=1)
        return entropy
        
    def compute_sparsity(self, data: torch.Tensor, threshold: float = 0.1) -> torch.Tensor:
        """
        Compute sparsity ratio of input features.
        """
        # Calculate ratio of near-zero values
        sparsity = torch.mean((torch.abs(data) < threshold).float(), dim=1)
        return sparsity
        
    def analyze(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute all complexity metrics for input data.
        
        Args:
            data: Input tensor of shape (batch_size, feature_dim)
            
        Returns:
            Dictionary containing computed complexity metrics
        """
        return {
            'variance': self.compute_variance(data),
            'entropy': self.compute_entropy(data),
            'sparsity': self.compute_sparsity(data)
        }
