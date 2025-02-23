# src/analyzer.py

import torch
import torch.nn.functional as F

class Analyzer:
    """
    Analyzer class to compute complexity metrics of input data.
    """

    def compute_variance(self, data: torch.Tensor) -> torch.Tensor:
        """
        Computes the variance of the data along the specified dimension.

        Args:
            data (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Variance tensor.
        """
        var = torch.var(data, dim=1)
        return var

    def compute_entropy(self, data: torch.Tensor) -> torch.Tensor:
        """
        Computes the entropy of the data using softmax normalization.

        Args:
            data (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Entropy tensor.
        """
        probabilities = F.softmax(data, dim=1)
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1)
        return entropy

    def compute_sparsity(self, data: torch.Tensor) -> torch.Tensor:
        """
        Computes the sparsity of the data based on a threshold.

        Args:
            data (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Sparsity tensor.
        """
        threshold = 0.1  # Example threshold
        sparsity = torch.mean((data.abs() < threshold).float(), dim=1)
        return sparsity

    def analyze(self, data: torch.Tensor) -> dict:
        """
        Aggregates the computed variance, entropy, and sparsity into a dictionary.

        Args:
            data (torch.Tensor): Input tensor.

        Returns:
            dict: Dictionary containing complexity metrics.
        """
        variance = self.compute_variance(data)
        entropy = self.compute_entropy(data)
        sparsity = self.compute_sparsity(data)
        complexities = {
            'variance': variance,
            'entropy': entropy,
            'sparsity': sparsity
        }
        return complexities
