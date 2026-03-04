import torch
from collections import defaultdict


class ComplexityAnalyzer:
    """
    Analyzes per-sample complexity based on variance, entropy, and sparsity.
    """

    def __init__(self, window_size=100):
        """
        Initializes the complexity analyzer.

        Args:
            window_size (int): Size of the rolling window for complexity tracking.
        """
        self.window_size = window_size
        self.history = defaultdict(list)

    def calculate_complexities(self, x):
        """
        Computes variance, entropy, and sparsity metrics.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            dict: Dictionary of complexity metrics.
        """
        complexities = {}

        # Variance complexity
        complexities['variance'] = torch.var(x, dim=1, unbiased=False).mean()

        # Entropy complexity
        softmax_vals = torch.nn.functional.softmax(x, dim=1)
        entropy = -(softmax_vals * torch.log(softmax_vals + 1e-10)).sum(dim=1).mean()
        complexities['entropy'] = entropy

        # Sparsity complexity
        sparsity = (torch.abs(x) < 0.01).float().mean()
        complexities['sparsity'] = sparsity

        return complexities

    def update_history(self, complexities):
        """
        Updates historical complexity records.

        Args:
            complexities (dict): Complexity metrics.
        """
        for metric, value in complexities.items():
            self.history[metric].append(value)
            if len(self.history[metric]) > self.window_size:
                self.history[metric].pop(0)

    def detect_drift(self, threshold=0.1):
        """
        Detects significant drift in complexity trends.

        Args:
            threshold (float): Minimum difference to trigger a drift detection.

        Returns:
            dict: Dictionary indicating drift status for each metric.
        """
        for metric, values in self.history.items():
            if len(values) >= self.window_size:
                first_half = torch.tensor(values[: self.window_size // 2]).mean().item()
                second_half = torch.tensor(values[self.window_size // 2 :]).mean().item()
                ({})[metric] = abs(first_half - second_half) > threshold

        return {}
