import torch


class HybridThresholds:
    """
    Manages dynamic thresholds for complexity-based routing with cosine annealing.
    """

    def __init__(self, initial_thresholds, annealing_start_epoch, total_epochs, min_threshold=0.1):
        """
        Initializes hybrid thresholds manager.

        Args:
            initial_thresholds (dict): Initial threshold values for each metric.
            annealing_start_epoch (int): Epoch to start threshold annealing.
            total_epochs (int): Total number of training epochs.
            min_threshold (float): Minimum threshold value after annealing.
        """
        self.initial_thresholds = initial_thresholds
        self.annealing_start_epoch = annealing_start_epoch
        self.total_epochs = total_epochs
        self.min_threshold = min_threshold
        self.current_thresholds = initial_thresholds.copy()

    def compute_annealing_factor(self, current_epoch: int) -> float:
        """
        Computes cosine annealing factor based on current epoch.

        Args:
            current_epoch (int): The current training epoch.

        Returns:
            float: Annealing factor.
        """
        if current_epoch < self.annealing_start_epoch:
            return 1.0

        progress = (current_epoch - self.annealing_start_epoch) / (
            self.total_epochs - self.annealing_start_epoch
        )
        return self.min_threshold + (1 - self.min_threshold) * (1 + torch.cos(torch.pi * progress)) / 2

    def update_thresholds(self, current_epoch: int) -> None:
        """
        Updates thresholds based on cosine annealing.

        Args:
            current_epoch (int): The current training epoch.
        """
        factor = self.compute_annealing_factor(current_epoch)
        self.current_thresholds = {k: v * factor for k, v in self.initial_thresholds.items()}

    def __call__(self, variance, entropy, sparsity, current_epoch):
        """
        Applies thresholds to complexity metrics.

        Args:
            variance (torch.Tensor): Variance metric.
            entropy (torch.Tensor): Entropy metric.
            sparsity (torch.Tensor): Sparsity metric.
            current_epoch (int): Current training epoch.

        Returns:
            dict: Thresholded complexity indicators.
        """
        self.update_thresholds(current_epoch)
        return {
            'variance': variance > self.current_thresholds['variance'],
            'entropy': entropy > self.current_thresholds['entropy'],
            'sparsity': sparsity < self.current_thresholds['sparsity']
        }

    def get_current_thresholds(self) -> dict:
        """
        Returns current threshold values.

        Returns:
            dict: Dictionary of threshold values.
        """
        return self.current_thresholds.copy()