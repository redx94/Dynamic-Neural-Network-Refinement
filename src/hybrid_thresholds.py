# src/hybrid_thresholds.py

import torch

class HybridThresholds:
    """
    Handles dynamic threshold adjustments based on annealing schedule.
    """

    def __init__(self, initial_thresholds, annealing_start_epoch, total_epochs):
        """
        Initializes the HybridThresholds class.

        Args:
            initial_thresholds (dict): Initial thresholds for variance, entropy, and sparsity.
            annealing_start_epoch (int): Epoch to start annealing.
            total_epochs (int): Total number of epochs for annealing.
        """
        self.initial_thresholds = initial_thresholds
        self.annealing_start_epoch = annealing_start_epoch
        self.total_epochs = total_epochs

    def anneal_thresholds(self, current_epoch):
        """
        Calculates the current thresholds based on the annealing schedule.

        Args:
            current_epoch (int): The current training epoch.

        Returns:
            dict: Updated thresholds.
        """
        if current_epoch < self.annealing_start_epoch:
            return self.initial_thresholds
        else:
            progress = (current_epoch - self.annealing_start_epoch) / (self.total_epochs - self.annealing_start_epoch)
            annealed_thresholds = {k: v * (1 - progress) for k, v in self.initial_thresholds.items()}
            return annealed_thresholds

    def __call__(self, variance, entropy, sparsity, current_epoch):
        """
        Updates thresholds based on the current epoch and applies them to the complexities.

        Args:
            variance (torch.Tensor): Variance tensor.
            entropy (torch.Tensor): Entropy tensor.
            sparsity (torch.Tensor): Sparsity tensor.
            current_epoch (int): The current training epoch.

        Returns:
            dict: Thresholded complexities.
        """
        thresholds = self.anneal_thresholds(current_epoch)
        thresholded = {
            'variance': variance > thresholds['variance'],
            'entropy': entropy > thresholds['entropy'],
            'sparsity': sparsity < thresholds['sparsity']
        }
        return thresholded
