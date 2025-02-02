class HybridThresholds:
    """
    Handles dynamic threshold adjustments based on an annealing schedule.
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
        Calculates the current thresholds based on annealing.

        Args:
            current_epoch (int): The current training epoch.

        Returns:
            dict: Updated thresholds.
        """
        if current_epoch < self.annealing_start_epoch:
            return self.initial_thresholds

        progress = (current_epoch - self.annealing_start_epoch) / (
            self.total_epochs - self.annealing_start_epoch
        )

        return {
            k: v * (1 - progress) for k, v in self.initial_thresholds.items()
        }

    def __call__(self, variance, entropy, sparsity, current_epoch):
        """
        Updates thresholds based on the current epoch and applies them to complexities.

        Args:
            variance (float): Variance metric.
            entropy (float): Entropy metric.
            sparsity (float): Sparsity metric.
            current_epoch (int): Current training epoch.

        Returns:
            dict: Thresholded complexities.
        """
        thresholds = self.anneal_thresholds(current_epoch)
        return {
            'variance': variance > thresholds['variance'],
            'entropy': entropy > thresholds['entropy'],
            'sparsity': sparsity < thresholds['sparsity']
        }
