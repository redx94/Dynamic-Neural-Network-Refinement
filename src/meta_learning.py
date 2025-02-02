import torch
import torch.nn as nn
from typing import Dict, List


class MetaArchitectureOptimizer(nn.Module):
    """
    Meta-learning based optimizer for dynamically adapting neural architectures.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.architecture_history: List[Dict] = []

    def forward(self, architecture_sequence: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predicts optimal architecture configurations.

        Args:
            architecture_sequence (torch.Tensor): Input architecture sequence.

        Returns:
            Tuple[torch.Tensor, Dict]: Predicted architecture and metadata.
        """
        lstm_out, (h_n, c_n) = self.lstm(architecture_sequence)
        predicted_architecture = self.predictor(lstm_out[:, -1, :])

        return predicted_architecture, {'h_n': h_n, 'c_n': c_n}

    def optimize_architecture(self, current_metrics: Dict[str, float]) -> Dict:
        """
        Optimizes the architecture based on performance metrics.

        Args:
            current_metrics (Dict[str, float]): Performance metrics.

        Returns:
            Dict: Optimized architecture configuration.
        """
        architecture_tensor = torch.tensor([
            [current_metrics['complexity'], current_metrics['performance'], current_metrics['efficiency']]
        ], dtype=torch.float32)

        predicted_config, _ = self(architecture_tensor)
        return self._decode_architecture(predicted_config[0])

    def _decode_architecture(self, config_tensor: torch.Tensor) -> Dict:
        """
        Decodes architecture configuration from tensor.

        Args:
            config_tensor (torch.Tensor): Encoded configuration.

        Returns:
            Dict: Decoded architecture configuration.
        """
        return {
            'num_layers': max(1, int(config_tensor[0].item() * 10)),
            'hidden_dim': max(32, int(config_tensor[1].item() * 512)),
            'learning_rate': float(torch.sigmoid(config_tensor[2]) * 0.01)
        }