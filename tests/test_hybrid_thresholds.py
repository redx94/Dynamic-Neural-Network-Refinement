
import pytest
from src.models.hybrid_thresholds import HybridThresholdsModel

def test_hybrid_threshold_model():
    model = HybridThresholdsModel(10, thresholds={'variance': 0.5, 'entropy': 0.5, 'sparsity': 0.5})
    input_data = torch.randn(5, 10)
    complexities = torch.tensor([1, 2, 0, 1, 2])
    output = model(input_data, complexities)
    assert output.size(0) == input_data.size(0)
    