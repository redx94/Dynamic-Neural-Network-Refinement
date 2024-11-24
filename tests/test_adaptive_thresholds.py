
import pytest
from src.adaptive_thresholds import HybridThresholds

@pytest.fixture
def sample_data():
    return torch.randn(10, 5), torch.randint(0, 3, (10,))

def test_threshold_forward(sample_data):
    data, labels = sample_data
    model = HybridThresholds({'variance': 0.5, 'entropy': 0.5, 'sparsity': 0.5}, 10, 100)
    output = model(data, labels, data, 1)
    assert output is not None
    