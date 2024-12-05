
import pytest
from src.per_sample_complexity import process_batch_dynamic

@pytest.fixture
def mock_model():
    class MockModel:
        output_layer = nn.Linear(10, 3)
        def __call__(self, data, complexity):
            return torch.randn(data.size(0), 3)
    return MockModel()

def test_process_batch(mock_model):
    data = torch.randn(20, 10)
    complexities = torch.randint(0, 3, (20,))
    result = process_batch_dynamic(mock_model, data, complexities, 'cpu')
    assert result.size(0) == data.size(0)
    