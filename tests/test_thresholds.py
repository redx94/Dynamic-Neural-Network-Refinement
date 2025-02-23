import torch
from src.hybrid_thresholds import HybridThresholds

def test_hybrid_thresholds():
    thresholds = {
        'variance': 0.5,
        'entropy': 0.7,
        'sparsity': 0.9,
    }
    model = HybridThresholds(thresholds, annealing_start_epoch=5, total_epochs=50)
    var, ent, spar = torch.rand(100), torch.rand(100), torch.rand(100)
    result = model(var, ent, spar, current_epoch=10)
    assert all([key in result for key in ['variance', 'entropy', 'sparsity']]), "Keys missing in result"
    print("All tests passed!")
