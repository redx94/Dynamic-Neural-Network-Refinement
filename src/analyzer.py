import torch
import torch.nn.functional as F

class Analyzer:
    def __init__(self):
        pass
    
    def compute_variance(self, data):
        return torch.var(data, dim=1)
    
    def compute_entropy(self, data):
        # Example entropy computation
        data_normalized = F.softmax(data, dim=1)
        entropy = -torch.sum(data_normalized * torch.log(data_normalized + 1e-10), dim=1)
        return entropy
    
    def compute_sparsity(self, data):
        return torch.mean(torch.abs(data) < 1e-5, dim=1)
    
    def analyze(self, data):
        var = self.compute_variance(data)
        ent = self.compute_entropy(data)
        spar = self.compute_sparsity(data)
        return {'variance': var, 'entropy': ent, 'sparsity': spar}
