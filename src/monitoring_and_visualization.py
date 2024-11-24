
from captum.attr import IntegratedGradients
import torch.nn.functional as F

class MonitoringAndVisualization:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.ig = IntegratedGradients(self.model)

    def compute_feature_attributions(self, data, complexities):
        attributions = {}
        for i in range(data.size(0)):
            input = data[i].unsqueeze(0).requires_grad_(True).to(self.device)
            complexity = complexities[i].unsqueeze(0).to(self.device)
            def forward_func(x): return F.cross_entropy(self.model(x, complexity), torch.tensor([0]).to(self.device))
            attribution, _ = self.ig.attribute(input, target=0, return_convergence_delta=True)
            attributions[i] = attribution.cpu().detach().numpy()
        return attributions
    