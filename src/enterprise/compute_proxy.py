import torch

class ComputeProxy:
    """
    Acts as an Enterprise Gateway Proxy. 
    Decides whether to route the request to a local lightweight model (Shallow Path equivalent)
    or a heavy Cloud API (Deep Path equivalent) based on the calculated complexity.
    This saves companies millions of dollars in compute costs for simple queries.
    """
    def __init__(self, thresholds):
        self.variance_threshold = thresholds.get("variance", 0.5)
        self.entropy_threshold = thresholds.get("entropy", 0.5)
        self.sparsity_threshold = thresholds.get("sparsity", 0.5)

    def route_request(self, complexities: dict) -> str:
        """
        Determines the routing destination based on live tensor complexity.
        Returns "CLOUD_DEEP_ENGINE" for complex data or "LOCAL_SHALLOW_ENGINE" for simple data.
        """
        variance = complexities.get("variance", torch.tensor(0.0)).mean().item()
        entropy = complexities.get("entropy", torch.tensor(0.0)).mean().item()
        sparsity = complexities.get("sparsity", torch.tensor(1.0)).mean().item()

        is_complex = (
            variance > self.variance_threshold and
            entropy > self.entropy_threshold and
            sparsity < self.sparsity_threshold
        )
        return "CLOUD_DEEP_ENGINE" if is_complex else "LOCAL_SHALLOW_ENGINE"
