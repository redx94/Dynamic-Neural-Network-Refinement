import torch
import torch.distributions as dist

class PrivacyCloakLayer:
    """
    Enterprise Data Privacy Module.
    Injects epsilon-bounded Laplacian noise into the input tensor before it is sent 
    to the Cloud Engine. This prevents Man-in-the-Middle attacks from reconstructing
    raw user data (like faces or proprietary documents) while maintaining enough statistical 
    fidelity for the Deep Neural Network to accurately evaluate the payload.
    """
    
    def __init__(self, epsilon: float = 1.0, sensitivity: float = 1.0):
        """
        Args:
            epsilon: The privacy budget. Lower epsilon = more noise (more privacy, less accuracy).
            sensitivity: The maximum amount any single input pixel could change the output.
        """
        self.epsilon = epsilon
        self.sensitivity = sensitivity
        
        # Scale for the Laplace distribution (b = sensitivity / epsilon)
        self.scale = self.sensitivity / self.epsilon

    def apply_cloak(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Scrambles the tensor using Differential Privacy math.
        """
        # Create a Laplacian distribution centered at 0
        laplace = dist.Laplace(torch.tensor([0.0]), torch.tensor([self.scale]))
        
        # Sample noise matching the tensor's shape
        noise = laplace.sample(tensor.size()).squeeze(-1)
        
        # Move noise to the same device as the input tensor
        noise = noise.to(tensor.device)
        
        # Cloak the data by injecting the noise
        cloaked_tensor = tensor + noise
        
        return cloaked_tensor

# End of PrivacyCloakLayer
