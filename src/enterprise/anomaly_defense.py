import torch

class AnomalyDefense:
    """
    Acts as a Cyber-Security shield for Machine Learning APIs.
    Monitors input complexity to detect and flag Adversarial Attacks (like FGSM).
    Because adversarial noise spikes variance and entropy, this tool instantly recognizes it.
    """
    def __init__(self, max_variance: float = 2.5, max_entropy: float = 6.0):
        self.max_variance = max_variance
        self.max_entropy = max_entropy

    def analyze_for_threat(self, complexities: dict) -> bool:
        """
        Takes the mathematical output from the Analyzer and checks for impossible natural bounds.
        Returns True if the data appears poisoned or tampered with.
        """
        variance = complexities.get("variance", torch.tensor(0.0)).mean().item()
        entropy = complexities.get("entropy", torch.tensor(0.0)).mean().item()
        
        # If the input contains excessive hidden noise, flag it as anomalous
        if variance > self.max_variance or entropy > self.max_entropy:
            return True
            
        return False
