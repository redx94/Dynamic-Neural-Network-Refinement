import torch
from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np
from cryptography.fernet import Fernet

@dataclass
class ClientState:
    client_id: str
    model_version: str
    performance_metrics: Dict[str, float]
    last_update: float
    trust_score: float

class FederatedCoordinator:
    def __init__(self, 
                 min_clients: int = 3,
                 aggregation_threshold: float = 0.7,
                 encryption_key: Optional[str] = None):
        self.min_clients = min_clients
        self.aggregation_threshold = aggregation_threshold
        self.clients: Dict[str, ClientState] = {}
        self.encryption = Fernet(encryption_key.encode()) if encryption_key else None
        
    def register_client(self, client_id: str) -> str:
        """Register a new client and return authentication token."""
        token = self._generate_secure_token()
        self.clients[client_id] = ClientState(
            client_id=client_id,
            model_version="initial",
            performance_metrics={},
            last_update=0.0,
            trust_score=1.0
        )
        return token
        
    def aggregate_models(self, 
                        client_updates: List[Tuple[str, torch.nn.Module]]) -> Optional[Dict]:
        if len(client_updates) < self.min_clients:
            return None
            
        # Verify client trust scores
        trusted_updates = [
            (cid, model) for cid, model in client_updates
            if self.clients[cid].trust_score >= self.aggregation_threshold
        ]
        
        if not trusted_updates:
            return None
            
        # Perform secure aggregation
        aggregated_state = self._secure_aggregate([
            model.state_dict() for _, model in trusted_updates
        ])
        
        # Update client trust scores
        self._update_trust_scores(trusted_updates)
        
        return aggregated_state
