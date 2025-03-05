import pytest
import os
from src.federated.federated_coordinator import FederatedCoordinator

def test_federated_coordinator_no_encryption_key():
    with pytest.raises(ValueError, match="FEDERATED_ENCRYPTION_KEY environment variable must be set"):
        FederatedCoordinator()

def test_federated_coordinator_short_encryption_key():
    with pytest.raises(ValueError, match="FEDERATED_ENCRYPTION_KEY must be at least 32 bytes long"):
        os.environ["FEDERATED_ENCRYPTION_KEY"] = "short_key"
        FederatedCoordinator()
    del os.environ["FEDERATED_ENCRYPTION_KEY"]

FERNET_KEY = "sCnLNN125etQwnXeca1kfNlLTqhcdsa0Lz0OnDWhGM8="

def test_generate_secure_token():
    os.environ["FEDERATED_ENCRYPTION_KEY"] = FERNET_KEY
    coordinator = FederatedCoordinator()
    token1 = coordinator._generate_secure_token()
    token2 = coordinator._generate_secure_token()
    assert token1 != token2
    del os.environ["FEDERATED_ENCRYPTION_KEY"]

def test_penalize_client():
    os.environ["FEDERATED_ENCRYPTION_KEY"] = FERNET_KEY
    coordinator = FederatedCoordinator()
    client_id = "test_client"
    coordinator.register_client(client_id)
    initial_trust_score = coordinator.clients[client_id].trust_score
    coordinator.penalize_client(client_id, 0.1)
    assert coordinator.clients[client_id].trust_score < initial_trust_score
    del os.environ["FEDERATED_ENCRYPTION_KEY"]
