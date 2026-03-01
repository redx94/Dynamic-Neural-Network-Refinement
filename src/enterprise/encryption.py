"""
Enhanced security module with encryption for enterprise deployments.
Implements secure model storage, encrypted inference, and threat detection.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import hashlib
import hmac
import os
import json
import time
from dataclasses import dataclass
from enum import Enum
import secrets


class ThreatLevel(Enum):
    """Threat severity levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ThreatReport:
    """Report of detected threats."""
    level: ThreatLevel
    threat_type: str
    description: str
    confidence: float
    timestamp: float
    metadata: Dict[str, Any]


class SecureRandom:
    """Cryptographically secure random number generator."""

    @staticmethod
    def generate_key(length: int = 32) -> bytes:
        """Generate a secure random key."""
        return secrets.token_bytes(length)

    @staticmethod
    def generate_nonce() -> bytes:
        """Generate a random nonce for encryption."""
        return secrets.token_bytes(16)


class SimpleCipher:
    """
    Simple encryption for model weights (demo purposes).
    For production, use proper cryptographic libraries like cryptography.io.
    """

    def __init__(self, key: Optional[bytes] = None):
        self.key = key or SecureRandom.generate_key()

    def _xor_bytes(self, data: bytes, key: bytes) -> bytes:
        """XOR bytes with key (repeating key if necessary)."""
        key_extended = (key * ((len(data) // len(key)) + 1))[:len(data)]
        return bytes(a ^ b for a, b in zip(data, key_extended))

    def encrypt(self, data: bytes) -> bytes:
        """Encrypt data using XOR cipher with key stretching."""
        # Add nonce
        nonce = SecureRandom.generate_nonce()
        stretched_key = hashlib.sha256(self.key + nonce).digest()

        encrypted = self._xor_bytes(data, stretched_key)
        return nonce + encrypted

    def decrypt(self, encrypted_data: bytes) -> bytes:
        """Decrypt data."""
        nonce = encrypted_data[:16]
        ciphertext = encrypted_data[16:]

        stretched_key = hashlib.sha256(self.key + nonce).digest()
        return self._xor_bytes(ciphertext, stretched_key)

    def encrypt_tensor(self, tensor: torch.Tensor) -> bytes:
        """Encrypt a tensor."""
        # Serialize tensor
        buffer = tensor.numpy().tobytes()
        shape_info = json.dumps(list(tensor.shape)).encode()
        dtype_info = str(tensor.dtype).encode()

        # Pack: shape_len(4) + shape + dtype_len(4) + dtype + data
        packed = (
            len(shape_info).to_bytes(4, 'big') + shape_info +
            len(dtype_info).to_bytes(4, 'big') + dtype_info +
            buffer
        )
        return self.encrypt(packed)

    def decrypt_tensor(self, encrypted: bytes) -> torch.Tensor:
        """Decrypt a tensor."""
        packed = self.decrypt(encrypted)

        shape_len = int.from_bytes(packed[:4], 'big')
        shape = json.loads(packed[4:4+shape_len].decode())

        dtype_start = 4 + shape_len
        dtype_len = int.from_bytes(packed[dtype_start:dtype_start+4], 'big')
        dtype_str = packed[dtype_start+4:dtype_start+4+dtype_len].decode()

        data_start = dtype_start + 4 + dtype_len
        data = packed[data_start:]

        # Map dtype string back to numpy dtype
        dtype_map = {
            'torch.float32': np.float32,
            'torch.float64': np.float64,
            'torch.int32': np.int32,
            'torch.int64': np.int64,
        }

        arr = np.frombuffer(data, dtype=dtype_map.get(dtype_str, np.float32))
        return torch.from_numpy(arr.reshape(shape))


class SecureModelStorage:
    """
    Secure storage for model weights with encryption and integrity verification.
    """

    def __init__(
        self,
        storage_path: str = 'secure_models',
        encryption_key: Optional[bytes] = None
    ):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        self.cipher = SimpleCipher(encryption_key)

    def save_model(
        self,
        model: nn.Module,
        name: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Securely save a model with encryption.

        Returns:
            Path to saved model
        """
        model_path = os.path.join(self.storage_path, f"{name}.enc")
        meta_path = os.path.join(self.storage_path, f"{name}.meta")

        # Get model state dict
        state_dict = model.state_dict()

        # Encrypt each tensor
        encrypted_state = {}
        checksums = {}

        for key, tensor in state_dict.items():
            encrypted_tensor = self.cipher.encrypt_tensor(tensor)
            encrypted_state[key] = encrypted_tensor
            checksums[key] = hashlib.sha256(tensor.numpy().tobytes()).hexdigest()

        # Save encrypted state
        with open(model_path, 'wb') as f:
            for key, encrypted in encrypted_state.items():
                key_bytes = key.encode()
                f.write(len(key_bytes).to_bytes(4, 'big'))
                f.write(key_bytes)
                f.write(len(encrypted).to_bytes(8, 'big'))
                f.write(encrypted)

        # Save metadata and checksums
        meta = {
            'name': name,
            'timestamp': time.time(),
            'checksums': checksums,
            'metadata': metadata or {},
            'model_class': model.__class__.__name__
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        return model_path

    def load_model(
        self,
        model: nn.Module,
        name: str,
        verify_integrity: bool = True
    ) -> nn.Module:
        """
        Load and decrypt a model.

        Args:
            model: Model instance to load weights into
            name: Name of saved model
            verify_integrity: Whether to verify checksums

        Returns:
            Model with loaded weights
        """
        model_path = os.path.join(self.storage_path, f"{name}.enc")
        meta_path = os.path.join(self.storage_path, f"{name}.meta")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {name}")

        # Load metadata
        with open(meta_path, 'r') as f:
            meta = json.load(f)

        # Load and decrypt state dict
        state_dict = {}
        with open(model_path, 'rb') as f:
            while True:
                key_len_bytes = f.read(4)
                if not key_len_bytes:
                    break

                key_len = int.from_bytes(key_len_bytes, 'big')
                key = f.read(key_len).decode()

                encrypted_len = int.from_bytes(f.read(8), 'big')
                encrypted = f.read(encrypted_len)

                tensor = self.cipher.decrypt_tensor(encrypted)

                # Verify integrity
                if verify_integrity and key in meta['checksums']:
                    expected_checksum = meta['checksums'][key]
                    actual_checksum = hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
                    if expected_checksum != actual_checksum:
                        raise ValueError(f"Integrity check failed for parameter: {key}")

                state_dict[key] = tensor

        model.load_state_dict(state_dict)
        return model


class AdvancedAnomalyDetector:
    """
    Advanced anomaly detection for adversarial attacks and data poisoning.
    """

    def __init__(
        self,
        variance_threshold: float = 2.5,
        entropy_threshold: float = 6.0,
        sparsity_threshold: float = 0.9,
        enable_ml_detection: bool = True
    ):
        self.variance_threshold = variance_threshold
        self.entropy_threshold = entropy_threshold
        self.sparsity_threshold = sparsity_threshold
        self.enable_ml_detection = enable_ml_detection

        # Statistical baselines (updated during normal operation)
        self.baselines = {
            'variance_mean': 0.5,
            'variance_std': 0.2,
            'entropy_mean': 2.0,
            'entropy_std': 0.5,
            'sparsity_mean': 0.3,
            'sparsity_std': 0.1
        }

        # Detection history
        self.threat_history: List[ThreatReport] = []

    def update_baselines(self, complexities: Dict[str, torch.Tensor]):
        """Update statistical baselines with new normal data."""
        variance = complexities['variance'].mean().item()
        entropy = complexities['entropy'].mean().item()
        sparsity = complexities['sparsity'].mean().item()

        # Exponential moving average update
        alpha = 0.01

        self.baselines['variance_mean'] = (1 - alpha) * self.baselines['variance_mean'] + alpha * variance
        self.baselines['entropy_mean'] = (1 - alpha) * self.baselines['entropy_mean'] + alpha * entropy
        self.baselines['sparsity_mean'] = (1 - alpha) * self.baselines['sparsity_mean'] + alpha * sparsity

        # Update std estimates
        self.baselines['variance_std'] = (1 - alpha) * self.baselines['variance_std'] + alpha * abs(variance - self.baselines['variance_mean'])
        self.baselines['entropy_std'] = (1 - alpha) * self.baselines['entropy_std'] + alpha * abs(entropy - self.baselines['entropy_mean'])
        self.baselines['sparsity_std'] = (1 - alpha) * self.baselines['sparsity_std'] + alpha * abs(sparsity - self.baselines['sparsity_mean'])

    def detect_fgsm(self, complexities: Dict[str, torch.Tensor]) -> ThreatReport:
        """Detect Fast Gradient Sign Method (FGSM) attacks."""
        variance = complexities['variance'].mean().item()
        entropy = complexities['entropy'].mean().item()

        # FGSM typically causes high variance and entropy spikes
        variance_zscore = (variance - self.baselines['variance_mean']) / (self.baselines['variance_std'] + 1e-6)
        entropy_zscore = (entropy - self.baselines['entropy_mean']) / (self.baselines['entropy_std'] + 1e-6)

        # Combined score
        fgsm_score = (variance_zscore + entropy_zscore) / 2

        if fgsm_score > 3.0:
            level = ThreatLevel.HIGH if fgsm_score > 5.0 else ThreatLevel.MEDIUM
            return ThreatReport(
                level=level,
                threat_type='FGSM_ATTACK',
                description=f'Signature of FGSM adversarial attack detected (z-score: {fgsm_score:.2f})',
                confidence=min(1.0, fgsm_score / 5.0),
                timestamp=time.time(),
                metadata={'variance_zscore': variance_zscore, 'entropy_zscore': entropy_zscore}
            )

        return ThreatReport(
            level=ThreatLevel.LOW,
            threat_type='NORMAL',
            description='No FGSM signature detected',
            confidence=1.0,
            timestamp=time.time(),
            metadata={}
        )

    def detect_data_poisoning(
        self,
        complexities: Dict[str, torch.Tensor],
        batch_stats: Optional[Dict] = None
    ) -> ThreatReport:
        """Detect potential data poisoning attempts."""
        sparsity = complexities['sparsity'].mean().item()

        # Data poisoning often causes unusual sparsity patterns
        if batch_stats:
            batch_variance = batch_stats.get('batch_variance', 0)
            if batch_variance > 2.0 * self.baselines.get('batch_variance_mean', 1.0):
                return ThreatReport(
                    level=ThreatLevel.HIGH,
                    threat_type='DATA_POISONING',
                    description='Unusual batch variance detected, possible label flipping attack',
                    confidence=0.8,
                    timestamp=time.time(),
                    metadata={'batch_variance': batch_variance}
                )

        # Check for extreme sparsity (possible backdoor triggers)
        if sparsity > self.sparsity_threshold:
            return ThreatReport(
                level=ThreatLevel.MEDIUM,
                threat_type='POTENTIAL_BACKDOOR',
                description=f'Extreme sparsity detected ({sparsity:.3f}), possible backdoor trigger',
                confidence=0.6,
                timestamp=time.time(),
                metadata={'sparsity': sparsity}
            )

        return ThreatReport(
            level=ThreatLevel.LOW,
            threat_type='NORMAL',
            description='No data poisoning signature detected',
            confidence=1.0,
            timestamp=time.time(),
            metadata={}
        )

    def detect_model_extraction(
        self,
        query_patterns: List[Dict],
        time_window: float = 60.0
    ) -> ThreatReport:
        """Detect model extraction attacks based on query patterns."""
        if not query_patterns:
            return ThreatReport(
                level=ThreatLevel.LOW,
                threat_type='NORMAL',
                description='Insufficient data for model extraction detection',
                confidence=1.0,
                timestamp=time.time(),
                metadata={}
            )

        # Analyze recent queries
        current_time = time.time()
        recent_queries = [q for q in query_patterns if current_time - q['timestamp'] < time_window]

        if len(recent_queries) < 10:
            return ThreatReport(
                level=ThreatLevel.LOW,
                threat_type='NORMAL',
                description='Insufficient recent queries for analysis',
                confidence=1.0,
                timestamp=time.time(),
                metadata={'query_count': len(recent_queries)}
            )

        # Check for systematic probing patterns
        input_diversity = len(set(q.get('input_hash', '') for q in recent_queries)) / len(recent_queries)

        # High query rate with diverse inputs suggests extraction
        query_rate = len(recent_queries) / time_window

        if query_rate > 10 and input_diversity > 0.8:
            return ThreatReport(
                level=ThreatLevel.HIGH,
                threat_type='MODEL_EXTRACTION',
                description=f'Possible model extraction: {query_rate:.1f} queries/sec with {input_diversity:.1%} diversity',
                confidence=min(1.0, query_rate / 20),
                timestamp=time.time(),
                metadata={'query_rate': query_rate, 'input_diversity': input_diversity}
            )

        return ThreatReport(
            level=ThreatLevel.LOW,
            threat_type='NORMAL',
            description='Normal query pattern',
            confidence=1.0,
            timestamp=time.time(),
            metadata={'query_rate': query_rate, 'input_diversity': input_diversity}
        )

    def analyze(self, complexities: Dict[str, torch.Tensor]) -> List[ThreatReport]:
        """Run all threat detection analyses."""
        reports = []

        # FGSM detection
        fgsm_report = self.detect_fgsm(complexities)
        if fgsm_report.level != ThreatLevel.LOW:
            reports.append(fgsm_report)

        # Data poisoning detection
        poisoning_report = self.detect_data_poisoning(complexities)
        if poisoning_report.level != ThreatLevel.LOW:
            reports.append(poisoning_report)

        # Update baselines if no threats detected
        if not reports:
            self.update_baselines(complexities)

        # Store in history
        self.threat_history.extend(reports)

        return reports


class SecureInference:
    """
    Secure inference with encrypted data handling.
    """

    def __init__(
        self,
        model: nn.Module,
        cipher: Optional[SimpleCipher] = None
    ):
        self.model = model
        self.cipher = cipher or SimpleCipher()
        self.anomaly_detector = AdvancedAnomalyDetector()
        self.query_log: List[Dict] = []

    def secure_forward(
        self,
        encrypted_input: bytes,
        verify_input: bool = True
    ) -> bytes:
        """
        Perform secure inference on encrypted input.

        Args:
            encrypted_input: Encrypted input tensor
            verify_input: Whether to run anomaly detection

        Returns:
            Encrypted output tensor
        """
        # Decrypt input
        input_tensor = self.cipher.decrypt_tensor(encrypted_input)

        # Anomaly detection
        if verify_input:
            from src.analyzer import Analyzer
            analyzer = Analyzer()
            complexities = analyzer.analyze(input_tensor)

            threats = self.anomaly_detector.analyze(complexities)
            high_severity = [t for t in threats if t.level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]]

            if high_severity:
                raise SecurityError(f"Threat detected: {high_severity[0].threat_type}")

        # Run inference
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_tensor)

        # Log query (for extraction detection)
        input_hash = hashlib.sha256(input_tensor.numpy().tobytes()).hexdigest()[:16]
        self.query_log.append({
            'timestamp': time.time(),
            'input_hash': input_hash,
            'output_shape': list(output.shape)
        })

        # Trim log to prevent memory bloat
        if len(self.query_log) > 1000:
            self.query_log = self.query_log[-500:]

        # Encrypt and return output
        return self.cipher.encrypt_tensor(output)


class SecurityError(Exception):
    """Exception raised when security violation is detected."""
    pass


class APIKeyManager:
    """
    Secure API key management with rate limiting and key rotation.
    """

    def __init__(
        self,
        key_rotation_days: int = 30,
        max_requests_per_hour: int = 1000
    ):
        self.key_rotation_days = key_rotation_days
        self.max_requests_per_hour = max_requests_per_hour

        # Key storage (in production, use proper key-value store)
        self.keys: Dict[str, Dict] = {}
        self.request_log: Dict[str, List[float]] = {}

    def generate_api_key(
        self,
        client_id: str,
        permissions: List[str] = None,
        expires_days: int = 365
    ) -> str:
        """Generate a new API key for a client."""
        import base64

        # Generate random key
        raw_key = SecureRandom.generate_key(32)
        api_key = base64.urlsafe_b64encode(raw_key).decode()

        self.keys[api_key] = {
            'client_id': client_id,
            'permissions': permissions or ['read', 'inference'],
            'created_at': time.time(),
            'expires_at': time.time() + (expires_days * 86400),
            'last_rotation': time.time(),
            'active': True
        }

        return api_key

    def validate_key(self, api_key: str, required_permission: str = None) -> Tuple[bool, str]:
        """
        Validate an API key.

        Returns:
            Tuple of (is_valid, message)
        """
        if api_key not in self.keys:
            return False, "Invalid API key"

        key_info = self.keys[api_key]

        # Check if active
        if not key_info['active']:
            return False, "API key is disabled"

        # Check expiration
        if time.time() > key_info['expires_at']:
            return False, "API key has expired"

        # Check permissions
        if required_permission and required_permission not in key_info['permissions']:
            return False, f"Permission denied: {required_permission} required"

        # Check rate limit
        if not self._check_rate_limit(api_key):
            return False, "Rate limit exceeded"

        return True, "Valid"

    def _check_rate_limit(self, api_key: str) -> bool:
        """Check if rate limit is exceeded."""
        current_time = time.time()
        hour_ago = current_time - 3600

        # Initialize log if needed
        if api_key not in self.request_log:
            self.request_log[api_key] = []

        # Clean old requests
        self.request_log[api_key] = [t for t in self.request_log[api_key] if t > hour_ago]

        # Check limit
        if len(self.request_log[api_key]) >= self.max_requests_per_hour:
            return False

        # Log this request
        self.request_log[api_key].append(current_time)
        return True

    def rotate_key(self, old_key: str) -> str:
        """Rotate an API key (generate new, invalidate old)."""
        if old_key not in self.keys:
            raise ValueError("Invalid key to rotate")

        client_id = self.keys[old_key]['client_id']
        permissions = self.keys[old_key]['permissions']

        # Generate new key
        new_key = self.generate_api_key(client_id, permissions)

        # Deactivate old key
        self.keys[old_key]['active'] = False

        return new_key

    def revoke_key(self, api_key: str):
        """Revoke an API key."""
        if api_key in self.keys:
            self.keys[api_key]['active'] = False
