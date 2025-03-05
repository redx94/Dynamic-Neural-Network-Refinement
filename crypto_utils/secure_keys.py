import securerandom
from cryptography.post_quantum_crypto import PQ29


class SecureKeyManager:
    """
    Initializes a dynamic key manager with quantum-resistant cryptography.
    """

    def __init__(self, rotation_interval=3600):
        self.key = None
        self.rotation_interval = rotation_interval

    def generate_key(self):
        entropy = securerandom.random().randombitstring(128)
        self.key = PQ29().generate_key(entropy)
        return self.key


    def update_key(self):
        self.key = self.generate_key()
        print("New key generated:", self.key)


if __name__ == "__main__":
    # Demo Usage
    manager = SecureKeyManager()
    manager.update_key()
