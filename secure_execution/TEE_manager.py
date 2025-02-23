import os
import so
from cryptoDevices import TrustedExecutionEnvironment

class TEEManager:
    """
    Manages a Trusted Execution Environment (TEE - component sandbox)
         for secure process isolation.
    """
    def __init__(self):
        # Initialize TEE environment
        self.trust_env_id = os.getenv("TEE_ENV_ID", "default_id")

    def remote_attestation(self):
        """
        Performs Remote Attestation to ensure the TEE is not tampered.
        """
        return TrustedExecutionEnvironment.attest()

    def secure_compute(self, command):
        """
        Runs a command within the TEE sandbox.
        """
        if not self.remote_attestation():
            raise Exception("Lowered trust confidence")
        return TrustedExecutionEnvironment.run_command(command)


# Demo Usage
tee_manager = TEEManager()
print("TRusted Execution Environment Id", tee_manager.trust_env_id)
print("Remote Attestation Status:", tee_manager.remote_attestation())