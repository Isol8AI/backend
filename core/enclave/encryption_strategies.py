"""
Encryption strategy pattern for agent state handling.

Two strategies:
- ZeroTrustStrategy: State encrypted to user's X25519 key. User must be online.
- BackgroundStrategy: State encrypted with KMS envelope. Enclave decrypts autonomously.

Usage:
    strategy = get_strategy(encryption_mode)
    state_dict = strategy.prepare_state_for_vsock(encrypted_state, kms_envelope)
    result = strategy.extract_state_from_response(response, encryption_mode)
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional

from core.crypto import EncryptedPayload


class EncryptionStrategy(ABC):
    """Base class for agent state encryption strategies."""

    @abstractmethod
    def prepare_state_for_vsock(
        self,
        encrypted_state: Optional[EncryptedPayload] = None,
        kms_envelope: Optional[Dict[str, bytes]] = None,
    ) -> Optional[dict]:
        """Prepare encrypted state dict for vsock transmission to enclave."""
        ...

    @abstractmethod
    def extract_state_from_response(self, response: dict) -> dict:
        """Extract encrypted state and related fields from enclave response.

        Returns dict with:
            encrypted_state: Optional[EncryptedPayload] (zero_trust)
            kms_envelope: Optional[Dict[str, str]] (background)
        """
        ...


class ZeroTrustStrategy(EncryptionStrategy):
    """User-key encryption. User must be online.

    Flow: Client decrypts state -> re-encrypts to enclave key ->
          enclave decrypts -> processes -> re-encrypts to user key
    """

    def prepare_state_for_vsock(
        self,
        encrypted_state: Optional[EncryptedPayload] = None,
        kms_envelope: Optional[Dict[str, bytes]] = None,
    ) -> Optional[dict]:
        if encrypted_state:
            return encrypted_state.to_dict()
        return None

    def extract_state_from_response(self, response: dict) -> dict:
        encrypted_state = None
        if response.get("encrypted_state"):
            encrypted_state = EncryptedPayload.from_dict(response["encrypted_state"])
        return {
            "encrypted_state": encrypted_state,
            "kms_envelope": None,
        }


class BackgroundStrategy(EncryptionStrategy):
    """KMS envelope encryption. Enclave decrypts autonomously.

    Flow: Server sends KMS envelope -> enclave calls KMS Decrypt ->
          processes -> calls KMS Encrypt -> returns new envelope
    """

    def prepare_state_for_vsock(
        self,
        encrypted_state: Optional[EncryptedPayload] = None,
        kms_envelope: Optional[Dict[str, bytes]] = None,
    ) -> Optional[dict]:
        if kms_envelope:
            return {
                "encrypted_dek": kms_envelope["encrypted_dek"].hex(),
                "iv": kms_envelope["iv"].hex(),
                "ciphertext": kms_envelope["ciphertext"].hex(),
                "auth_tag": kms_envelope["auth_tag"].hex(),
            }
        return None

    def extract_state_from_response(self, response: dict) -> dict:
        kms_envelope = None
        if response.get("encrypted_state"):
            # Background mode: encrypted_state is a raw KMS envelope dict (hex strings)
            kms_envelope = response["encrypted_state"]
        return {
            "encrypted_state": None,
            "kms_envelope": kms_envelope,
        }


def get_strategy(mode: str) -> EncryptionStrategy:
    """Get the encryption strategy for the given mode."""
    if mode == "background":
        return BackgroundStrategy()
    return ZeroTrustStrategy()
