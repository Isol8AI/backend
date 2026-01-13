"""
Mock enclave package for development and testing.

This package provides a mock implementation of the secure enclave that:
- Manages the enclave's X25519 transport keypair
- Decrypts messages encrypted to the enclave
- Calls LLM inference with plaintext
- Re-encrypts responses for storage and transport

In production, this will be replaced by a real AWS Nitro Enclave implementation.
"""
from enum import Enum


class EncryptionContext(str, Enum):
    """
    HKDF context strings for domain separation.

    These context strings MUST match between encryption and decryption.
    They ensure that keys derived for different purposes cannot be
    confused or misused.

    Security Note:
    - Using the wrong context will result in decryption failure
    - This prevents cross-protocol attacks
    """

    # Transport contexts (ephemeral per-request)
    CLIENT_TO_ENCLAVE = "client-to-enclave-transport"
    ENCLAVE_TO_CLIENT = "enclave-to-client-transport"

    # Storage contexts (long-term storage encryption)
    USER_MESSAGE_STORAGE = "user-message-storage"
    ASSISTANT_MESSAGE_STORAGE = "assistant-message-storage"

    # Key distribution contexts
    ORG_KEY_DISTRIBUTION = "org-key-distribution"
    RECOVERY_KEY_ENCRYPTION = "recovery-key-encryption"


# Export the mock enclave when this package is imported
from .mock_enclave import (
    MockEnclave,
    EnclaveInterface,
    ProcessedMessage,
    StreamChunk,
    EnclaveInfo,
    DecryptedMessage,
    get_enclave,
    reset_enclave,
)


async def shutdown_enclave() -> None:
    """
    Shutdown the enclave (cleanup on application exit).

    For the mock enclave, this is a no-op.
    For Nitro enclave, this would handle cleanup.
    """
    # Currently a no-op for mock enclave
    pass


__all__ = [
    "EncryptionContext",
    "MockEnclave",
    "EnclaveInterface",
    "ProcessedMessage",
    "StreamChunk",
    "EnclaveInfo",
    "DecryptedMessage",
    "get_enclave",
    "reset_enclave",
    "shutdown_enclave",
]
