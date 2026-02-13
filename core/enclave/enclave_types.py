"""
Shared types and interfaces for enclave implementations.

These types are used by NitroEnclaveClient (production) and
throughout the codebase for enclave operations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncGenerator, Dict, List, Optional, Tuple

from core.crypto import EncryptedPayload


# =============================================================================
# Data Structures
# =============================================================================


@dataclass(frozen=True)
class DecryptedMessage:
    """
    A message after decryption in the enclave.

    This structure only exists within the enclave's memory.
    It is NEVER sent to the server or persisted unencrypted.

    Attributes:
        role: "user" or "assistant"
        content: Plaintext message content
    """

    role: str
    content: str


@dataclass(frozen=True)
class ProcessedMessage:
    """
    Result of enclave processing a user message.

    Contains:
    - Encrypted user message for storage (to user/org key)
    - Encrypted assistant response for storage (to user/org key)
    - Encrypted assistant response for transport (to user key)
    - Model and token usage metadata

    The server stores the first two but cannot read them.
    The client receives the third and decrypts it.
    """

    # For database storage (encrypted to user's or org's storage key)
    stored_user_message: EncryptedPayload
    stored_assistant_message: EncryptedPayload

    # For transport back to client (encrypted to user's key)
    transport_response: EncryptedPayload

    # Metadata (not encrypted - for billing/logging)
    model_used: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class AgentStreamChunk:
    """
    A chunk of streaming response from the enclave for agent chat.

    Key difference from StreamChunk: no stored_user_message/stored_assistant_message
    (agent state IS the storage). Instead has encrypted_state (the updated tarball).

    For zero_trust mode: encrypted_state contains the EncryptedPayload (ECDH to user key).
    For background mode: kms_envelope contains the KMS envelope dict (hex strings).
    """

    encrypted_content: Optional[EncryptedPayload] = None  # streaming text chunk
    encrypted_state: Optional[EncryptedPayload] = None  # updated tarball - zero_trust only
    kms_envelope: Optional[Dict[str, str]] = None  # KMS envelope - background only
    is_final: bool = False
    error: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class StreamChunk:
    """
    A chunk of streaming response from the enclave.

    Used for SSE streaming where each chunk may contain:
    - encrypted_content: Encrypted chunk for client (during streaming)
    - encrypted_thinking: Encrypted thinking process chunk (during streaming)
    - stored_messages: Final encrypted messages for storage (at end)
    - is_final: True for the last chunk
    - error: Error message if something went wrong
    """

    encrypted_content: Optional[EncryptedPayload] = None
    encrypted_thinking: Optional[EncryptedPayload] = None
    stored_user_message: Optional[EncryptedPayload] = None
    stored_assistant_message: Optional[EncryptedPayload] = None
    model_used: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    is_final: bool = False
    error: Optional[str] = None


@dataclass(frozen=True)
class EnclaveInfo:
    """
    Public information about the enclave.

    This is returned to clients so they know how to encrypt
    messages to the enclave.

    Attributes:
        enclave_public_key: X25519 public key for encrypting to enclave
        attestation_document: AWS Nitro attestation (None in development)
    """

    enclave_public_key: bytes
    attestation_document: Optional[bytes] = None

    def to_hex_dict(self) -> Dict[str, Optional[str]]:
        """Convert to hex-encoded dict for API response."""
        return {
            "enclave_public_key": self.enclave_public_key.hex(),
            "attestation_document": (self.attestation_document.hex() if self.attestation_document else None),
        }


@dataclass
class AgentRunResponse:
    """Response from enclave run_agent operation.

    For background mode, kms_envelope contains the full KMS-encrypted state (hex strings).
    For zero_trust mode, encrypted_state contains state encrypted to user's key.
    """

    success: bool
    encrypted_response: Optional[EncryptedPayload] = None
    encrypted_state: Optional[EncryptedPayload] = None  # zero_trust mode
    encrypted_dek: Optional[bytes] = None  # Deprecated: use kms_envelope instead
    kms_envelope: Optional[Dict[str, str]] = None  # background mode: hex-encoded KMS envelope
    error: str = ""


# =============================================================================
# Enclave Interface
# =============================================================================


class EnclaveInterface(ABC):
    """
    Abstract interface for enclave operations.

    Production uses NitroEnclaveClient (AWS Nitro Enclave via vsock).
    """

    @abstractmethod
    def get_info(self) -> EnclaveInfo:
        """Get enclave's public key and attestation document."""
        pass

    @abstractmethod
    def decrypt_transport_message(
        self,
        payload: EncryptedPayload,
    ) -> bytes:
        """
        Decrypt a message encrypted to the enclave's transport key.

        Args:
            payload: Encrypted message from client

        Returns:
            Decrypted plaintext bytes

        Raises:
            DecryptionError: If decryption fails
        """
        pass

    @abstractmethod
    def encrypt_for_storage(
        self,
        plaintext: bytes,
        storage_public_key: bytes,
        is_assistant: bool,
    ) -> EncryptedPayload:
        """
        Encrypt a message for long-term storage.

        Args:
            plaintext: Message content
            storage_public_key: User's or org's public key
            is_assistant: True for assistant messages, False for user messages

        Returns:
            Encrypted payload for database storage
        """
        pass

    @abstractmethod
    def encrypt_for_transport(
        self,
        plaintext: bytes,
        recipient_public_key: bytes,
    ) -> EncryptedPayload:
        """
        Encrypt a response for transport back to client.

        Args:
            plaintext: Response content
            recipient_public_key: Client's public key

        Returns:
            Encrypted payload for transport
        """
        pass

    @abstractmethod
    async def process_message(
        self,
        encrypted_message: EncryptedPayload,
        encrypted_history: List[EncryptedPayload],
        storage_public_key: bytes,
        transport_public_key: bytes,
        model: str,
    ) -> ProcessedMessage:
        """
        Process a complete chat message through the enclave.

        This is the main entry point that:
        1. Decrypts the message and history
        2. Calls LLM inference
        3. Re-encrypts everything for storage and transport

        Args:
            encrypted_message: User's message encrypted to enclave
            encrypted_history: Previous messages encrypted to enclave
            storage_public_key: Key for storage encryption (user or org)
            transport_public_key: Key for response transport (always user)
            model: LLM model identifier

        Returns:
            ProcessedMessage with all encrypted outputs
        """
        pass

    @abstractmethod
    async def process_message_stream(
        self,
        encrypted_message: EncryptedPayload,
        encrypted_history: List[EncryptedPayload],
        storage_public_key: bytes,
        transport_public_key: bytes,
        model: str,
    ) -> AsyncGenerator[Tuple[str, Optional[ProcessedMessage]], None]:
        """
        Process a chat message with streaming response.

        Yields tuples of (chunk, final_result):
        - During streaming: (chunk, None)
        - Final yield: ("", ProcessedMessage)

        The final ProcessedMessage contains fully encrypted versions
        for storage after streaming completes.
        """
        pass

    @abstractmethod
    async def run_agent(
        self,
        encrypted_message: EncryptedPayload,
        encrypted_state: Optional[EncryptedPayload],
        user_public_key: bytes,
        agent_name: str,
        model: str,
        encryption_mode: str = "zero_trust",
        kms_envelope: Optional[Dict[str, bytes]] = None,
    ) -> AgentRunResponse:
        """
        Run an OpenClaw agent with an encrypted message.

        This method:
        1. Decrypts the message and state tarball (if any)
        2. Unpacks state to tmpfs
        3. Processes agent chat via Bedrock
        4. Packs updated state
        5. Re-encrypts state and response

        Args:
            encrypted_message: User's message encrypted to enclave
            encrypted_state: For zero_trust: state encrypted to enclave transport key
                           For background: None (uses kms_envelope instead)
            user_public_key: User's public key for response encryption
            agent_name: Name of the agent to run
            model: LLM model identifier
            encryption_mode: "zero_trust" (default) or "background"
            kms_envelope: For background mode, KMS-encrypted state envelope

        Returns:
            AgentRunResponse with encrypted response and state
        """
        pass

    @abstractmethod
    async def agent_chat_streaming(
        self,
        encrypted_message: EncryptedPayload,
        encrypted_state: Optional[EncryptedPayload],
        client_public_key: bytes,
        user_public_key: bytes,
        agent_name: str,
        encrypted_soul_content: Optional[EncryptedPayload] = None,
        encryption_mode: str = "zero_trust",
        kms_envelope: Optional[Dict[str, bytes]] = None,
    ) -> AsyncGenerator[AgentStreamChunk, None]:
        """
        Process an agent chat message with streaming response.

        Yields AgentStreamChunk objects:
        - encrypted_content: Encrypted text chunk during streaming
        - encrypted_state + is_final: Updated tarball at end

        Args:
            encrypted_message: User's message encrypted to enclave
            encrypted_state: Existing agent state tarball (None for new agent)
            client_public_key: Ephemeral transport key for response chunk encryption
            user_public_key: User's long-term public key for state encryption
            agent_name: Name of the agent
            encrypted_soul_content: Optional custom SOUL.md content
            encryption_mode: "zero_trust" or "background"

        Yields:
            AgentStreamChunk objects
        """
        pass
        # Make this a generator
        yield  # type: ignore


__all__ = [
    "DecryptedMessage",
    "ProcessedMessage",
    "AgentStreamChunk",
    "StreamChunk",
    "EnclaveInfo",
    "AgentRunResponse",
    "EnclaveInterface",
]
