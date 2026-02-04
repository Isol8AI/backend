"""
Mock enclave implementation for development and testing.

Security Note:
- In production, the enclave runs in an isolated AWS Nitro Enclave
- The private key NEVER leaves the enclave
- This mock implementation simulates that behavior for development

What the enclave does:
1. Receives messages encrypted to its transport public key
2. Decrypts to plaintext (only place plaintext exists)
3. Calls LLM inference
4. Re-encrypts for storage (to user/org public key)
5. Encrypts response for transport back to client
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncGenerator, Dict, List, Optional, Tuple

from core.config import settings
from core.services.credential_service import AWSCredentials, CredentialService
from core.services.bedrock_client import BedrockClientFactory
from core.crypto import (
    KeyPair,
    EncryptedPayload,
    generate_x25519_keypair,
    encrypt_to_public_key,
    decrypt_with_private_key,
)

logger = logging.getLogger(__name__)


def _debug_print(*args, **kwargs):
    """Print only when DEBUG mode is enabled."""
    if settings.DEBUG:
        print(*args, **kwargs)


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
    """

    encrypted_content: Optional[EncryptedPayload] = None  # streaming text chunk
    encrypted_state: Optional[EncryptedPayload] = None  # updated tarball (final)
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
        attestation_document: AWS Nitro attestation (None for mock)
    """

    enclave_public_key: bytes
    attestation_document: Optional[bytes] = None

    def to_hex_dict(self) -> Dict[str, Optional[str]]:
        """Convert to hex-encoded dict for API response."""
        return {
            "enclave_public_key": self.enclave_public_key.hex(),
            "attestation_document": (self.attestation_document.hex() if self.attestation_document else None),
        }


# =============================================================================
# Enclave Interface
# =============================================================================


class EnclaveInterface(ABC):
    """
    Abstract interface for enclave operations.

    This allows swapping between MockEnclave (development) and
    NitroEnclave (production) implementations.
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
    ) -> "AgentRunResponse":
        """
        Run an OpenClaw agent with an encrypted message.

        This method:
        1. Decrypts the message and state tarball (if any)
        2. Unpacks state to tmpfs
        3. Runs OpenClaw CLI
        4. Packs updated state
        5. Re-encrypts state and response

        Args:
            encrypted_message: User's message encrypted to enclave
            encrypted_state: Existing agent state tarball (None for new agent)
            user_public_key: User's public key for response encryption
            agent_name: Name of the agent to run
            model: LLM model identifier

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
        agent_name: str,
    ) -> AsyncGenerator["AgentStreamChunk", None]:
        """
        Process an agent chat message with streaming response.

        Yields AgentStreamChunk objects:
        - encrypted_content: Encrypted text chunk during streaming
        - encrypted_state + is_final: Updated tarball at end

        Args:
            encrypted_message: User's message encrypted to enclave
            encrypted_state: Existing agent state tarball (None for new agent)
            client_public_key: Client's public key for response encryption
            agent_name: Name of the agent

        Yields:
            AgentStreamChunk objects
        """
        pass
        # Make this a generator
        yield  # type: ignore


@dataclass
class AgentRunResponse:
    """Response from enclave run_agent operation."""

    success: bool
    encrypted_response: Optional[EncryptedPayload] = None
    encrypted_state: Optional[EncryptedPayload] = None
    error: str = ""


# =============================================================================
# Mock Enclave Implementation
# =============================================================================


class MockEnclave(EnclaveInterface):
    """
    Mock enclave for development and testing.

    This implementation:
    - Generates a persistent keypair on initialization
    - Simulates all enclave cryptographic operations
    - Uses the real LLM service for inference

    Security Note:
    - In development, the private key is in process memory
    - In production (Nitro), the private key never leaves the enclave
    """

    def __init__(
        self,
        aws_region: str = "us-east-1",
        inference_timeout: float = 120.0,
    ):
        """
        Initialize the mock enclave.

        Args:
            aws_region: AWS region for Bedrock
            inference_timeout: Timeout for inference requests
        """
        # Generate enclave keypair (in production, this happens inside Nitro)
        self._keypair: KeyPair = generate_x25519_keypair()

        # AWS Bedrock configuration
        self._aws_region = aws_region
        self._inference_timeout = inference_timeout

        # Default credentials (IAM role) - can be overridden per-request
        self._default_credentials = AWSCredentials(region=aws_region)

        logger.info("MockEnclave initialized with public key: %s", self._keypair.public_key.hex()[:16] + "...")

    # -------------------------------------------------------------------------
    # EnclaveInterface Implementation
    # -------------------------------------------------------------------------

    def get_info(self) -> EnclaveInfo:
        """Get enclave's public key (no attestation for mock)."""
        return EnclaveInfo(
            enclave_public_key=self._keypair.public_key,
            attestation_document=None,  # No attestation for mock
        )

    def get_transport_public_key(self) -> str:
        """Get enclave's transport public key as hex string."""
        return self._keypair.public_key.hex()

    def decrypt_transport_message(
        self,
        payload: EncryptedPayload,
    ) -> bytes:
        """
        Decrypt a message sent to the enclave.

        Uses the CLIENT_TO_ENCLAVE context for transport decryption.
        """
        from . import EncryptionContext

        return decrypt_with_private_key(
            self._keypair.private_key,
            payload,
            EncryptionContext.CLIENT_TO_ENCLAVE.value,
        )

    def encrypt_for_storage(
        self,
        plaintext: bytes,
        storage_public_key: bytes,
        is_assistant: bool,
    ) -> EncryptedPayload:
        """
        Encrypt a message for database storage.

        Uses USER_MESSAGE_STORAGE or ASSISTANT_MESSAGE_STORAGE context
        depending on the message role.
        """
        from . import EncryptionContext

        context = (
            EncryptionContext.ASSISTANT_MESSAGE_STORAGE.value
            if is_assistant
            else EncryptionContext.USER_MESSAGE_STORAGE.value
        )

        return encrypt_to_public_key(
            storage_public_key,
            plaintext,
            context,
        )

    def encrypt_for_transport(
        self,
        plaintext: bytes,
        recipient_public_key: bytes,
    ) -> EncryptedPayload:
        """
        Encrypt a response for transport back to client.

        Uses ENCLAVE_TO_CLIENT context for transport encryption.
        """
        from . import EncryptionContext

        return encrypt_to_public_key(
            recipient_public_key,
            plaintext,
            EncryptionContext.ENCLAVE_TO_CLIENT.value,
        )

    def decrypt_history_message(
        self,
        payload: EncryptedPayload,
        is_assistant: bool,
    ) -> bytes:
        """
        Decrypt a message from history (for re-sending to LLM).

        History messages are encrypted to the enclave using the same
        CLIENT_TO_ENCLAVE context (they were re-encrypted by client).

        Args:
            payload: Encrypted history message
            is_assistant: Whether this is an assistant message

        Returns:
            Decrypted plaintext
        """
        # History is re-encrypted to enclave by client, so use transport context
        from . import EncryptionContext

        return decrypt_with_private_key(
            self._keypair.private_key,
            payload,
            EncryptionContext.CLIENT_TO_ENCLAVE.value,
        )

    async def process_message(
        self,
        encrypted_message: EncryptedPayload,
        encrypted_history: List[EncryptedPayload],
        storage_public_key: bytes,
        transport_public_key: bytes,
        model: str,
    ) -> ProcessedMessage:
        """
        Process a complete chat message (non-streaming).

        Steps:
        1. Decrypt user message
        2. Decrypt history (if any)
        3. Call LLM inference
        4. Encrypt user message for storage
        5. Encrypt assistant response for storage
        6. Encrypt assistant response for transport
        """
        # 1. Decrypt the new user message
        user_plaintext = self.decrypt_transport_message(encrypted_message)
        user_content = user_plaintext.decode("utf-8")

        # 2. Decrypt history messages
        history = self._decrypt_history(encrypted_history)

        # 3. Call LLM inference
        assistant_content, input_tokens, output_tokens = await self._call_inference(
            user_content,
            history,
            model,
        )

        # 4-6. Encrypt outputs
        return self._build_processed_message(
            user_content,
            assistant_content,
            storage_public_key,
            transport_public_key,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

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

        Yields plaintext chunks during streaming (enclave has access to plaintext).
        The final yield includes the ProcessedMessage with everything encrypted.

        Note: In production, streaming chunks would also be encrypted,
        but for simplicity we encrypt the full response at the end.
        """
        # 1. Decrypt the new user message
        user_plaintext = self.decrypt_transport_message(encrypted_message)
        user_content = user_plaintext.decode("utf-8")

        # 2. Decrypt history messages
        history = self._decrypt_history(encrypted_history)

        # 3. Stream LLM inference, collecting full response
        full_response = ""
        async for chunk, is_thinking in self._call_inference_stream(user_content, history, model):
            # For non-streaming endpoint, we might just ignore thinking or concatenate?
            # Usually users of non-streaming want the final answer.
            # But wait, this method process_message_stream yields plaintext chunks.
            # Let's just yield content.
            if not is_thinking:
                full_response += chunk
                yield (chunk, None)

        # 4. Build encrypted outputs
        # Note: For streaming, we estimate tokens based on character count (~4 chars/token)
        # In production, the final SSE event would contain actual usage stats
        estimated_input_tokens = len(user_content) // 4 + sum(len(m.content) for m in history) // 4
        estimated_output_tokens = len(full_response) // 4

        processed = self._build_processed_message(
            user_content,
            full_response,
            storage_public_key,
            transport_public_key,
            model=model,
            input_tokens=estimated_input_tokens,
            output_tokens=estimated_output_tokens,
        )

        # 5. Final yield with encrypted message
        yield ("", processed)

    async def process_message_streaming(
        self,
        encrypted_message: EncryptedPayload,
        encrypted_history: List[EncryptedPayload],
        facts_context: Optional[str],
        storage_public_key: bytes,
        client_public_key: bytes,
        session_id: str,
        model: str,
        user_id: str = "",
        org_id: Optional[str] = None,
        user_metadata: Optional[dict] = None,
        org_metadata: Optional[dict] = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Process a chat message with encrypted streaming response.

        This is the main streaming API for routes. Each chunk's content
        is encrypted to the client's public key for secure transport.

        Args:
            encrypted_message: User's message encrypted to enclave
            encrypted_history: Previous messages re-encrypted to enclave
            facts_context: Client-side formatted facts context (already decrypted, plaintext)
            storage_public_key: Key for storage encryption (user or org)
            client_public_key: Client's ephemeral key for response encryption
            session_id: Session ID for logging
            model: LLM model to use

        Yields:
            StreamChunk objects with encrypted content or final stored messages
        """
        try:
            _debug_print("\n" + "=" * 80)
            _debug_print("🔐 ENCRYPTED CHAT FLOW - ENCLAVE (Backend)")
            _debug_print("=" * 80)
            _debug_print(f"Session ID: {session_id}")
            _debug_print(f"Model: {model}")

            # 1. Decrypt the new user message
            _debug_print("\n📥 STEP 1: Decrypt User Message from Client")
            _debug_print("-" * 60)
            _debug_print("Enclave Private Key: [HELD SECURELY IN ENCLAVE]")
            _debug_print("Encrypted Message:")
            _debug_print(f"  ephemeral_public_key: {encrypted_message.ephemeral_public_key.hex()[:32]}...")
            _debug_print(f"  iv: {encrypted_message.iv.hex()}")
            _debug_print(f"  ciphertext: {encrypted_message.ciphertext.hex()[:32]}...")
            _debug_print(f"  auth_tag: {encrypted_message.auth_tag.hex()}")

            user_plaintext = self.decrypt_transport_message(encrypted_message)
            user_content = user_plaintext.decode("utf-8")
            _debug_print(f"✅ Decrypted User Message: {user_content}")

            # 2. Decrypt history messages
            if encrypted_history:
                _debug_print(f"\n📥 STEP 2: Decrypt History ({len(encrypted_history)} messages)")
                _debug_print("-" * 60)
            history = self._decrypt_history(encrypted_history)
            for i, msg in enumerate(history):
                _debug_print(
                    f"  [{i}] {msg.role}: {msg.content[:50]}..."
                    if len(msg.content) > 50
                    else f"  [{i}] {msg.role}: {msg.content}"
                )

            # 2b. Log facts context if provided
            if facts_context:
                _debug_print("\n📋 STEP 2b: Session Facts Context")
                _debug_print("-" * 60)
                preview = facts_context[:200] + "..." if len(facts_context) > 200 else facts_context
                _debug_print(f"Facts context: {preview}")

            # 3. Stream LLM inference, encrypting each chunk for transport
            _debug_print("\n🤖 STEP 3: Call LLM Inference")
            _debug_print("-" * 60)
            _debug_print(f"Model: {model}")
            _debug_print(f"Messages count: {len(history) + 1}")
            _debug_print(f"Facts context: {'Yes' if facts_context else 'No'}")

            full_response = ""
            current_thinking = ""
            chunk_count = 0
            _debug_print("\n📤 STEP 4: Stream Encrypted Chunks to Client")
            _debug_print("-" * 60)
            _debug_print(f"Client Transport Public Key: {client_public_key.hex()[:32]}...")

            # Resolve credentials for this request
            credentials = CredentialService.resolve_credentials(
                user_metadata=user_metadata,
                org_metadata=org_metadata,
            )
            _debug_print(f"Using credentials: {'custom' if credentials.is_custom else 'IAM role'}")

            async for chunk, is_thinking in self._call_inference_stream(
                user_content, history, model, None, facts_context, credentials
            ):
                chunk_count += 1

                # Encrypt this chunk for transport to client
                encrypted_chunk = self.encrypt_for_transport(
                    chunk.encode("utf-8"),
                    client_public_key,
                )

                if is_thinking:
                    current_thinking += chunk
                    _debug_print(f"  Thinking Chunk {chunk_count}: '{chunk}' → encrypted")
                    yield StreamChunk(encrypted_thinking=encrypted_chunk)
                else:
                    full_response += chunk
                    _debug_print(f"  Content Chunk {chunk_count}: '{chunk}' → encrypted")
                    yield StreamChunk(encrypted_content=encrypted_chunk)

            _debug_print(f"\n✅ Total chunks streamed: {chunk_count}")
            _debug_print(
                f"Full response: {full_response[:100]}..."
                if len(full_response) > 100
                else f"Full response: {full_response}"
            )
            _debug_print(
                f"Thinking content: {current_thinking[:100]}..."
                if len(current_thinking) > 100
                else f"Thinking content: {current_thinking}"
            )

            # Handle case where model only outputs thinking content (e.g., DeepSeek R1)
            # If full_response is empty but we have thinking content, use that as the response
            if not full_response.strip() and current_thinking.strip():
                logger.warning("Model returned only thinking content, using as response")
                _debug_print("⚠️ No content outside <think> tags, using thinking content as response")
                full_response = current_thinking
            elif not full_response.strip() and not current_thinking.strip():
                # Both empty - likely an error occurred
                logger.error("Model returned no content at all")
                _debug_print("❌ No content received from model!")
                full_response = "[Error: The model did not return a response. Please try again.]"

            # 4. Build final encrypted messages for storage
            _debug_print("\n💾 STEP 5: Encrypt Messages for Storage")
            _debug_print("-" * 60)
            _debug_print(f"Storage Public Key: {storage_public_key.hex()[:32]}...")

            user_bytes = user_content.encode("utf-8")
            assistant_bytes = full_response.encode("utf-8")

            stored_user = self.encrypt_for_storage(user_bytes, storage_public_key, is_assistant=False)
            stored_assistant = self.encrypt_for_storage(assistant_bytes, storage_public_key, is_assistant=True)

            _debug_print("User message encrypted for storage:")
            _debug_print(f"  ephemeral_public_key: {stored_user.ephemeral_public_key.hex()[:32]}...")
            _debug_print(f"  ciphertext length: {len(stored_user.ciphertext)} bytes")
            _debug_print("Assistant message encrypted for storage:")
            _debug_print(f"  ephemeral_public_key: {stored_assistant.ephemeral_public_key.hex()[:32]}...")
            _debug_print(f"  ciphertext length: {len(stored_assistant.ciphertext)} bytes")

            # Estimate tokens for streaming
            estimated_input_tokens = len(user_content) // 4 + sum(len(m.content) for m in history) // 4
            estimated_output_tokens = len(full_response) // 4

            _debug_print("\n📋 FINAL SUMMARY")
            _debug_print("-" * 60)
            _debug_print(f"Input tokens (estimated): {estimated_input_tokens}")
            _debug_print(f"Output tokens (estimated): {estimated_output_tokens}")
            _debug_print("=" * 80 + "\n")

            # 6. Final chunk with stored messages
            yield StreamChunk(
                stored_user_message=stored_user,
                stored_assistant_message=stored_assistant,
                model_used=model,
                input_tokens=estimated_input_tokens,
                output_tokens=estimated_output_tokens,
                is_final=True,
            )

        except Exception as e:
            logger.exception("Streaming error in enclave")
            _debug_print(f"\n❌ ENCLAVE ERROR: {str(e)}")
            _debug_print("=" * 80 + "\n")
            yield StreamChunk(error=str(e), is_final=True)

    # -------------------------------------------------------------------------
    # Internal Helper Methods
    # -------------------------------------------------------------------------

    def _decrypt_history(
        self,
        encrypted_history: List[EncryptedPayload],
    ) -> List[DecryptedMessage]:
        """
        Decrypt conversation history.

        History is a list of alternating user/assistant messages,
        starting with the user message.
        """
        history = []
        for i, payload in enumerate(encrypted_history):
            is_assistant = i % 2 == 1  # 0=user, 1=assistant, 2=user, ...
            plaintext = self.decrypt_history_message(payload, is_assistant)
            history.append(
                DecryptedMessage(
                    role="assistant" if is_assistant else "user",
                    content=plaintext.decode("utf-8"),
                )
            )
        return history

    def _build_processed_message(
        self,
        user_content: str,
        assistant_content: str,
        storage_public_key: bytes,
        transport_public_key: bytes,
        model: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> ProcessedMessage:
        """
        Build the ProcessedMessage with all encrypted outputs.
        """
        user_bytes = user_content.encode("utf-8")
        assistant_bytes = assistant_content.encode("utf-8")

        return ProcessedMessage(
            stored_user_message=self.encrypt_for_storage(
                user_bytes,
                storage_public_key,
                is_assistant=False,
            ),
            stored_assistant_message=self.encrypt_for_storage(
                assistant_bytes,
                storage_public_key,
                is_assistant=True,
            ),
            transport_response=self.encrypt_for_transport(
                assistant_bytes,
                transport_public_key,
            ),
            model_used=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def _build_converse_request(
        self,
        user_message: str,
        history: List[DecryptedMessage],
        memories: Optional[List[str]] = None,
        facts_context: Optional[str] = None,
    ) -> Tuple[str, List[Dict]]:
        """
        Build request for AWS Bedrock Converse API.

        The Converse API provides a unified interface for all Bedrock models.
        It uses a 'system' field separate from messages, and messages contain
        'content' as an array of content blocks.

        Args:
            user_message: The current user message
            history: Previous conversation messages
            memories: Deprecated, ignored (kept for API compatibility)
            facts_context: Deprecated, ignored (kept for API compatibility)

        Returns:
            Tuple of (system_prompt, messages) for Converse API
        """
        # Build system prompt
        system_content = (
            "You are a helpful AI assistant. Note: Previous assistant "
            "messages may contain '[Response from model-name]' prefixes - "
            "these are internal metadata annotations showing which AI model "
            "generated that response. Do not include such prefixes in your "
            "own responses; just respond naturally."
        )

        # Build messages array in Converse API format
        # Each message has 'role' and 'content' (array of content blocks)
        messages = []
        for msg in history:
            messages.append(
                {
                    "role": msg.role,
                    "content": [{"text": msg.content}],
                }
            )
        messages.append(
            {
                "role": "user",
                "content": [{"text": user_message}],
            }
        )

        return system_content, messages

    async def _call_inference(
        self,
        user_message: str,
        history: List[DecryptedMessage],
        model: str,
        memories: Optional[List[str]] = None,
        facts_context: Optional[str] = None,
        credentials: Optional[AWSCredentials] = None,
    ) -> Tuple[str, int, int]:
        """
        Call LLM inference (non-streaming) using AWS Bedrock Converse API.

        Returns:
            Tuple of (response_content, input_tokens, output_tokens)
        """
        system_prompt, messages = self._build_converse_request(user_message, history, memories, facts_context)

        creds = credentials or self._default_credentials
        bedrock_client = BedrockClientFactory.create_client(creds, timeout=self._inference_timeout)

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: bedrock_client.converse(
                    modelId=model,
                    messages=messages,
                    system=[{"text": system_prompt}],
                    inferenceConfig={
                        "maxTokens": 4096,
                        "temperature": 0.7,
                    },
                ),
            )

            # Extract response content
            output_message = response.get("output", {}).get("message", {})
            content_blocks = output_message.get("content", [])
            content = ""
            for block in content_blocks:
                if "text" in block:
                    content += block["text"]

            # Extract token usage
            usage = response.get("usage", {})
            input_tokens = usage.get("inputTokens", 0)
            output_tokens = usage.get("outputTokens", 0)

            return content, input_tokens, output_tokens

        except Exception as e:
            logger.error(f"Bedrock inference error: {e}")
            raise RuntimeError(f"Inference failed: {e}")

    async def _call_inference_stream(
        self,
        user_message: str,
        history: List[DecryptedMessage],
        model: str,
        memories: Optional[List[str]] = None,
        facts_context: Optional[str] = None,
        credentials: Optional[AWSCredentials] = None,
    ) -> AsyncGenerator[Tuple[str, bool], None]:
        """
        Call AWS Bedrock LLM inference with streaming using the Converse API.

        The Converse API provides a unified interface that works with ALL
        Bedrock models (Claude, Llama, Titan, Mistral, etc.) without needing
        model-specific request formats.

        Yields tuples of (text_chunk, is_thinking).
        """
        # Build system prompt and messages
        system_prompt, messages = self._build_converse_request(user_message, history, memories, facts_context)

        # Use provided credentials or default
        creds = credentials or self._default_credentials

        print(f"[LLM] Starting Bedrock Converse stream with model: {model}")
        print(f"[LLM] Region: {creds.region}")
        print(f"[LLM] Using custom credentials: {creds.is_custom}")
        print(f"[LLM] Messages count: {len(messages)}")

        try:
            # Create Bedrock client
            bedrock_client = BedrockClientFactory.create_client(
                creds,
                timeout=self._inference_timeout,
            )

            # Call Bedrock Converse API with streaming
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: bedrock_client.converse_stream(
                    modelId=model,
                    messages=messages,
                    system=[{"text": system_prompt}],
                    inferenceConfig={
                        "maxTokens": 4096,
                        "temperature": 0.7,
                    },
                ),
            )

            # Use an async queue to bridge between the sync stream iteration
            # and the async generator. This allows proper streaming without
            # blocking the event loop.
            queue: asyncio.Queue = asyncio.Queue()

            # Sentinel value to signal end of stream
            _STREAM_END = object()
            _STREAM_ERROR = object()

            def read_stream_sync():
                """
                Read from the sync stream in a background thread.
                Puts each event into the async queue.
                """
                try:
                    for event in response["stream"]:
                        # Put the event in the queue (thread-safe with asyncio)
                        loop.call_soon_threadsafe(queue.put_nowait, event)
                    # Signal end of stream
                    loop.call_soon_threadsafe(queue.put_nowait, _STREAM_END)
                except Exception as e:
                    # Signal error
                    loop.call_soon_threadsafe(queue.put_nowait, (_STREAM_ERROR, e))

            # Start the background thread to read from the stream
            import concurrent.futures

            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            stream_future = executor.submit(read_stream_sync)

            # State for thinking tags
            is_thinking = False
            buffer = ""
            chunk_count = 0
            total_chars = 0

            # Process events from the queue asynchronously
            try:
                while True:
                    # Await the next event from the queue (non-blocking for event loop)
                    event = await queue.get()

                    # Check for end of stream
                    if event is _STREAM_END:
                        break

                    # Check for error
                    if isinstance(event, tuple) and len(event) == 2 and event[0] is _STREAM_ERROR:
                        raise event[1]

                    # contentBlockDelta contains the text chunks
                    if "contentBlockDelta" in event:
                        delta = event["contentBlockDelta"].get("delta", {})
                        content = delta.get("text", "")

                        if not content:
                            continue

                        chunk_count += 1
                        total_chars += len(content)

                        # Process thinking tags (for models that use them)
                        buffer += content

                        while buffer:
                            if not is_thinking:
                                if "<think>" in buffer:
                                    pre_think, post_think = buffer.split("<think>", 1)
                                    if pre_think:
                                        yield (pre_think, False)
                                    is_thinking = True
                                    buffer = post_think
                                else:
                                    if any(buffer.endswith(x) for x in ["<", "<t", "<th", "<thi", "<thin", "<think"]):
                                        break
                                    else:
                                        yield (buffer, False)
                                        buffer = ""
                            else:
                                if "</think>" in buffer:
                                    think_content, post_think = buffer.split("</think>", 1)
                                    if think_content:
                                        yield (think_content, True)
                                    is_thinking = False
                                    buffer = post_think
                                else:
                                    if any(
                                        buffer.endswith(x)
                                        for x in ["<", "</", "</t", "</th", "</thi", "</thin", "</think"]
                                    ):
                                        break
                                    else:
                                        yield (buffer, True)
                                        buffer = ""

                    # messageStop signals end of response
                    elif "messageStop" in event:
                        stop_reason = event["messageStop"].get("stopReason", "")
                        print(f"[LLM] Stream stopped: {stop_reason}")

                    # metadata contains token usage
                    elif "metadata" in event:
                        usage = event["metadata"].get("usage", {})
                        print(
                            f"[LLM] Usage - input: {usage.get('inputTokens', 0)}, output: {usage.get('outputTokens', 0)}"
                        )

            finally:
                # Ensure the background thread completes
                stream_future.result(timeout=5.0)
                executor.shutdown(wait=False)

            # Flush remaining buffer
            if buffer:
                print(f"[LLM] Flushing remaining buffer: {len(buffer)} chars")
                yield (buffer, is_thinking)

            print(f"[LLM] Bedrock stream completed: {chunk_count} chunks, {total_chars} total chars")

        except Exception as e:
            print(f"[LLM] Bedrock inference error: {str(e)}")
            import traceback

            traceback.print_exc()
            yield (f"Error: {str(e)}", False)

    async def agent_chat_streaming(
        self,
        encrypted_message: EncryptedPayload,
        encrypted_state: Optional[EncryptedPayload],
        client_public_key: bytes,
        agent_name: str,
    ) -> AsyncGenerator[AgentStreamChunk, None]:
        """
        Agent chat streaming is only available in Nitro Enclave mode.

        MockEnclave does not support agent streaming because agent chat
        requires Bedrock streaming which runs inside the enclave.
        """
        raise NotImplementedError(
            "agent_chat_streaming is only available in Nitro Enclave mode. "
            "Use ENCLAVE_MODE=nitro for agent streaming."
        )
        yield  # type: ignore - make this a generator

    async def run_agent(
        self,
        encrypted_message: EncryptedPayload,
        encrypted_state: Optional[EncryptedPayload],
        user_public_key: bytes,
        agent_name: str,
        model: str,
    ) -> AgentRunResponse:
        """
        Run an OpenClaw agent with an encrypted message.

        In mock mode, this tries to run the actual OpenClaw CLI if available,
        otherwise returns a mock response for testing.
        """
        import shutil
        import tempfile
        from pathlib import Path

        from . import EncryptionContext

        tmpfs_path = None

        try:
            # Create tmpfs directory
            tmpfs_base = "/tmp/openclaw"
            Path(tmpfs_base).mkdir(parents=True, exist_ok=True)
            tmpfs_path = Path(tempfile.mkdtemp(dir=tmpfs_base, prefix=f"agent_{agent_name}_"))

            logger.info(f"[MockEnclave] Running agent {agent_name} in {tmpfs_path}")

            # Decrypt and extract existing state, or create fresh agent
            if encrypted_state:
                state_bytes = decrypt_with_private_key(
                    self._keypair.private_key,
                    encrypted_state,
                    EncryptionContext.AGENT_STATE_STORAGE.value,
                )
                self._unpack_tarball(state_bytes, tmpfs_path)
                logger.info(f"[MockEnclave] Extracted existing state ({len(state_bytes)} bytes)")
            else:
                self._create_fresh_agent(tmpfs_path, agent_name, model)
                logger.info("[MockEnclave] Created fresh agent directory")

            # Decrypt user message
            message_bytes = decrypt_with_private_key(
                self._keypair.private_key,
                encrypted_message,
                EncryptionContext.CLIENT_TO_ENCLAVE.value,
            )
            message = message_bytes.decode("utf-8")
            logger.info(f"[MockEnclave] Decrypted message: {message[:50]}...")

            # Try to run OpenClaw CLI
            result = self._run_openclaw(tmpfs_path, message, agent_name, model)

            if not result["success"]:
                return AgentRunResponse(
                    success=False,
                    error=result["error"],
                )

            logger.info(f"[MockEnclave] Agent response: {result['response'][:50]}...")

            # Pack updated state
            tarball_bytes = self._pack_directory(tmpfs_path)
            logger.info(f"[MockEnclave] Packed state: {len(tarball_bytes)} bytes")

            # Encrypt state for storage (to enclave's key for future decryption)
            encrypted_state_out = encrypt_to_public_key(
                self._keypair.public_key,
                tarball_bytes,
                EncryptionContext.AGENT_STATE_STORAGE.value,
            )

            # Encrypt response for transport (to user's key)
            encrypted_response = encrypt_to_public_key(
                user_public_key,
                result["response"].encode("utf-8"),
                EncryptionContext.ENCLAVE_TO_CLIENT.value,
            )

            return AgentRunResponse(
                success=True,
                encrypted_response=encrypted_response,
                encrypted_state=encrypted_state_out,
            )

        except Exception as e:
            logger.exception(f"[MockEnclave] run_agent error: {e}")
            return AgentRunResponse(
                success=False,
                error=str(e),
            )

        finally:
            # Always cleanup tmpfs
            if tmpfs_path and tmpfs_path.exists():
                shutil.rmtree(tmpfs_path, ignore_errors=True)
                logger.debug(f"[MockEnclave] Cleaned up tmpfs: {tmpfs_path}")

    def _unpack_tarball(self, tarball_bytes: bytes, target_dir) -> None:
        """Unpack a gzip tarball to a directory."""
        import io
        import tarfile
        from pathlib import Path

        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        buffer = io.BytesIO(tarball_bytes)
        with tarfile.open(fileobj=buffer, mode="r:gz") as tar:
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    raise ValueError(f"Unsafe path in tarball: {member.name}")
            tar.extractall(target_dir)

    def _pack_directory(self, directory) -> bytes:
        """Pack a directory into a gzip tarball."""
        import io
        import tarfile
        from pathlib import Path

        directory = Path(directory)
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            for item in directory.rglob("*"):
                if item.is_file():
                    arcname = item.relative_to(directory)
                    tar.add(item, arcname=str(arcname))
        buffer.seek(0)
        return buffer.read()

    def _create_fresh_agent(self, agent_dir, agent_name: str, model: str) -> None:
        """Create a fresh OpenClaw agent directory structure."""
        import json
        from pathlib import Path

        agent_dir = Path(agent_dir)
        agent_dir.mkdir(parents=True, exist_ok=True)

        soul_content = f"""# {agent_name}

You are {agent_name}, a personal AI companion.

## Personality
- Friendly and helpful
- Remember past conversations
- Learn user preferences over time

## Guidelines
- Be concise but thorough
- Ask clarifying questions when needed
- Respect user privacy
"""

        config = {
            "version": "1.0",
            "agents": {agent_name: {"model": model}},
            "defaults": {"model": model, "agent": agent_name},
        }
        (agent_dir / "openclaw.json").write_text(json.dumps(config, indent=2))

        agent_subdir = agent_dir / "agents" / agent_name
        agent_subdir.mkdir(parents=True, exist_ok=True)
        (agent_subdir / "SOUL.md").write_text(soul_content)

        memory_dir = agent_subdir / "memory"
        memory_dir.mkdir(exist_ok=True)
        (memory_dir / "MEMORY.md").write_text("# Memories\n\nNo memories yet.\n")

        (agent_subdir / "sessions").mkdir(exist_ok=True)

    def _run_openclaw(self, agent_dir, message: str, agent_name: str, model: str, timeout: int = 120) -> dict:
        """Run the OpenClaw CLI with a message."""
        import os
        import subprocess
        from pathlib import Path

        agent_dir = Path(agent_dir)
        env = os.environ.copy()
        env["OPENCLAW_STATE_DIR"] = str(agent_dir)
        env["OPENCLAW_HOME"] = str(agent_dir)
        env["HOME"] = str(agent_dir)

        cmd = [
            "openclaw",
            "agent",
            "--message",
            message,
            "--agent",
            agent_name,
            "--model",
            model,
            "--non-interactive",
        ]

        logger.info(f"[MockEnclave] Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(agent_dir),
            )

            if result.returncode == 0:
                return {
                    "success": True,
                    "response": result.stdout.strip(),
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            else:
                # If openclaw isn't available, return a mock response
                if "not found" in result.stderr.lower() or result.returncode == 127:
                    logger.warning("[MockEnclave] openclaw CLI not found, returning mock response")
                    return {
                        "success": True,
                        "response": f"[Mock Response] Hello! I'm {agent_name}. OpenClaw CLI is not installed in the mock enclave, so this is a simulated response to: {message[:100]}",
                    }
                return {
                    "success": False,
                    "error": result.stderr.strip() or f"Exit code: {result.returncode}",
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
        except FileNotFoundError:
            # openclaw not installed - return mock response
            logger.warning("[MockEnclave] openclaw CLI not found, returning mock response")
            return {
                "success": True,
                "response": f"[Mock Response] Hello! I'm {agent_name}. OpenClaw CLI is not installed, so this is a simulated response to: {message[:100]}",
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Command timed out after {timeout} seconds",
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }


# =============================================================================
# Singleton Instance
# =============================================================================

# The enclave instance is created lazily by get_enclave()
_enclave_instance: Optional[MockEnclave] = None


def get_enclave() -> MockEnclave:
    """
    Get the singleton enclave instance.

    Creates the instance on first call, reuses it thereafter.
    This ensures the same keypair is used throughout the application.

    Returns:
        The MockEnclave singleton instance
    """
    global _enclave_instance
    if _enclave_instance is None:
        from core.config import settings

        _enclave_instance = MockEnclave(
            aws_region=settings.AWS_REGION,
            inference_timeout=settings.ENCLAVE_INFERENCE_TIMEOUT,
        )
    return _enclave_instance


def reset_enclave() -> None:
    """
    Reset the enclave singleton (for testing only).

    This forces a new keypair to be generated on next get_enclave() call.
    """
    global _enclave_instance
    _enclave_instance = None
