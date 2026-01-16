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
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncGenerator, Dict, List, Optional, Tuple

import httpx

from core.crypto import (
    KeyPair,
    EncryptedPayload,
    generate_x25519_keypair,
    encrypt_to_public_key,
    decrypt_with_private_key,
)

logger = logging.getLogger(__name__)


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
class StreamChunk:
    """
    A chunk of streaming response from the enclave.

    Used for SSE streaming where each chunk may contain:
    - encrypted_content: Encrypted chunk for client (during streaming)
    - stored_messages: Final encrypted messages for storage (at end)
    - is_final: True for the last chunk
    - error: Error message if something went wrong
    """
    encrypted_content: Optional[EncryptedPayload] = None
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
            "attestation_document": (
                self.attestation_document.hex() if self.attestation_document else None
            ),
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
        inference_url: str = "https://router.huggingface.co/v1",
        inference_token: Optional[str] = None,
        inference_timeout: float = 120.0,
    ):
        """
        Initialize the mock enclave.

        Args:
            inference_url: LLM inference API URL
            inference_token: Bearer token for inference API
            inference_timeout: Timeout for inference requests
        """
        # Generate enclave keypair (in production, this happens inside Nitro)
        self._keypair: KeyPair = generate_x25519_keypair()

        # Inference configuration
        self._inference_url = inference_url
        self._inference_token = inference_token
        self._inference_timeout = inference_timeout

        logger.info(
            "MockEnclave initialized with public key: %s",
            self._keypair.public_key.hex()[:16] + "..."
        )

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
        async for chunk in self._call_inference_stream(user_content, history, model):
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
        storage_public_key: bytes,
        client_public_key: bytes,
        session_id: str,
        model: str,
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Process a chat message with encrypted streaming response.

        This is the main streaming API for routes. Each chunk's content
        is encrypted to the client's public key for secure transport.

        Yields:
            StreamChunk objects with encrypted content or final stored messages
        """
        try:
            print("\n" + "=" * 80)
            print("🔐 ENCRYPTED CHAT FLOW - ENCLAVE (Backend)")
            print("=" * 80)
            print(f"Session ID: {session_id}")
            print(f"Model: {model}")

            # 1. Decrypt the new user message
            print("\n📥 STEP 1: Decrypt User Message from Client")
            print("-" * 60)
            print("Enclave Private Key: [HELD SECURELY IN ENCLAVE]")
            print("Encrypted Message:")
            print(f"  ephemeral_public_key: {encrypted_message.ephemeral_public_key.hex()[:32]}...")
            print(f"  iv: {encrypted_message.iv.hex()}")
            print(f"  ciphertext: {encrypted_message.ciphertext.hex()[:32]}...")
            print(f"  auth_tag: {encrypted_message.auth_tag.hex()}")

            user_plaintext = self.decrypt_transport_message(encrypted_message)
            user_content = user_plaintext.decode("utf-8")
            print(f"✅ Decrypted User Message: {user_content}")

            # 2. Decrypt history messages
            if encrypted_history:
                print(f"\n📥 STEP 2: Decrypt History ({len(encrypted_history)} messages)")
                print("-" * 60)
            history = self._decrypt_history(encrypted_history)
            for i, msg in enumerate(history):
                print(f"  [{i}] {msg.role}: {msg.content[:50]}..." if len(msg.content) > 50 else f"  [{i}] {msg.role}: {msg.content}")

            # 3. Stream LLM inference, encrypting each chunk for transport
            print("\n🤖 STEP 3: Call LLM Inference")
            print("-" * 60)
            print(f"Model: {model}")
            print(f"Messages count: {len(history) + 1}")

            full_response = ""
            chunk_count = 0
            print("\n📤 STEP 4: Stream Encrypted Chunks to Client")
            print("-" * 60)
            print(f"Client Transport Public Key: {client_public_key.hex()[:32]}...")

            async for chunk in self._call_inference_stream(user_content, history, model):
                full_response += chunk
                chunk_count += 1
                # Encrypt this chunk for transport to client
                encrypted_chunk = self.encrypt_for_transport(
                    chunk.encode("utf-8"),
                    client_public_key,
                )
                print(f"  Chunk {chunk_count}: '{chunk}' → encrypted")
                yield StreamChunk(encrypted_content=encrypted_chunk)

            print(f"\n✅ Total chunks streamed: {chunk_count}")
            print(f"Full response: {full_response[:100]}..." if len(full_response) > 100 else f"Full response: {full_response}")

            # 4. Build final encrypted messages for storage
            print("\n💾 STEP 5: Encrypt Messages for Storage")
            print("-" * 60)
            print(f"Storage Public Key: {storage_public_key.hex()[:32]}...")

            user_bytes = user_content.encode("utf-8")
            assistant_bytes = full_response.encode("utf-8")

            stored_user = self.encrypt_for_storage(user_bytes, storage_public_key, is_assistant=False)
            stored_assistant = self.encrypt_for_storage(assistant_bytes, storage_public_key, is_assistant=True)

            print("User message encrypted for storage:")
            print(f"  ephemeral_public_key: {stored_user.ephemeral_public_key.hex()[:32]}...")
            print(f"  ciphertext length: {len(stored_user.ciphertext)} bytes")
            print("Assistant message encrypted for storage:")
            print(f"  ephemeral_public_key: {stored_assistant.ephemeral_public_key.hex()[:32]}...")
            print(f"  ciphertext length: {len(stored_assistant.ciphertext)} bytes")

            # Estimate tokens for streaming
            estimated_input_tokens = len(user_content) // 4 + sum(len(m.content) for m in history) // 4
            estimated_output_tokens = len(full_response) // 4

            print("\n📋 FINAL SUMMARY")
            print("-" * 60)
            print(f"Input tokens (estimated): {estimated_input_tokens}")
            print(f"Output tokens (estimated): {estimated_output_tokens}")
            print("=" * 80 + "\n")

            # 5. Final chunk with stored messages
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
            print(f"\n❌ ENCLAVE ERROR: {str(e)}")
            print("=" * 80 + "\n")
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
            is_assistant = (i % 2 == 1)  # 0=user, 1=assistant, 2=user, ...
            plaintext = self.decrypt_history_message(payload, is_assistant)
            history.append(DecryptedMessage(
                role="assistant" if is_assistant else "user",
                content=plaintext.decode("utf-8"),
            ))
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

    def _build_messages_for_inference(
        self,
        user_message: str,
        history: List[DecryptedMessage],
    ) -> List[Dict[str, str]]:
        """
        Build OpenAI-compatible messages array for LLM.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful AI assistant. Note: Previous assistant "
                    "messages may contain '[Response from model-name]' prefixes - "
                    "these are internal metadata annotations showing which AI model "
                    "generated that response. Do not include such prefixes in your "
                    "own responses; just respond naturally."
                ),
            }
        ]

        for msg in history:
            messages.append({
                "role": msg.role,
                "content": msg.content,
            })

        messages.append({
            "role": "user",
            "content": user_message,
        })

        return messages

    async def _call_inference(
        self,
        user_message: str,
        history: List[DecryptedMessage],
        model: str,
    ) -> Tuple[str, int, int]:
        """
        Call LLM inference (non-streaming).

        Returns:
            Tuple of (response_content, input_tokens, output_tokens)
        """
        messages = self._build_messages_for_inference(user_message, history)

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._inference_url}/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": 1024,
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "stream": False,
                },
                headers={
                    "Authorization": f"Bearer {self._inference_token}",
                    "Content-Type": "application/json",
                },
                timeout=self._inference_timeout,
            )

            if response.status_code != 200:
                logger.error(
                    "Inference error %d: %s",
                    response.status_code,
                    response.text,
                )
                raise RuntimeError(f"Inference failed: {response.status_code}")

            data = response.json()
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            return content, input_tokens, output_tokens

    async def _call_inference_stream(
        self,
        user_message: str,
        history: List[DecryptedMessage],
        model: str,
    ) -> AsyncGenerator[str, None]:
        """
        Call LLM inference with streaming.

        Yields text chunks as they arrive.
        """
        messages = self._build_messages_for_inference(user_message, history)

        async with httpx.AsyncClient() as client:
            try:
                async with client.stream(
                    "POST",
                    f"{self._inference_url}/chat/completions",
                    json={
                        "model": model,
                        "messages": messages,
                        "max_tokens": 1024,
                        "temperature": 0.7,
                        "top_p": 0.95,
                        "stream": True,
                    },
                    headers={
                        "Authorization": f"Bearer {self._inference_token}",
                        "Content-Type": "application/json",
                    },
                    timeout=self._inference_timeout,
                ) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        logger.error(
                            "Stream inference error %d: %s",
                            response.status_code,
                            error_text,
                        )
                        yield f"Error: Inference failed with status {response.status_code}"
                        return

                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str.strip() == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                if "choices" in data and len(data["choices"]) > 0:
                                    delta = data["choices"][0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content:
                                        yield content
                            except json.JSONDecodeError:
                                continue

            except httpx.ReadTimeout:
                logger.error("Inference timeout")
                yield "Error: The model is taking too long to respond."
            except Exception as e:
                logger.error("Inference error: %s", str(e))
                yield f"Error: {str(e)}"


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
            inference_url=settings.HF_API_URL,
            inference_token=settings.HUGGINGFACE_TOKEN,
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
