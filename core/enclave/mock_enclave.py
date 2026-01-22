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

from core.config import settings
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
class StreamChunk:
    """
    A chunk of streaming response from the enclave.

    Used for SSE streaming where each chunk may contain:
    - encrypted_content: Encrypted chunk for client (during streaming)
    - stored_messages: Final encrypted messages for storage (at end)
    - extracted_memories: Memories extracted from conversation (at end)
    - extracted_facts: Facts extracted from conversation (at end, encrypted for client)
    - is_final: True for the last chunk
    - error: Error message if something went wrong
    """

    encrypted_content: Optional[EncryptedPayload] = None
    stored_user_message: Optional[EncryptedPayload] = None
    stored_assistant_message: Optional[EncryptedPayload] = None
    extracted_memories: Optional[List["ExtractedMemory"]] = None  # Forward reference
    extracted_facts: Optional[List["ExtractedFact"]] = None  # Facts for client-side storage
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


@dataclass
class ExtractedMemory:
    """
    A memory extracted from conversation and ready for storage.

    This is returned by the enclave's extract_memories method.
    The content is already encrypted to the storage key.
    """

    encrypted_content: EncryptedPayload
    embedding: List[float]
    sector: str  # episodic, semantic, procedural, emotional, reflective
    tags: List[str]
    metadata: Dict  # Contains iv, auth_tag for decryption
    salience: float = 0.5  # Importance score 0.0-1.0, computed from plaintext


@dataclass
class ExtractedFact:
    """
    A temporal fact extracted from conversation.

    Facts are structured knowledge in subject-predicate-object form.
    The encrypted_payload contains the full fact data, encrypted to the
    client's transport key so they can decrypt and store locally.

    Attributes:
        encrypted_payload: Encrypted JSON containing {subject, predicate, object, confidence, type, entities}
        fact_id: Unique identifier for the fact
    """

    encrypted_payload: EncryptedPayload
    fact_id: str


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

    # Model for memory extraction (uses Gemma 2 2B by default)
    EXTRACTION_MODEL = "google/gemma-2-2b-it"

    # Embedding model for generating embeddings from plaintext
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

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

        # Embedding model (lazy loaded)
        self._embedding_model = None

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
        encrypted_memories: List[EncryptedPayload],
        facts_context: Optional[str],
        storage_public_key: bytes,
        client_public_key: bytes,
        session_id: str,
        model: str,
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Process a chat message with encrypted streaming response.

        This is the main streaming API for routes. Each chunk's content
        is encrypted to the client's public key for secure transport.

        Args:
            encrypted_message: User's message encrypted to enclave
            encrypted_history: Previous messages re-encrypted to enclave
            encrypted_memories: Relevant memories re-encrypted to enclave for context
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

            # 2b. Decrypt memories for context injection
            memories = []
            if encrypted_memories:
                _debug_print(f"\n🧠 STEP 2b: Decrypt Memories ({len(encrypted_memories)} memories)")
                _debug_print("-" * 60)
                memories = self._decrypt_memories(encrypted_memories)
                for i, mem in enumerate(memories):
                    preview = mem[:60] + "..." if len(mem) > 60 else mem
                    _debug_print(f"  [{i}] {preview}")
                _debug_print(f"✅ Decrypted {len(memories)} memories for context injection")

            # 2c. Log facts context if provided
            if facts_context:
                _debug_print("\n📋 STEP 2c: Session Facts Context")
                _debug_print("-" * 60)
                preview = facts_context[:200] + "..." if len(facts_context) > 200 else facts_context
                _debug_print(f"Facts context: {preview}")

            # 3. Stream LLM inference, encrypting each chunk for transport
            _debug_print("\n🤖 STEP 3: Call LLM Inference")
            _debug_print("-" * 60)
            _debug_print(f"Model: {model}")
            _debug_print(f"Messages count: {len(history) + 1}")
            _debug_print(f"Memories injected: {len(memories)}")
            _debug_print(f"Facts context: {'Yes' if facts_context else 'No'}")

            full_response = ""
            chunk_count = 0
            _debug_print("\n📤 STEP 4: Stream Encrypted Chunks to Client")
            _debug_print("-" * 60)
            _debug_print(f"Client Transport Public Key: {client_public_key.hex()[:32]}...")

            async for chunk in self._call_inference_stream(user_content, history, model, memories, facts_context):
                full_response += chunk
                chunk_count += 1
                # Encrypt this chunk for transport to client
                encrypted_chunk = self.encrypt_for_transport(
                    chunk.encode("utf-8"),
                    client_public_key,
                )
                _debug_print(f"  Chunk {chunk_count}: '{chunk}' → encrypted")
                yield StreamChunk(encrypted_content=encrypted_chunk)

            _debug_print(f"\n✅ Total chunks streamed: {chunk_count}")
            _debug_print(
                f"Full response: {full_response[:100]}..."
                if len(full_response) > 100
                else f"Full response: {full_response}"
            )

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

            # 5. Extract memories from conversation (async - runs in background)
            _debug_print("\n🧠 STEP 6: Extract Memories from Conversation")
            _debug_print("-" * 60)
            extracted_memories = []
            try:
                extracted_memories = await self.extract_memories(
                    user_message=user_content,
                    assistant_response=full_response,
                    storage_public_key=storage_public_key,
                )
                _debug_print(f"Extracted {len(extracted_memories)} memories")
                for mem in extracted_memories:
                    _debug_print(f"  - [{mem.sector}]")
            except Exception as e:
                logger.warning(f"Memory extraction failed (non-fatal): {e}")
                _debug_print(f"  Memory extraction failed (non-fatal): {e}")

            # 6. Extract facts from conversation (encrypted to client for local storage)
            _debug_print("\n📝 STEP 7: Extract Facts from Conversation")
            _debug_print("-" * 60)
            extracted_facts = []
            try:
                extracted_facts = await self.extract_facts(
                    user_message=user_content,
                    assistant_response=full_response,
                    client_public_key=client_public_key,
                )
                _debug_print(f"Extracted {len(extracted_facts)} facts")
                for fact in extracted_facts:
                    _debug_print(f"  - fact_id: {fact.fact_id[:8]}...")
            except Exception as e:
                logger.warning(f"Fact extraction failed (non-fatal): {e}")
                _debug_print(f"  Fact extraction failed (non-fatal): {e}")

            _debug_print("\n📋 FINAL SUMMARY")
            _debug_print("-" * 60)
            _debug_print(f"Session facts injected: {'Yes' if facts_context else 'No'}")
            _debug_print(f"Long-term memories injected: {len(memories)}")
            _debug_print(f"Input tokens (estimated): {estimated_input_tokens}")
            _debug_print(f"Output tokens (estimated): {estimated_output_tokens}")
            _debug_print(f"New memories extracted: {len(extracted_memories)}")
            _debug_print(f"New facts extracted: {len(extracted_facts)}")
            _debug_print("=" * 80 + "\n")

            # 7. Final chunk with stored messages, memories, and facts
            yield StreamChunk(
                stored_user_message=stored_user,
                stored_assistant_message=stored_assistant,
                extracted_memories=extracted_memories if extracted_memories else None,
                extracted_facts=extracted_facts if extracted_facts else None,
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

    def _decrypt_memories(
        self,
        encrypted_memories: List[EncryptedPayload],
    ) -> List[str]:
        """
        Decrypt memories for context injection.

        Memories are re-encrypted to the enclave by the client,
        so we use the CLIENT_TO_ENCLAVE context for decryption.

        Returns:
            List of decrypted memory texts
        """
        from . import EncryptionContext

        memories = []
        for payload in encrypted_memories:
            try:
                plaintext = decrypt_with_private_key(
                    self._keypair.private_key,
                    payload,
                    EncryptionContext.CLIENT_TO_ENCLAVE.value,
                )
                memories.append(plaintext.decode("utf-8"))
            except Exception as e:
                logger.warning(f"Failed to decrypt memory (skipping): {e}")
                continue
        return memories

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
        memories: Optional[List[str]] = None,
        facts_context: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """
        Build OpenAI-compatible messages array for LLM.

        Args:
            user_message: The current user message
            history: Previous conversation messages
            memories: Optional list of relevant memory texts to inject (from long-term storage)
            facts_context: Optional client-side formatted facts context string (session facts)

        Returns:
            List of messages in OpenAI format
        """
        # Build system prompt with optional memory and facts context
        system_content = (
            "You are a helpful AI assistant. Note: Previous assistant "
            "messages may contain '[Response from model-name]' prefixes - "
            "these are internal metadata annotations showing which AI model "
            "generated that response. Do not include such prefixes in your "
            "own responses; just respond naturally."
        )

        # Inject facts context (client-side session facts) if available
        if facts_context:
            system_content += f"\n\n{facts_context}"

        # Inject memories (long-term storage) into system prompt if available
        if memories:
            memory_context = "\n\n## Long-Term Memory\n"
            memory_context += "The following facts are from long-term memory:\n"
            for i, memory in enumerate(memories, 1):
                memory_context += f"- {memory}\n"
            memory_context += "\nUse this context naturally in your response when relevant, but don't explicitly mention that you're using memories."
            system_content += memory_context

        messages = [
            {
                "role": "system",
                "content": system_content,
            }
        ]

        for msg in history:
            messages.append(
                {
                    "role": msg.role,
                    "content": msg.content,
                }
            )

        messages.append(
            {
                "role": "user",
                "content": user_message,
            }
        )

        return messages

    async def _call_inference(
        self,
        user_message: str,
        history: List[DecryptedMessage],
        model: str,
        memories: Optional[List[str]] = None,
        facts_context: Optional[str] = None,
    ) -> Tuple[str, int, int]:
        """
        Call LLM inference (non-streaming).

        Returns:
            Tuple of (response_content, input_tokens, output_tokens)
        """
        messages = self._build_messages_for_inference(user_message, history, memories, facts_context)

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
        memories: Optional[List[str]] = None,
        facts_context: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Call LLM inference with streaming.

        Yields text chunks as they arrive.
        """
        messages = self._build_messages_for_inference(user_message, history, memories, facts_context)

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

    # -------------------------------------------------------------------------
    # Memory Extraction Methods
    # -------------------------------------------------------------------------

    def _get_embedding_model(self):
        """
        Lazy-load the sentence-transformers embedding model.

        Only loads on first use to avoid startup delay.
        """
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._embedding_model = SentenceTransformer(self.EMBEDDING_MODEL)
                logger.info(f"Loaded embedding model: {self.EMBEDDING_MODEL}")
            except ImportError:
                logger.error("sentence-transformers not installed")
                raise RuntimeError(
                    "sentence-transformers package required for memory extraction. "
                    "Install with: pip install sentence-transformers"
                )
        return self._embedding_model

    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector from plaintext.

        Args:
            text: Plaintext to embed

        Returns:
            List of floats (384-dimensional for all-MiniLM-L6-v2)
        """
        model = self._get_embedding_model()
        embedding = model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def _build_extraction_prompt(
        self,
        user_message: str,
        assistant_response: str,
    ) -> str:
        """
        Build the prompt for memory extraction LLM.
        """
        return f"""Extract memorable facts from this conversation as JSON.
Only extract facts worth remembering long-term (preferences, facts about the user, how-to knowledge, etc.).
If nothing is worth remembering, return an empty array [].

Categorize each fact as one of:
- semantic: Facts and knowledge (e.g., "User's favorite color is blue", "User works at Google")
- episodic: Events and experiences (e.g., "User went to Paris last week")
- procedural: How-to and preferences (e.g., "User prefers TypeScript over JavaScript")
- emotional: Feelings and sentiments (e.g., "User is excited about their new project")
- reflective: Meta-observations (e.g., "User tends to ask detailed follow-up questions")

Rate each fact's importance (salience) from 0.0 to 1.0:
- 0.9-1.0: Critical personal info (name, job, location, core preferences)
- 0.7-0.8: Important preferences or facts that affect future interactions
- 0.5-0.6: Useful context that may be relevant later
- 0.3-0.4: Minor observations, less likely to be needed

Conversation:
User: {user_message}
Assistant: {assistant_response}

Output ONLY a valid JSON array with this exact format (no other text):
[{{"text": "the memorable fact", "sector": "semantic|episodic|procedural|emotional|reflective", "salience": 0.7}}]

IMPORTANT: Do NOT include tags or any other metadata that could reveal content details.
The sector and salience provide sufficient categorization.

If nothing memorable, output: []"""

    async def _call_extraction_llm(self, prompt: str) -> List[Dict]:
        """
        Call the extraction LLM to parse memories from conversation.

        Returns:
            List of extracted memories with text, sector, and tags
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self._inference_url}/chat/completions",
                    json={
                        "model": self.EXTRACTION_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 1024,
                        "temperature": 0.3,  # Lower temp for structured output
                    },
                    headers={
                        "Authorization": f"Bearer {self._inference_token}",
                        "Content-Type": "application/json",
                    },
                    timeout=60.0,
                )

                if response.status_code != 200:
                    logger.error(
                        "Extraction LLM error %d: %s",
                        response.status_code,
                        response.text,
                    )
                    return []

                data = response.json()
                content = data["choices"][0]["message"]["content"].strip()

                # Parse JSON from response
                # Handle potential markdown code blocks
                if content.startswith("```"):
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]
                    content = content.strip()

                extracted = json.loads(content)

                if not isinstance(extracted, list):
                    logger.warning("Extraction LLM returned non-list: %s", content)
                    return []

                return extracted

        except json.JSONDecodeError as e:
            logger.warning("Failed to parse extraction JSON: %s", str(e))
            return []
        except Exception as e:
            logger.error("Extraction LLM error: %s", str(e))
            return []

    def encrypt_for_memory_storage(
        self,
        plaintext: bytes,
        storage_public_key: bytes,
    ) -> EncryptedPayload:
        """
        Encrypt memory content for storage.

        Uses MEMORY_STORAGE context for domain separation.
        """
        from . import EncryptionContext

        return encrypt_to_public_key(
            storage_public_key,
            plaintext,
            EncryptionContext.MEMORY_STORAGE.value,
        )

    async def extract_memories(
        self,
        user_message: str,
        assistant_response: str,
        storage_public_key: bytes,
    ) -> List[ExtractedMemory]:
        """
        Extract memories from a conversation turn.

        This method:
        1. Calls the extraction LLM to identify memorable facts
        2. Generates embeddings from plaintext (for vector search)
        3. Encrypts the memory text to the storage key
        4. Returns ExtractedMemory objects ready for storage

        Args:
            user_message: Plaintext user message
            assistant_response: Plaintext assistant response
            storage_public_key: Public key for encrypting memory content
                               (user's key for personal, org's key for org context)

        Returns:
            List of ExtractedMemory objects
        """
        # 1. Call extraction LLM
        prompt = self._build_extraction_prompt(user_message, assistant_response)
        extracted = await self._call_extraction_llm(prompt)

        if not extracted:
            logger.debug("No memories extracted from conversation")
            return []

        logger.info(f"Extracted {len(extracted)} potential memories")

        memories = []
        for item in extracted:
            try:
                text = item.get("text", "")
                sector = item.get("sector", "semantic")
                salience = item.get("salience", 0.5)
                # Tags removed for security - they could reveal content details
                # Sectors provide sufficient categorization

                if not text:
                    continue

                # Validate sector
                valid_sectors = ["episodic", "semantic", "procedural", "emotional", "reflective"]
                if sector not in valid_sectors:
                    sector = "semantic"

                # Validate salience (0.0-1.0)
                try:
                    salience = float(salience)
                    salience = max(0.0, min(1.0, salience))
                except (ValueError, TypeError):
                    salience = 0.5

                # 2. Generate embedding from plaintext
                embedding = self._generate_embedding(text)

                # 3. Encrypt the memory text
                encrypted_content = self.encrypt_for_memory_storage(
                    text.encode("utf-8"),
                    storage_public_key,
                )

                # 4. Build ExtractedMemory object
                # Note: tags intentionally empty for security (content details stay encrypted)
                memories.append(
                    ExtractedMemory(
                        encrypted_content=encrypted_content,
                        embedding=embedding,
                        sector=sector,
                        tags=[],  # Empty - content categorization via sector only
                        metadata={
                            "iv": encrypted_content.iv.hex(),
                            "auth_tag": encrypted_content.auth_tag.hex(),
                            "ephemeral_public_key": encrypted_content.ephemeral_public_key.hex(),
                            "hkdf_salt": encrypted_content.hkdf_salt.hex(),
                        },
                        salience=salience,
                    )
                )

                logger.debug(f"Extracted memory: {text[:50]}... (sector: {sector}, salience: {salience})")

            except Exception as e:
                logger.warning(f"Failed to process extracted memory: {e}")
                continue

        logger.info(f"Successfully processed {len(memories)} memories")
        return memories

    async def extract_facts(
        self,
        user_message: str,
        assistant_response: str,
        client_public_key: bytes,
    ) -> List[ExtractedFact]:
        """
        Extract temporal facts from a conversation turn.

        Facts are structured knowledge (subject-predicate-object) that can be
        queried later. They are encrypted to the client's transport key so
        the client can decrypt and store them in local IndexedDB.

        Args:
            user_message: Plaintext user message
            assistant_response: Plaintext assistant response
            client_public_key: Client's transport key for encrypting facts

        Returns:
            List of ExtractedFact objects (encrypted for client)
        """
        import uuid

        # Build fact extraction prompt
        prompt = self._build_fact_extraction_prompt(user_message, assistant_response)

        # Call extraction LLM with error handling
        try:
            extracted = await self._call_fact_extraction_llm(prompt)
        except Exception as e:
            logger.warning(f"Fact extraction LLM call failed: {e}")
            return []

        if not extracted:
            logger.debug("No facts extracted from conversation")
            return []

        logger.info(f"Extracted {len(extracted)} potential facts")

        facts = []
        valid_predicates = {
            "prefers",
            "works_at",
            "located_in",
            "interested_in",
            "has_skill",
            "dislikes",
            "plans_to",
            "uses",
            "knows",
            "mentioned",
        }
        predicate_to_type = {
            "prefers": "preference",
            "works_at": "identity",
            "located_in": "identity",
            "interested_in": "preference",
            "has_skill": "identity",
            "dislikes": "preference",
            "plans_to": "plan",
            "uses": "preference",
            "knows": "observation",
            "mentioned": "observation",
        }

        for item in extracted:
            try:
                subject = item.get("subject", "").lower().strip()
                predicate = item.get("predicate", "").lower().strip()
                obj = item.get("object", "").strip()
                confidence = item.get("confidence", 0.7)

                if not subject or not predicate or not obj:
                    continue

                if predicate not in valid_predicates:
                    continue

                # Validate confidence
                try:
                    confidence = float(confidence)
                    confidence = max(0.0, min(1.0, confidence))
                except (ValueError, TypeError):
                    confidence = 0.7

                # Skip low confidence
                if confidence < 0.5:
                    continue

                # Generate fact ID
                fact_id = str(uuid.uuid4())

                # Build fact data
                fact_data = {
                    "id": fact_id,
                    "subject": subject,
                    "predicate": predicate,
                    "object": obj,
                    "confidence": confidence,
                    "type": predicate_to_type.get(predicate, "observation"),
                    "source": "system",  # 'system' matches frontend FactSource type
                    "entities": self._extract_entities(obj, predicate),
                }

                # Encrypt fact data to client's transport key
                fact_json = json.dumps(fact_data).encode("utf-8")
                encrypted_fact = encrypt_to_public_key(
                    recipient_public_key=client_public_key,
                    plaintext=fact_json,
                    context="fact-extraction",
                )

                facts.append(
                    ExtractedFact(
                        encrypted_payload=encrypted_fact,
                        fact_id=fact_id,
                    )
                )

                logger.debug(f"Extracted fact: {subject} {predicate} {obj} (confidence: {confidence})")

            except Exception as e:
                logger.warning(f"Failed to process extracted fact: {e}")
                continue

        logger.info(f"Successfully processed {len(facts)} facts")
        return facts

    def _build_fact_extraction_prompt(self, user_message: str, assistant_response: str) -> str:
        """Build prompt for fact extraction LLM."""
        return f"""Extract facts from this conversation as JSON. Only extract facts worth remembering long-term.

Valid predicates: prefers, works_at, located_in, interested_in, has_skill, dislikes, plans_to, uses, knows, mentioned

Conversation:
User: {user_message}
Assistant: {assistant_response}

Output ONLY a valid JSON array with this format (no other text):
[{{"subject": "user", "predicate": "prefers", "object": "TypeScript", "confidence": 0.9}}]

If no facts worth extracting, output: []"""

    async def _call_fact_extraction_llm(self, prompt: str) -> List[Dict]:
        """Call extraction LLM for facts."""
        # Use the same LLM as memory extraction
        try:
            messages = [{"role": "user", "content": prompt}]
            # Use a smaller, faster model for extraction
            model = "mistralai/Mistral-7B-Instruct-v0.3"

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.inference_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.inference_token}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": messages,
                        "max_tokens": 256,
                        "temperature": 0.3,
                    },
                )
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]

                # Parse JSON from response
                import re

                json_match = re.search(r"\[[\s\S]*?\]", content)
                if json_match:
                    return json.loads(json_match.group())
                return []

        except Exception as e:
            logger.warning(f"Fact extraction LLM call failed: {e}")
            return []

    def _extract_entities(self, obj: str, predicate: str) -> List[str]:
        """Extract entity tags from the object and predicate."""
        import re

        entities = []

        # Normalize object
        normalized = re.sub(r"[^a-z0-9\s]", "", obj.lower()).strip()
        if len(normalized) > 2:
            entities.append(normalized)

        # Add predicate as category
        entities.append(predicate)

        # Split multi-word objects
        words = [w for w in normalized.split() if len(w) > 3]
        for word in words:
            if word not in entities:
                entities.append(word)

        return entities


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
