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
from core.enclave.embeddings import EnclaveEmbeddings
from core.enclave.fact_extraction import FactExtractor

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
    - encrypted_thinking: Encrypted thinking process chunk (during streaming)
    - stored_messages: Final encrypted messages for storage (at end)
    - extracted_memories: Memories extracted from conversation (at end)
    - extracted_facts: Facts extracted from conversation (at end, encrypted for client)
    - is_final: True for the last chunk
    - error: Error message if something went wrong
    """

    encrypted_content: Optional[EncryptedPayload] = None
    encrypted_thinking: Optional[EncryptedPayload] = None
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

    # Model for memory extraction (configurable via EXTRACTION_MODEL env var)
    # Uses smaller, faster model for efficient fact extraction
    EXTRACTION_MODEL = settings.EXTRACTION_MODEL

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

        # Embedding generator for memory extraction
        self._embeddings = EnclaveEmbeddings()

        # Pattern-based fact extractor (fast, no LLM needed)
        self._fact_extractor = FactExtractor()

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

        Zero-Trust Memory Flow (Option A):
        - Client searches memories via /memories/search/encrypted (enclave generates embedding)
        - Client receives encrypted memories (encrypted to user's key)
        - Client decrypts with private key (key NEVER leaves client)
        - Client re-encrypts memories TO enclave's public key
        - Enclave decrypts using its own private key

        This ensures user's private key never leaves the browser.

        Args:
            encrypted_message: User's message encrypted to enclave
            encrypted_history: Previous messages re-encrypted to enclave
            encrypted_memories: Relevant memories re-encrypted TO enclave by client
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

            # 2b. Decrypt memories (Option A: client searched, decrypted, re-encrypted TO enclave)
            _debug_print("\n🧠 STEP 2b: Decrypt Memories (Zero-Trust Option A)")
            _debug_print("-" * 60)
            _debug_print(f"Memories provided by client: {len(encrypted_memories)}")

            # Client already did the search and re-encrypted memories TO enclave's public key
            # We just decrypt them using enclave's private key
            memories = self._decrypt_memories(encrypted_memories)

            if memories:
                for i, mem in enumerate(memories):
                    preview = mem[:60] + "..." if len(mem) > 60 else mem
                    _debug_print(f"  [{i}] {preview}")
                _debug_print(f"✅ Decrypted {len(memories)} memories from client")
            else:
                _debug_print("  No memories provided by client")

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
            current_thinking = ""
            chunk_count = 0
            _debug_print("\n📤 STEP 4: Stream Encrypted Chunks to Client")
            _debug_print("-" * 60)
            _debug_print(f"Client Transport Public Key: {client_public_key.hex()[:32]}...")

            async for chunk, is_thinking in self._call_inference_stream(user_content, history, model, memories, facts_context):
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
    ) -> AsyncGenerator[Tuple[str, bool], None]:
        """
        Call LLM inference with streaming, handling <think> tags.

        Yields tuples of (text_chunk, is_thinking).
        """
        messages = self._build_messages_for_inference(user_message, history, memories, facts_context)

        # State for thinking tags
        is_thinking = False
        buffer = ""

        print(f"[LLM] Starting inference with model: {model}")
        print(f"[LLM] API URL: {self._inference_url}/chat/completions")
        print(f"[LLM] Messages count: {len(messages)}")
        print(f"[LLM] Token present: {bool(self._inference_token)}")
        print(f"[LLM] Token value (first 10 chars): {str(self._inference_token)[:10] if self._inference_token else 'None'}...")

        if not self._inference_token:
            print("[LLM] ERROR: No inference token configured!")
            yield ("Error: HuggingFace API token is not configured. Please set HUGGINGFACE_TOKEN.", False)
            return

        async with httpx.AsyncClient() as client:
            try:
                async with client.stream(
                    "POST",
                    f"{self._inference_url}/chat/completions",
                    json={
                        "model": model,
                        "messages": messages,
                        "max_tokens": 4096,
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
                    print(f"[LLM] Response status: {response.status_code}")
                    chunk_count = 0
                    total_chars = 0
                    if response.status_code != 200:
                        error_text = await response.aread()
                        logger.error(
                            "[LLM] Stream inference error %d: %s",
                            response.status_code,
                            error_text.decode() if isinstance(error_text, bytes) else error_text,
                        )
                        yield (f"Error: Inference failed with status {response.status_code}", False)
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
                                    
                                    if not content:
                                        continue

                                    # Track chunks and characters
                                    chunk_count += 1
                                    total_chars += len(content)

                                    # Process thinking tags
                                    buffer += content
                                    
                                    while buffer:
                                        if not is_thinking:
                                            # Check for start of think tag
                                            if "<think>" in buffer:
                                                pre_think, post_think = buffer.split("<think>", 1)
                                                if pre_think:
                                                    yield (pre_think, False)
                                                is_thinking = True
                                                buffer = post_think
                                            else:
                                                # Optimization: if no partial tag, yield everything
                                                # Check if buffer ends with partial tag like "<", "<t", etc.
                                                if any(buffer.endswith(x) for x in ["<", "<t", "<th", "<thi", "<thin", "<think"]):
                                                    # Possible split tag, keep in buffer
                                                    break
                                                else:
                                                    yield (buffer, False)
                                                    buffer = ""
                                        else:
                                            # Inside think block, check for end tag
                                            if "</think>" in buffer:
                                                think_content, post_think = buffer.split("</think>", 1)
                                                if think_content:
                                                    yield (think_content, True)
                                                is_thinking = False
                                                buffer = post_think
                                            else:
                                                # Optimization: if no partial closing tag
                                                if any(buffer.endswith(x) for x in ["<", "</", "</t", "</th", "</thi", "</thin", "</think"]):
                                                    break
                                                else:
                                                    yield (buffer, True)
                                                    buffer = ""

                            except json.JSONDecodeError:
                                logger.debug(f"[LLM] Failed to parse JSON: {data_str[:100]}")
                                continue

                    # Flush remaining buffer
                    if buffer:
                        print(f"[LLM] Flushing remaining buffer: {len(buffer)} chars")
                        yield (buffer, is_thinking)

                    print(f"[LLM] Stream completed: {chunk_count} chunks, {total_chars} total chars")

            except httpx.ReadTimeout:
                print(f"[LLM] Inference timeout after {self._inference_timeout} seconds")
                yield ("Error: The model is taking too long to respond.", False)
            except httpx.ConnectError as e:
                print(f"[LLM] Connection error: {str(e)}")
                yield (f"Error: Could not connect to inference API: {str(e)}", False)
            except Exception as e:
                print(f"[LLM] Inference error: {str(e)}")
                import traceback
                traceback.print_exc()
                yield (f"Error: {str(e)}", False)

    # -------------------------------------------------------------------------
    # Memory Extraction Methods
    # -------------------------------------------------------------------------

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
                embedding = self._embeddings.generate_embedding(text)

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

    def generate_embedding_from_encrypted(
        self,
        encrypted_query: EncryptedPayload,
    ) -> List[float]:
        """
        Decrypt a query and generate its embedding.

        This is used for encrypted memory search where:
        1. Client encrypts the query to the enclave
        2. Enclave decrypts to plaintext
        3. Enclave generates embedding from plaintext
        4. Embedding is used for similarity search (not encrypted)

        Args:
            encrypted_query: Query text encrypted to enclave's transport key

        Returns:
            384-dimensional embedding vector
        """
        # Decrypt the query
        plaintext = self.decrypt_transport_message(encrypted_query)
        query_text = plaintext.decode("utf-8")

        # Generate embedding from plaintext
        embedding = self._embeddings.generate_embedding(query_text)

        logger.debug(f"Generated embedding for encrypted query (length: {len(embedding)})")
        return embedding

    async def search_and_decrypt_memories(
        self,
        query_text: str,
        user_id: str,
        org_id: Optional[str],
        storage_private_key: bytes,
        limit: int = 5,
    ) -> List[str]:
        """
        Search memories by semantic similarity and decrypt results.

        This is the zero-trust memory search that runs entirely inside the enclave:
        1. Generate embedding from plaintext query
        2. Search memories via MemoryService
        3. Decrypt memory content using storage private key
        4. Return plaintext memories for LLM context injection

        Args:
            query_text: Plaintext query (already decrypted user message)
            user_id: User ID for memory search
            org_id: Optional org ID for org context
            storage_private_key: User's or org's private key for decrypting memories
            limit: Maximum number of memories to return

        Returns:
            List of decrypted memory texts
        """
        from core.services.memory_service import MemoryService

        try:
            # 1. Generate embedding from plaintext
            embedding = self._embeddings.generate_embedding(query_text)
            _debug_print(f"Generated embedding for memory search (dim: {len(embedding)})")

            # 2. Search memories via MemoryService
            service = MemoryService()
            results = await service.search_memories(
                query_text=query_text,
                query_embedding=embedding,
                user_id=user_id,
                org_id=org_id,
                limit=limit,
                include_personal_in_org=False,
            )

            if not results:
                _debug_print("No memories found")
                return []

            _debug_print(f"Found {len(results)} memories, decrypting...")

            # 3. Decrypt each memory
            decrypted_memories = []
            for r in results:
                try:
                    # Get encrypted content and metadata
                    content = r.get("content", "")
                    meta = r.get("meta") or r.get("metadata") or {}
                    if isinstance(meta, str):
                        import json

                        meta = json.loads(meta)

                    # Build encrypted payload
                    encrypted_payload = EncryptedPayload(
                        ephemeral_public_key=bytes.fromhex(meta.get("ephemeral_public_key", "")),
                        iv=bytes.fromhex(meta.get("iv", "")),
                        ciphertext=bytes.fromhex(content),
                        auth_tag=bytes.fromhex(meta.get("auth_tag", "")),
                        hkdf_salt=bytes.fromhex(meta.get("hkdf_salt", "")),
                    )

                    # Decrypt using MEMORY_STORAGE context
                    from . import EncryptionContext

                    plaintext = decrypt_with_private_key(
                        storage_private_key,
                        encrypted_payload,
                        EncryptionContext.MEMORY_STORAGE.value,
                    )
                    decrypted_memories.append(plaintext.decode("utf-8"))

                except Exception as e:
                    logger.warning(f"Failed to decrypt memory {r.get('id')}: {e}")
                    continue

            _debug_print(f"Successfully decrypted {len(decrypted_memories)} memories")
            return decrypted_memories

        except Exception as e:
            logger.warning(f"Memory search failed (non-fatal): {e}")
            return []

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

        Uses pattern-based extraction (FactExtractor) for speed instead of LLM.

        Args:
            user_message: Plaintext user message
            assistant_response: Plaintext assistant response
            client_public_key: Client's transport key for encrypting facts

        Returns:
            List of ExtractedFact objects (encrypted for client)
        """
        import uuid

        # Use pattern-based extraction (fast, no LLM needed)
        extracted = self._fact_extractor.extract(
            user_message=user_message,
            assistant_response=assistant_response,
        )

        if not extracted:
            logger.debug("No facts extracted from conversation")
            return []

        logger.info(f"Extracted {len(extracted)} facts via pattern matching")

        facts = []
        for item in extracted:
            try:
                # Skip low confidence facts
                if item.confidence < 0.5:
                    continue

                # Generate fact ID
                fact_id = str(uuid.uuid4())

                # Build fact data from ExtractedFact dataclass
                fact_data = {
                    "id": fact_id,
                    "subject": item.subject,
                    "predicate": item.predicate,
                    "object": item.object,
                    "confidence": item.confidence,
                    "type": item.type,
                    "source": item.source,
                    "entities": item.entities,
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

                logger.debug(
                    f"Extracted fact: {item.subject} {item.predicate} {item.object} (confidence: {item.confidence})"
                )

            except Exception as e:
                logger.warning(f"Failed to process extracted fact: {e}")
                continue

        logger.info(f"Successfully processed {len(facts)} facts")
        return facts


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
