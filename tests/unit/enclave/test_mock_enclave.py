"""
Tests for the mock enclave implementation.

These tests verify:
- Enclave key generation and management
- Transport encryption/decryption
- Storage encryption with correct contexts
- Message processing flow (without actual LLM calls)
"""

import pytest
from unittest.mock import AsyncMock, patch

from core.crypto import (
    EncryptedPayload,
    generate_x25519_keypair,
    encrypt_to_public_key,
    decrypt_with_private_key,
)
from core.enclave import EncryptionContext, MockEnclave
from core.enclave.mock_enclave import (
    DecryptedMessage,
    ProcessedMessage,
    EnclaveInfo,
    get_enclave,
    reset_enclave,
)


# =============================================================================
# EncryptionContext Tests
# =============================================================================


class TestEncryptionContext:
    """Tests for EncryptionContext enum."""

    def test_context_values(self):
        """EncryptionContext enum has expected string values."""
        assert EncryptionContext.CLIENT_TO_ENCLAVE.value == "client-to-enclave-transport"
        assert EncryptionContext.ENCLAVE_TO_CLIENT.value == "enclave-to-client-transport"
        assert EncryptionContext.USER_MESSAGE_STORAGE.value == "user-message-storage"
        assert EncryptionContext.ASSISTANT_MESSAGE_STORAGE.value == "assistant-message-storage"
        assert EncryptionContext.ORG_KEY_DISTRIBUTION.value == "org-key-distribution"
        assert EncryptionContext.RECOVERY_KEY_ENCRYPTION.value == "recovery-key-encryption"

    def test_context_is_string_enum(self):
        """EncryptionContext values can be used as strings."""
        assert isinstance(EncryptionContext.CLIENT_TO_ENCLAVE, str)
        assert EncryptionContext.CLIENT_TO_ENCLAVE == "client-to-enclave-transport"


# =============================================================================
# EnclaveInfo Tests
# =============================================================================


class TestEnclaveInfo:
    """Tests for EnclaveInfo data structure."""

    def test_enclave_info_creation(self):
        """EnclaveInfo can be created with public key."""
        keypair = generate_x25519_keypair()
        info = EnclaveInfo(enclave_public_key=keypair.public_key)

        assert info.enclave_public_key == keypair.public_key
        assert info.attestation_document is None

    def test_enclave_info_with_attestation(self):
        """EnclaveInfo can include attestation document."""
        keypair = generate_x25519_keypair()
        attestation = b"mock_attestation_doc"
        info = EnclaveInfo(
            enclave_public_key=keypair.public_key,
            attestation_document=attestation,
        )

        assert info.attestation_document == attestation

    def test_to_hex_dict(self):
        """to_hex_dict returns hex-encoded values."""
        keypair = generate_x25519_keypair()
        info = EnclaveInfo(enclave_public_key=keypair.public_key)

        result = info.to_hex_dict()

        assert result["enclave_public_key"] == keypair.public_key.hex()
        assert result["attestation_document"] is None

    def test_to_hex_dict_with_attestation(self):
        """to_hex_dict includes hex-encoded attestation."""
        keypair = generate_x25519_keypair()
        attestation = b"\x01\x02\x03"
        info = EnclaveInfo(
            enclave_public_key=keypair.public_key,
            attestation_document=attestation,
        )

        result = info.to_hex_dict()

        assert result["attestation_document"] == "010203"


# =============================================================================
# DecryptedMessage Tests
# =============================================================================


class TestDecryptedMessage:
    """Tests for DecryptedMessage data structure."""

    def test_decrypted_message_creation(self):
        """DecryptedMessage can be created with role and content."""
        msg = DecryptedMessage(role="user", content="Hello")

        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_decrypted_message_immutable(self):
        """DecryptedMessage is frozen (immutable)."""
        msg = DecryptedMessage(role="user", content="Hello")

        with pytest.raises(AttributeError):
            msg.content = "Changed"


# =============================================================================
# MockEnclave Initialization Tests
# =============================================================================


class TestMockEnclaveInit:
    """Tests for MockEnclave initialization."""

    def test_generates_keypair_on_init(self):
        """MockEnclave generates keypair on initialization."""
        enclave = MockEnclave(inference_token="test_token")
        info = enclave.get_info()

        assert len(info.enclave_public_key) == 32

    def test_keypair_unique_per_instance(self):
        """Each MockEnclave instance has unique keypair."""
        enclave1 = MockEnclave(inference_token="test")
        enclave2 = MockEnclave(inference_token="test")

        assert enclave1.get_info().enclave_public_key != enclave2.get_info().enclave_public_key

    def test_no_attestation_for_mock(self):
        """MockEnclave returns no attestation document."""
        enclave = MockEnclave(inference_token="test")
        info = enclave.get_info()

        assert info.attestation_document is None


# =============================================================================
# Transport Encryption Tests
# =============================================================================


class TestTransportEncryption:
    """Tests for transport encryption/decryption."""

    @pytest.fixture
    def enclave(self):
        """Create a MockEnclave instance for testing."""
        return MockEnclave(inference_token="test_token")

    @pytest.fixture
    def user_keypair(self):
        """Create a user keypair for testing."""
        return generate_x25519_keypair()

    def test_decrypt_transport_message(self, enclave):
        """MockEnclave can decrypt message sent to it."""
        enclave_public_key = enclave.get_info().enclave_public_key
        plaintext = b"Hello from client"

        # Client encrypts to enclave
        payload = encrypt_to_public_key(
            enclave_public_key,
            plaintext,
            EncryptionContext.CLIENT_TO_ENCLAVE.value,
        )

        # Enclave decrypts
        decrypted = enclave.decrypt_transport_message(payload)

        assert decrypted == plaintext

    def test_decrypt_transport_rejects_wrong_context(self, enclave):
        """Transport decryption fails with wrong context."""
        from cryptography.exceptions import InvalidTag

        enclave_public_key = enclave.get_info().enclave_public_key
        plaintext = b"Wrong context test"

        # Encrypt with wrong context
        payload = encrypt_to_public_key(
            enclave_public_key,
            plaintext,
            "wrong-context",
        )

        # Decryption should fail
        with pytest.raises(InvalidTag):
            enclave.decrypt_transport_message(payload)

    def test_encrypt_for_transport(self, enclave, user_keypair):
        """MockEnclave can encrypt response for transport to client."""
        response = b"Hello from enclave"

        # Enclave encrypts to user
        payload = enclave.encrypt_for_transport(
            response,
            user_keypair.public_key,
        )

        # User can decrypt
        decrypted = decrypt_with_private_key(
            user_keypair.private_key,
            payload,
            EncryptionContext.ENCLAVE_TO_CLIENT.value,
        )

        assert decrypted == response


# =============================================================================
# Storage Encryption Tests
# =============================================================================


class TestStorageEncryption:
    """Tests for storage encryption with correct contexts."""

    @pytest.fixture
    def enclave(self):
        return MockEnclave(inference_token="test_token")

    @pytest.fixture
    def storage_keypair(self):
        """Storage key (user's or org's key)."""
        return generate_x25519_keypair()

    def test_encrypt_user_message_for_storage(self, enclave, storage_keypair):
        """User messages use USER_MESSAGE_STORAGE context."""
        message = b"User's question"

        payload = enclave.encrypt_for_storage(
            message,
            storage_keypair.public_key,
            is_assistant=False,
        )

        # Should decrypt with USER_MESSAGE_STORAGE context
        decrypted = decrypt_with_private_key(
            storage_keypair.private_key,
            payload,
            EncryptionContext.USER_MESSAGE_STORAGE.value,
        )

        assert decrypted == message

    def test_encrypt_assistant_message_for_storage(self, enclave, storage_keypair):
        """Assistant messages use ASSISTANT_MESSAGE_STORAGE context."""
        message = b"Assistant's response"

        payload = enclave.encrypt_for_storage(
            message,
            storage_keypair.public_key,
            is_assistant=True,
        )

        # Should decrypt with ASSISTANT_MESSAGE_STORAGE context
        decrypted = decrypt_with_private_key(
            storage_keypair.private_key,
            payload,
            EncryptionContext.ASSISTANT_MESSAGE_STORAGE.value,
        )

        assert decrypted == message

    def test_user_message_wrong_context_fails(self, enclave, storage_keypair):
        """User message decryption fails with assistant context."""
        from cryptography.exceptions import InvalidTag

        message = b"User message"

        payload = enclave.encrypt_for_storage(
            message,
            storage_keypair.public_key,
            is_assistant=False,
        )

        # Try to decrypt with wrong context
        with pytest.raises(InvalidTag):
            decrypt_with_private_key(
                storage_keypair.private_key,
                payload,
                EncryptionContext.ASSISTANT_MESSAGE_STORAGE.value,
            )

    def test_assistant_message_wrong_context_fails(self, enclave, storage_keypair):
        """Assistant message decryption fails with user context."""
        from cryptography.exceptions import InvalidTag

        message = b"Assistant message"

        payload = enclave.encrypt_for_storage(
            message,
            storage_keypair.public_key,
            is_assistant=True,
        )

        # Try to decrypt with wrong context
        with pytest.raises(InvalidTag):
            decrypt_with_private_key(
                storage_keypair.private_key,
                payload,
                EncryptionContext.USER_MESSAGE_STORAGE.value,
            )


# =============================================================================
# History Decryption Tests
# =============================================================================


class TestHistoryDecryption:
    """Tests for decrypting conversation history."""

    @pytest.fixture
    def enclave(self):
        return MockEnclave(inference_token="test_token")

    def test_decrypt_history_message(self, enclave):
        """Can decrypt history message re-encrypted to enclave."""
        enclave_public_key = enclave.get_info().enclave_public_key

        # Simulate client re-encrypting a history message to enclave
        history_msg = b"Previous user message"
        payload = encrypt_to_public_key(
            enclave_public_key,
            history_msg,
            EncryptionContext.CLIENT_TO_ENCLAVE.value,
        )

        decrypted = enclave.decrypt_history_message(payload, is_assistant=False)

        assert decrypted == history_msg


# =============================================================================
# ProcessedMessage Tests
# =============================================================================


class TestProcessedMessage:
    """Tests for ProcessedMessage data structure."""

    def test_processed_message_fields(self):
        """ProcessedMessage contains all required encrypted payloads."""
        # Create mock payloads
        user_payload = EncryptedPayload(
            ephemeral_public_key=b"\x00" * 32,
            iv=b"\x01" * 16,
            ciphertext=b"user",
            auth_tag=b"\x02" * 16,
            hkdf_salt=b"\x03" * 32,
        )
        assistant_payload = EncryptedPayload(
            ephemeral_public_key=b"\x00" * 32,
            iv=b"\x01" * 16,
            ciphertext=b"assistant",
            auth_tag=b"\x02" * 16,
            hkdf_salt=b"\x03" * 32,
        )
        transport_payload = EncryptedPayload(
            ephemeral_public_key=b"\x00" * 32,
            iv=b"\x01" * 16,
            ciphertext=b"transport",
            auth_tag=b"\x02" * 16,
            hkdf_salt=b"\x03" * 32,
        )

        processed = ProcessedMessage(
            stored_user_message=user_payload,
            stored_assistant_message=assistant_payload,
            transport_response=transport_payload,
        )

        assert processed.stored_user_message.ciphertext == b"user"
        assert processed.stored_assistant_message.ciphertext == b"assistant"
        assert processed.transport_response.ciphertext == b"transport"


# =============================================================================
# Full Message Processing Tests (with mocked LLM)
# =============================================================================


class TestMessageProcessing:
    """Tests for complete message processing with mocked LLM."""

    @pytest.fixture
    def enclave(self):
        return MockEnclave(
            inference_url="http://mock.llm",
            inference_token="test_token",
        )

    @pytest.fixture
    def user_keypair(self):
        return generate_x25519_keypair()

    @pytest.fixture
    def storage_keypair(self):
        """Storage key (same as user for personal messages)."""
        return generate_x25519_keypair()

    @pytest.mark.asyncio
    async def test_process_message_encrypts_all_outputs(self, enclave, user_keypair, storage_keypair):
        """process_message returns properly encrypted outputs."""
        enclave_public_key = enclave.get_info().enclave_public_key

        # Prepare encrypted input
        user_message = b"What is 2+2?"
        encrypted_input = encrypt_to_public_key(
            enclave_public_key,
            user_message,
            EncryptionContext.CLIENT_TO_ENCLAVE.value,
        )

        # Mock the inference call (returns tuple: content, input_tokens, output_tokens)
        with patch.object(enclave, "_call_inference", new_callable=AsyncMock, return_value=("4", 5, 1)):
            result = await enclave.process_message(
                encrypted_message=encrypted_input,
                encrypted_history=[],
                storage_public_key=storage_keypair.public_key,
                transport_public_key=user_keypair.public_key,
                model="test-model",
            )

        # Verify stored_user_message can be decrypted
        decrypted_user = decrypt_with_private_key(
            storage_keypair.private_key,
            result.stored_user_message,
            EncryptionContext.USER_MESSAGE_STORAGE.value,
        )
        assert decrypted_user == user_message

        # Verify stored_assistant_message can be decrypted
        decrypted_assistant = decrypt_with_private_key(
            storage_keypair.private_key,
            result.stored_assistant_message,
            EncryptionContext.ASSISTANT_MESSAGE_STORAGE.value,
        )
        assert decrypted_assistant == b"4"

        # Verify transport_response can be decrypted by user
        decrypted_transport = decrypt_with_private_key(
            user_keypair.private_key,
            result.transport_response,
            EncryptionContext.ENCLAVE_TO_CLIENT.value,
        )
        assert decrypted_transport == b"4"

    @pytest.mark.asyncio
    async def test_process_message_with_history(self, enclave, user_keypair, storage_keypair):
        """process_message correctly decrypts and uses history."""
        enclave_public_key = enclave.get_info().enclave_public_key

        # Prepare history (re-encrypted to enclave by client)
        history_user = encrypt_to_public_key(
            enclave_public_key,
            b"What is 1+1?",
            EncryptionContext.CLIENT_TO_ENCLAVE.value,
        )
        history_assistant = encrypt_to_public_key(
            enclave_public_key,
            b"2",
            EncryptionContext.CLIENT_TO_ENCLAVE.value,
        )

        # New message
        new_message = encrypt_to_public_key(
            enclave_public_key,
            b"And 2+2?",
            EncryptionContext.CLIENT_TO_ENCLAVE.value,
        )

        # Track what inference received
        inference_calls = []

        async def mock_inference(user_msg, history, model):
            inference_calls.append((user_msg, history, model))
            return ("4", 10, 1)  # Returns tuple: content, input_tokens, output_tokens

        with patch.object(
            enclave,
            "_call_inference",
            side_effect=mock_inference,
        ):
            await enclave.process_message(
                encrypted_message=new_message,
                encrypted_history=[history_user, history_assistant],
                storage_public_key=storage_keypair.public_key,
                transport_public_key=user_keypair.public_key,
                model="test-model",
            )

        # Verify inference received the history
        assert len(inference_calls) == 1
        user_msg, history, model = inference_calls[0]

        assert user_msg == "And 2+2?"
        assert len(history) == 2
        assert history[0].role == "user"
        assert history[0].content == "What is 1+1?"
        assert history[1].role == "assistant"
        assert history[1].content == "2"


# =============================================================================
# Streaming Tests
# =============================================================================


class TestStreamingMessageProcessing:
    """Tests for streaming message processing."""

    @pytest.fixture
    def enclave(self):
        return MockEnclave(
            inference_url="http://mock.llm",
            inference_token="test_token",
        )

    @pytest.fixture
    def user_keypair(self):
        return generate_x25519_keypair()

    @pytest.fixture
    def storage_keypair(self):
        return generate_x25519_keypair()

    @pytest.mark.asyncio
    async def test_process_message_stream_yields_chunks(self, enclave, user_keypair, storage_keypair):
        """process_message_stream yields chunks and final result."""
        enclave_public_key = enclave.get_info().enclave_public_key

        # Prepare input
        encrypted_input = encrypt_to_public_key(
            enclave_public_key,
            b"Tell me a story",
            EncryptionContext.CLIENT_TO_ENCLAVE.value,
        )

        # Mock streaming inference - yields (chunk, is_thinking) tuples
        async def mock_stream(*args, **kwargs):
            for chunk in ["Once", " upon", " a", " time"]:
                yield (chunk, False)

        with patch.object(
            enclave,
            "_call_inference_stream",
            side_effect=mock_stream,
        ):
            chunks = []
            final_result = None

            async for chunk, result in enclave.process_message_stream(
                encrypted_message=encrypted_input,
                encrypted_history=[],
                storage_public_key=storage_keypair.public_key,
                transport_public_key=user_keypair.public_key,
                model="test-model",
            ):
                chunks.append(chunk)
                if result is not None:
                    final_result = result

        # Should have received all chunks plus empty final
        assert chunks == ["Once", " upon", " a", " time", ""]

        # Final result should be ProcessedMessage
        assert final_result is not None
        assert isinstance(final_result, ProcessedMessage)

        # Verify the full response was encrypted
        decrypted = decrypt_with_private_key(
            user_keypair.private_key,
            final_result.transport_response,
            EncryptionContext.ENCLAVE_TO_CLIENT.value,
        )
        assert decrypted == b"Once upon a time"


# =============================================================================
# Singleton Tests
# =============================================================================


class TestEnclaveSingleton:
    """Tests for enclave singleton management."""

    def test_get_enclave_returns_same_instance(self):
        """get_enclave returns the same instance."""
        reset_enclave()  # Clear any existing instance

        with patch("core.config.settings") as mock_settings:
            mock_settings.HF_API_URL = "http://mock.llm"
            mock_settings.HUGGINGFACE_TOKEN = "test_token"
            mock_settings.ENCLAVE_INFERENCE_TIMEOUT = 120.0

            enclave1 = get_enclave()
            enclave2 = get_enclave()

            assert enclave1 is enclave2

    def test_reset_enclave_creates_new_instance(self):
        """reset_enclave forces new instance creation."""
        reset_enclave()

        with patch("core.config.settings") as mock_settings:
            mock_settings.HF_API_URL = "http://mock.llm"
            mock_settings.HUGGINGFACE_TOKEN = "test_token"
            mock_settings.ENCLAVE_INFERENCE_TIMEOUT = 120.0

            enclave1 = get_enclave()
            public_key1 = enclave1.get_info().enclave_public_key

            reset_enclave()

            enclave2 = get_enclave()
            public_key2 = enclave2.get_info().enclave_public_key

            # Different instances should have different keys
            assert public_key1 != public_key2

    def test_reset_enclave_for_testing(self):
        """reset_enclave is useful for test isolation."""
        reset_enclave()  # Clean up after tests


# =============================================================================
# Integration Tests
# =============================================================================


class TestEnclaveIntegration:
    """Integration tests for complete enclave flows."""

    def test_full_personal_message_flow(self):
        """Test complete flow for personal (non-org) message."""
        enclave = MockEnclave(inference_token="test")
        user = generate_x25519_keypair()
        enclave_public_key = enclave.get_info().enclave_public_key

        # 1. User encrypts message to enclave
        user_message = b"What's the meaning of life?"
        encrypted_to_enclave = encrypt_to_public_key(
            enclave_public_key,
            user_message,
            EncryptionContext.CLIENT_TO_ENCLAVE.value,
        )

        # 2. Enclave decrypts (simulating process_message start)
        decrypted = enclave.decrypt_transport_message(encrypted_to_enclave)
        assert decrypted == user_message

        # 3. Enclave re-encrypts for storage (to user's key)
        stored_user = enclave.encrypt_for_storage(
            decrypted,
            user.public_key,
            is_assistant=False,
        )

        # 4. Simulate LLM response
        assistant_response = b"42"
        stored_assistant = enclave.encrypt_for_storage(
            assistant_response,
            user.public_key,
            is_assistant=True,
        )

        # 5. Encrypt response for transport
        transport_response = enclave.encrypt_for_transport(
            assistant_response,
            user.public_key,
        )

        # 6. User decrypts transport response
        decrypted_response = decrypt_with_private_key(
            user.private_key,
            transport_response,
            EncryptionContext.ENCLAVE_TO_CLIENT.value,
        )
        assert decrypted_response == b"42"

        # 7. Later: User can decrypt stored messages
        stored_user_decrypted = decrypt_with_private_key(
            user.private_key,
            stored_user,
            EncryptionContext.USER_MESSAGE_STORAGE.value,
        )
        assert stored_user_decrypted == user_message

        stored_assistant_decrypted = decrypt_with_private_key(
            user.private_key,
            stored_assistant,
            EncryptionContext.ASSISTANT_MESSAGE_STORAGE.value,
        )
        assert stored_assistant_decrypted == b"42"

    def test_full_org_message_flow(self):
        """Test complete flow for org message (different storage key)."""
        enclave = MockEnclave(inference_token="test")
        user = generate_x25519_keypair()  # User's transport key
        org = generate_x25519_keypair()  # Org's storage key (all members have)
        enclave_public_key = enclave.get_info().enclave_public_key

        # 1. User encrypts message to enclave
        user_message = b"Org confidential question"
        encrypted_to_enclave = encrypt_to_public_key(
            enclave_public_key,
            user_message,
            EncryptionContext.CLIENT_TO_ENCLAVE.value,
        )

        # 2. Enclave decrypts
        decrypted = enclave.decrypt_transport_message(encrypted_to_enclave)

        # 3. Enclave re-encrypts for storage (to ORG's key, not user's)
        stored_user = enclave.encrypt_for_storage(
            decrypted,
            org.public_key,  # Org storage key
            is_assistant=False,
        )

        # 4. Simulate LLM response
        assistant_response = b"Org confidential answer"
        enclave.encrypt_for_storage(
            assistant_response,
            org.public_key,  # Org storage key
            is_assistant=True,
        )

        # 5. Transport response still goes to user's key
        transport_response = enclave.encrypt_for_transport(
            assistant_response,
            user.public_key,  # User's transport key
        )

        # 6. User decrypts immediate response
        decrypted_response = decrypt_with_private_key(
            user.private_key,
            transport_response,
            EncryptionContext.ENCLAVE_TO_CLIENT.value,
        )
        assert decrypted_response == assistant_response

        # 7. Any org member (who has org private key) can decrypt stored messages
        stored_user_decrypted = decrypt_with_private_key(
            org.private_key,  # Using org key
            stored_user,
            EncryptionContext.USER_MESSAGE_STORAGE.value,
        )
        assert stored_user_decrypted == user_message

        # 8. User's personal key CANNOT decrypt stored messages (they're org-encrypted)
        from cryptography.exceptions import InvalidTag

        with pytest.raises(InvalidTag):
            decrypt_with_private_key(
                user.private_key,
                stored_user,
                EncryptionContext.USER_MESSAGE_STORAGE.value,
            )


# =============================================================================
# ExtractedFact Tests
# =============================================================================


class TestExtractedFact:
    """Tests for ExtractedFact data structure."""

    def test_extracted_fact_creation(self):
        """ExtractedFact can be created with encrypted payload and fact_id."""
        from core.enclave.mock_enclave import ExtractedFact

        payload = EncryptedPayload(
            ephemeral_public_key=b"\x00" * 32,
            iv=b"\x01" * 16,
            ciphertext=b"encrypted_fact_data",
            auth_tag=b"\x02" * 16,
            hkdf_salt=b"\x03" * 32,
        )
        fact = ExtractedFact(
            encrypted_payload=payload,
            fact_id="test-fact-id-123",
        )

        assert fact.fact_id == "test-fact-id-123"
        assert fact.encrypted_payload.ciphertext == b"encrypted_fact_data"


# =============================================================================
# Fact Extraction Tests
# =============================================================================


class TestFactExtraction:
    """Tests for fact extraction from messages using pattern-based FactExtractor."""

    @pytest.fixture
    def enclave(self):
        return MockEnclave(
            inference_url="http://mock.llm",
            inference_token="test_token",
        )

    @pytest.fixture
    def client_keypair(self):
        return generate_x25519_keypair()

    @pytest.mark.asyncio
    async def test_extract_facts_returns_encrypted_facts(self, enclave, client_keypair):
        """extract_facts returns list of ExtractedFact objects."""
        from core.enclave.mock_enclave import ExtractedFact

        # Pattern-based extraction should find "I prefer dark mode"
        facts = await enclave.extract_facts(
            user_message="I prefer dark mode for coding",
            assistant_response="I'll remember that you prefer dark mode.",
            client_public_key=client_keypair.public_key,
        )

        # Should extract at least one fact from "I prefer dark mode"
        assert len(facts) >= 1
        assert isinstance(facts[0], ExtractedFact)
        assert facts[0].fact_id is not None

    @pytest.mark.asyncio
    async def test_extract_facts_encrypts_to_client_key(self, enclave, client_keypair):
        """Extracted facts are encrypted to client's transport key."""
        import json

        # Pattern should match "I work at Anthropic"
        facts = await enclave.extract_facts(
            user_message="I work at Anthropic",
            assistant_response="Nice, you work at Anthropic!",
            client_public_key=client_keypair.public_key,
        )

        assert len(facts) >= 1

        # Client should be able to decrypt the fact
        decrypted = decrypt_with_private_key(
            client_keypair.private_key,
            facts[0].encrypted_payload,
            EncryptionContext.ENCLAVE_TO_CLIENT.value,
        )

        fact_data = json.loads(decrypted.decode("utf-8"))
        assert fact_data["subject"] == "user"
        assert fact_data["predicate"] == "works_at"
        assert "anthropic" in fact_data["object"].lower()
        assert fact_data["type"] == "identity"
        assert fact_data["source"] == "system"

    @pytest.mark.asyncio
    async def test_extract_facts_handles_empty_response(self, enclave, client_keypair):
        """Handles case where no facts are extracted."""
        # Use a message with no extractable patterns
        facts = await enclave.extract_facts(
            user_message="What's the weather?",
            assistant_response="I don't know the current weather.",
            client_public_key=client_keypair.public_key,
        )

        assert len(facts) == 0

    @pytest.mark.asyncio
    async def test_extract_facts_extracts_preferences(self, enclave, client_keypair):
        """Pattern extractor finds preference patterns."""
        import json

        facts = await enclave.extract_facts(
            user_message="I prefer Python over JavaScript",
            assistant_response="Python is great!",
            client_public_key=client_keypair.public_key,
        )

        assert len(facts) >= 1

        decrypted = decrypt_with_private_key(
            client_keypair.private_key,
            facts[0].encrypted_payload,
            EncryptionContext.ENCLAVE_TO_CLIENT.value,
        )
        fact_data = json.loads(decrypted.decode("utf-8"))
        assert fact_data["predicate"] == "prefers"
        assert fact_data["type"] == "preference"

    @pytest.mark.asyncio
    async def test_extract_facts_extracts_location(self, enclave, client_keypair):
        """Pattern extractor finds location patterns."""
        import json

        facts = await enclave.extract_facts(
            user_message="I live in San Francisco",
            assistant_response="SF is a great city!",
            client_public_key=client_keypair.public_key,
        )

        assert len(facts) >= 1

        decrypted = decrypt_with_private_key(
            client_keypair.private_key,
            facts[0].encrypted_payload,
            EncryptionContext.ENCLAVE_TO_CLIENT.value,
        )
        fact_data = json.loads(decrypted.decode("utf-8"))
        assert fact_data["predicate"] == "located_in"
        assert fact_data["type"] == "identity"

    @pytest.mark.asyncio
    async def test_extract_facts_extracts_interests(self, enclave, client_keypair):
        """Pattern extractor finds interest patterns."""
        import json

        facts = await enclave.extract_facts(
            user_message="I'm interested in machine learning",
            assistant_response="ML is fascinating!",
            client_public_key=client_keypair.public_key,
        )

        assert len(facts) >= 1

        decrypted = decrypt_with_private_key(
            client_keypair.private_key,
            facts[0].encrypted_payload,
            EncryptionContext.ENCLAVE_TO_CLIENT.value,
        )
        fact_data = json.loads(decrypted.decode("utf-8"))
        assert fact_data["predicate"] == "interested_in"
        assert fact_data["type"] == "preference"

    @pytest.mark.asyncio
    async def test_extract_facts_extracts_dislikes(self, enclave, client_keypair):
        """Pattern extractor finds dislike patterns."""
        import json

        facts = await enclave.extract_facts(
            user_message="I don't like Java",
            assistant_response="I understand.",
            client_public_key=client_keypair.public_key,
        )

        assert len(facts) >= 1

        decrypted = decrypt_with_private_key(
            client_keypair.private_key,
            facts[0].encrypted_payload,
            EncryptionContext.ENCLAVE_TO_CLIENT.value,
        )
        fact_data = json.loads(decrypted.decode("utf-8"))
        assert fact_data["predicate"] == "dislikes"
        assert fact_data["type"] == "preference"

    @pytest.mark.asyncio
    async def test_extract_facts_extracts_plans(self, enclave, client_keypair):
        """Pattern extractor finds plan patterns."""
        import json

        facts = await enclave.extract_facts(
            user_message="I plan to learn Rust",
            assistant_response="Rust is a good choice!",
            client_public_key=client_keypair.public_key,
        )

        assert len(facts) >= 1

        decrypted = decrypt_with_private_key(
            client_keypair.private_key,
            facts[0].encrypted_payload,
            EncryptionContext.ENCLAVE_TO_CLIENT.value,
        )
        fact_data = json.loads(decrypted.decode("utf-8"))
        assert fact_data["predicate"] == "plans_to"
        assert fact_data["type"] == "plan"


# =============================================================================
# Streaming with Fact Extraction Tests
# =============================================================================


class TestStreamingWithFactExtraction:
    """Tests for streaming message processing with fact extraction."""

    @pytest.fixture
    def enclave(self):
        return MockEnclave(
            inference_url="http://mock.llm",
            inference_token="test_token",
        )

    @pytest.fixture
    def user_keypair(self):
        return generate_x25519_keypair()

    @pytest.fixture
    def storage_keypair(self):
        return generate_x25519_keypair()

    @pytest.mark.asyncio
    async def test_streaming_includes_extracted_facts_in_final_chunk(self, enclave, user_keypair, storage_keypair):
        """Streaming returns extracted facts in the final chunk."""
        from core.enclave.mock_enclave import ExtractedFact

        enclave_public_key = enclave.get_info().enclave_public_key

        # Prepare input
        encrypted_input = encrypt_to_public_key(
            enclave_public_key,
            b"I love Python programming",
            EncryptionContext.CLIENT_TO_ENCLAVE.value,
        )

        # Mock streaming inference - yields (chunk, is_thinking) tuples
        async def mock_stream(*args, **kwargs):
            for chunk in ["That's", " great!"]:
                yield (chunk, False)

        # Mock fact extraction
        mock_facts = [
            ExtractedFact(
                encrypted_payload=EncryptedPayload(
                    ephemeral_public_key=b"\x00" * 32,
                    iv=b"\x01" * 16,
                    ciphertext=b"fact",
                    auth_tag=b"\x02" * 16,
                    hkdf_salt=b"\x03" * 32,
                ),
                fact_id="test-fact-id",
            )
        ]

        with patch.object(enclave, "_call_inference_stream", side_effect=mock_stream):
            with patch.object(
                enclave,
                "extract_facts",
                new_callable=AsyncMock,
                return_value=mock_facts,
            ):
                with patch.object(enclave, "extract_memories", new_callable=AsyncMock, return_value=[]):
                    chunks = []
                    final_chunk = None

                    async for chunk in enclave.process_message_streaming(
                        encrypted_message=encrypted_input,
                        encrypted_history=[],
                        encrypted_memories=[],  # Option A: client provides pre-encrypted memories
                        facts_context=None,
                        storage_public_key=storage_keypair.public_key,
                        client_public_key=user_keypair.public_key,
                        session_id="test-session",
                        model="test-model",
                    ):
                        chunks.append(chunk)
                        if chunk.is_final:
                            final_chunk = chunk

        # Final chunk should have extracted facts
        assert final_chunk is not None
        assert final_chunk.extracted_facts is not None
        assert len(final_chunk.extracted_facts) == 1
        assert final_chunk.extracted_facts[0].fact_id == "test-fact-id"


# =============================================================================
# DEBUG Flag Tests
# =============================================================================


class TestDebugFlagControl:
    """Tests for DEBUG flag controlling verbose output."""

    @pytest.fixture
    def enclave(self):
        return MockEnclave(
            inference_url="http://mock.llm",
            inference_token="test_token",
        )

    @pytest.fixture
    def user_keypair(self):
        return generate_x25519_keypair()

    @pytest.fixture
    def storage_keypair(self):
        return generate_x25519_keypair()

    @pytest.mark.asyncio
    async def test_debug_false_suppresses_print_output(self, enclave, user_keypair, storage_keypair, capsys):
        """When DEBUG=False, process_message_streaming should not print debug info."""
        enclave_public_key = enclave.get_info().enclave_public_key

        encrypted_input = encrypt_to_public_key(
            enclave_public_key,
            b"Test message",
            EncryptionContext.CLIENT_TO_ENCLAVE.value,
        )

        # Mock streaming inference - yields (chunk, is_thinking) tuples
        async def mock_stream(*args, **kwargs):
            yield ("Hello", False)

        with patch("core.enclave.mock_enclave.settings") as mock_settings:
            mock_settings.DEBUG = False
            with patch.object(enclave, "_call_inference_stream", side_effect=mock_stream):
                with patch.object(enclave, "extract_memories", new_callable=AsyncMock, return_value=[]):
                    with patch.object(enclave, "extract_facts", new_callable=AsyncMock, return_value=[]):
                        async for _ in enclave.process_message_streaming(
                            encrypted_message=encrypted_input,
                            encrypted_history=[],
                            encrypted_memories=[],  # Option A: client provides pre-encrypted memories
                            facts_context=None,
                            storage_public_key=storage_keypair.public_key,
                            client_public_key=user_keypair.public_key,
                            session_id="test-session",
                            model="test-model",
                        ):
                            pass

        captured = capsys.readouterr()
        # When DEBUG=False, should NOT see the encryption flow banner
        assert "ENCRYPTED CHAT FLOW" not in captured.out

    @pytest.mark.asyncio
    async def test_debug_true_shows_print_output(self, enclave, user_keypair, storage_keypair, capsys):
        """When DEBUG=True, process_message_streaming should print debug info."""
        enclave_public_key = enclave.get_info().enclave_public_key

        encrypted_input = encrypt_to_public_key(
            enclave_public_key,
            b"Test message",
            EncryptionContext.CLIENT_TO_ENCLAVE.value,
        )

        # Mock streaming inference - yields (chunk, is_thinking) tuples
        async def mock_stream(*args, **kwargs):
            yield ("Hello", False)

        with patch("core.enclave.mock_enclave.settings") as mock_settings:
            mock_settings.DEBUG = True
            with patch.object(enclave, "_call_inference_stream", side_effect=mock_stream):
                with patch.object(enclave, "extract_memories", new_callable=AsyncMock, return_value=[]):
                    with patch.object(enclave, "extract_facts", new_callable=AsyncMock, return_value=[]):
                        async for _ in enclave.process_message_streaming(
                            encrypted_message=encrypted_input,
                            encrypted_history=[],
                            encrypted_memories=[],  # Option A: client provides pre-encrypted memories
                            facts_context=None,
                            storage_public_key=storage_keypair.public_key,
                            client_public_key=user_keypair.public_key,
                            session_id="test-session",
                            model="test-model",
                        ):
                            pass

        captured = capsys.readouterr()
        # When DEBUG=True, should see the encryption flow banner
        assert "ENCRYPTED CHAT FLOW" in captured.out
