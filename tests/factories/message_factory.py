"""Factory for creating encrypted Message test instances."""
import uuid

import factory

from models.message import Message, MessageRole
from core.crypto.primitives import (
    generate_x25519_keypair,
    encrypt_to_public_key,
)


def generate_encrypted_payload(content: str = "Test message content") -> dict:
    """
    Generate a valid encrypted payload for testing.

    Uses actual crypto primitives to create realistic encrypted content.
    """
    # Generate a test recipient keypair
    recipient = generate_x25519_keypair()

    # Encrypt the content
    payload = encrypt_to_public_key(
        recipient.public_key,
        content.encode("utf-8"),
        "user-message-storage"
    )

    # Return as hex strings matching the model structure
    return {
        "ephemeral_public_key": payload.ephemeral_public_key.hex(),
        "iv": payload.iv.hex(),
        "ciphertext": payload.ciphertext.hex(),
        "auth_tag": payload.auth_tag.hex(),
        "hkdf_salt": payload.hkdf_salt.hex(),
    }


class EncryptedMessageFactory(factory.Factory):
    """Factory for creating encrypted Message model instances."""

    class Meta:
        model = Message

    id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    session_id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    role = MessageRole.USER.value
    model_used = None
    input_tokens = None
    output_tokens = None

    # Encrypted payload fields - generated lazily
    ephemeral_public_key = factory.LazyAttribute(
        lambda o: generate_encrypted_payload()["ephemeral_public_key"]
    )
    iv = factory.LazyAttribute(
        lambda o: generate_encrypted_payload()["iv"]
    )
    ciphertext = factory.LazyAttribute(
        lambda o: generate_encrypted_payload()["ciphertext"]
    )
    auth_tag = factory.LazyAttribute(
        lambda o: generate_encrypted_payload()["auth_tag"]
    )
    hkdf_salt = factory.LazyAttribute(
        lambda o: generate_encrypted_payload()["hkdf_salt"]
    )

    @classmethod
    def create_with_content(cls, content: str, **kwargs) -> Message:
        """
        Create a message with specific plaintext content (encrypted).

        This encrypts the given content and creates a message with it.
        Useful for tests that need to verify specific content after decryption.
        """
        payload = generate_encrypted_payload(content)
        return cls.create(
            ephemeral_public_key=payload["ephemeral_public_key"],
            iv=payload["iv"],
            ciphertext=payload["ciphertext"],
            auth_tag=payload["auth_tag"],
            hkdf_salt=payload["hkdf_salt"],
            **kwargs
        )


class AssistantEncryptedMessageFactory(EncryptedMessageFactory):
    """Factory for encrypted assistant messages with model attribution."""

    role = MessageRole.ASSISTANT.value
    model_used = "Qwen/Qwen2.5-72B-Instruct"


# Backwards compatibility aliases
MessageFactory = EncryptedMessageFactory
AssistantMessageFactory = AssistantEncryptedMessageFactory
