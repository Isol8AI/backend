"""
Comprehensive tests for cryptographic primitives.

These tests verify:
- Correct implementation of all crypto operations
- Edge cases and error handling
- Security properties (randomness, authentication)
"""
import pytest
from cryptography.exceptions import InvalidTag

from core.crypto.primitives import (
    KeyPair,
    EncryptedPayload,
    generate_x25519_keypair,
    derive_key_from_ecdh,
    encrypt_aes_gcm,
    decrypt_aes_gcm,
    encrypt_to_public_key,
    decrypt_with_private_key,
    generate_salt,
    generate_recovery_code,
    secure_compare,
)
from tests.utils.crypto_test_utils import derive_key_from_passcode


# =============================================================================
# KeyPair Tests
# =============================================================================

class TestKeyPair:
    """Tests for KeyPair data structure."""

    def test_valid_keypair(self):
        """KeyPair accepts valid 32-byte keys."""
        keypair = KeyPair(
            private_key=b'\x00' * 32,
            public_key=b'\x01' * 32
        )
        assert len(keypair.private_key) == 32
        assert len(keypair.public_key) == 32

    def test_rejects_short_private_key(self):
        """KeyPair rejects private key shorter than 32 bytes."""
        with pytest.raises(ValueError, match="Private key must be 32 bytes"):
            KeyPair(private_key=b'\x00' * 31, public_key=b'\x01' * 32)

    def test_rejects_short_public_key(self):
        """KeyPair rejects public key shorter than 32 bytes."""
        with pytest.raises(ValueError, match="Public key must be 32 bytes"):
            KeyPair(private_key=b'\x00' * 32, public_key=b'\x01' * 31)

    def test_keypair_is_immutable(self):
        """KeyPair is frozen (immutable)."""
        keypair = KeyPair(private_key=b'\x00' * 32, public_key=b'\x01' * 32)
        with pytest.raises(AttributeError):
            keypair.private_key = b'\x02' * 32


# =============================================================================
# EncryptedPayload Tests
# =============================================================================

class TestEncryptedPayload:
    """Tests for EncryptedPayload data structure."""

    def test_valid_payload(self):
        """EncryptedPayload accepts valid field lengths."""
        payload = EncryptedPayload(
            ephemeral_public_key=b'\x00' * 32,
            iv=b'\x01' * 16,
            ciphertext=b'\x02' * 100,
            auth_tag=b'\x03' * 16,
            hkdf_salt=b'\x04' * 32
        )
        assert len(payload.ephemeral_public_key) == 32
        assert len(payload.iv) == 16
        assert len(payload.ciphertext) == 100
        assert len(payload.auth_tag) == 16
        assert len(payload.hkdf_salt) == 32

    def test_rejects_wrong_ephemeral_key_length(self):
        """EncryptedPayload rejects wrong ephemeral key length."""
        with pytest.raises(ValueError, match="Ephemeral public key must be 32 bytes"):
            EncryptedPayload(
                ephemeral_public_key=b'\x00' * 31,
                iv=b'\x01' * 16,
                ciphertext=b'\x02',
                auth_tag=b'\x03' * 16,
                hkdf_salt=b'\x04' * 32
            )

    def test_rejects_wrong_iv_length(self):
        """EncryptedPayload rejects wrong IV length."""
        with pytest.raises(ValueError, match="IV must be 16 bytes"):
            EncryptedPayload(
                ephemeral_public_key=b'\x00' * 32,
                iv=b'\x01' * 15,
                ciphertext=b'\x02',
                auth_tag=b'\x03' * 16,
                hkdf_salt=b'\x04' * 32
            )

    def test_rejects_wrong_auth_tag_length(self):
        """EncryptedPayload rejects wrong auth tag length."""
        with pytest.raises(ValueError, match="Auth tag must be 16 bytes"):
            EncryptedPayload(
                ephemeral_public_key=b'\x00' * 32,
                iv=b'\x01' * 16,
                ciphertext=b'\x02',
                auth_tag=b'\x03' * 15,
                hkdf_salt=b'\x04' * 32
            )

    def test_rejects_wrong_salt_length(self):
        """EncryptedPayload rejects wrong HKDF salt length."""
        with pytest.raises(ValueError, match="HKDF salt must be 32 bytes"):
            EncryptedPayload(
                ephemeral_public_key=b'\x00' * 32,
                iv=b'\x01' * 16,
                ciphertext=b'\x02',
                auth_tag=b'\x03' * 16,
                hkdf_salt=b'\x04' * 31
            )

    def test_payload_is_immutable(self):
        """EncryptedPayload is frozen (immutable)."""
        payload = EncryptedPayload(
            ephemeral_public_key=b'\x00' * 32,
            iv=b'\x01' * 16,
            ciphertext=b'\x02',
            auth_tag=b'\x03' * 16,
            hkdf_salt=b'\x04' * 32
        )
        with pytest.raises(AttributeError):
            payload.ciphertext = b'\x05'


# =============================================================================
# Key Generation Tests
# =============================================================================

class TestKeyGeneration:
    """Tests for key generation functions."""

    def test_generate_keypair_correct_lengths(self):
        """Generated keypair has correct key lengths."""
        keypair = generate_x25519_keypair()
        assert len(keypair.private_key) == 32
        assert len(keypair.public_key) == 32

    def test_generate_keypair_unique(self):
        """Each generated keypair is unique."""
        keypairs = [generate_x25519_keypair() for _ in range(100)]
        private_keys = {kp.private_key for kp in keypairs}
        public_keys = {kp.public_key for kp in keypairs}
        assert len(private_keys) == 100
        assert len(public_keys) == 100

    def test_generate_salt_default_length(self):
        """generate_salt produces 32 bytes by default."""
        salt = generate_salt()
        assert len(salt) == 32

    def test_generate_salt_custom_length(self):
        """generate_salt respects custom length."""
        salt = generate_salt(64)
        assert len(salt) == 64

    def test_generate_salt_unique(self):
        """Each salt is unique."""
        salts = {generate_salt() for _ in range(100)}
        assert len(salts) == 100

    def test_generate_recovery_code_default_length(self):
        """Recovery code is 20 digits by default."""
        code = generate_recovery_code()
        assert len(code) == 20
        assert code.isdigit()

    def test_generate_recovery_code_custom_length(self):
        """Recovery code respects custom length."""
        code = generate_recovery_code(10)
        assert len(code) == 10
        assert code.isdigit()

    def test_generate_recovery_code_unique(self):
        """Each recovery code is unique."""
        codes = {generate_recovery_code() for _ in range(100)}
        assert len(codes) == 100


# =============================================================================
# Passcode Derivation Tests
# =============================================================================

class TestPasscodeDerivation:
    """Tests for Argon2id passcode derivation."""

    # Use fast parameters for testing
    TEST_PARAMS = {
        'time_cost': 1,
        'memory_cost': 16384,  # 16 MB
        'parallelism': 1,
    }

    def test_derives_32_byte_key(self):
        """Derivation produces 32-byte key."""
        salt = generate_salt()
        key = derive_key_from_passcode("123456", salt, **self.TEST_PARAMS)
        assert len(key) == 32

    def test_deterministic_with_same_inputs(self):
        """Same passcode + salt produces same key."""
        salt = generate_salt()
        key1 = derive_key_from_passcode("123456", salt, **self.TEST_PARAMS)
        key2 = derive_key_from_passcode("123456", salt, **self.TEST_PARAMS)
        assert key1 == key2

    def test_different_passcodes_different_keys(self):
        """Different passcodes produce different keys."""
        salt = generate_salt()
        key1 = derive_key_from_passcode("123456", salt, **self.TEST_PARAMS)
        key2 = derive_key_from_passcode("654321", salt, **self.TEST_PARAMS)
        assert key1 != key2

    def test_different_salts_different_keys(self):
        """Different salts produce different keys."""
        key1 = derive_key_from_passcode("123456", generate_salt(), **self.TEST_PARAMS)
        key2 = derive_key_from_passcode("123456", generate_salt(), **self.TEST_PARAMS)
        assert key1 != key2

    def test_rejects_empty_passcode(self):
        """Empty passcode raises ValueError."""
        with pytest.raises(ValueError, match="Passcode cannot be empty"):
            derive_key_from_passcode("", generate_salt(), **self.TEST_PARAMS)

    def test_rejects_wrong_salt_length(self):
        """Salt must be exactly 32 bytes."""
        with pytest.raises(ValueError, match="Salt must be 32 bytes"):
            derive_key_from_passcode("123456", b'\x00' * 16, **self.TEST_PARAMS)


# =============================================================================
# ECDH Key Derivation Tests
# =============================================================================

class TestECDHDerivation:
    """Tests for X25519 ECDH + HKDF key derivation."""

    def test_derives_32_byte_key(self):
        """Derivation produces 32-byte key."""
        alice = generate_x25519_keypair()
        bob = generate_x25519_keypair()
        key, salt = derive_key_from_ecdh(
            alice.private_key, bob.public_key, "test-context"
        )
        assert len(key) == 32
        assert len(salt) == 32

    def test_ecdh_symmetry(self):
        """Both parties derive the same key (with same salt)."""
        alice = generate_x25519_keypair()
        bob = generate_x25519_keypair()

        key1, salt = derive_key_from_ecdh(
            alice.private_key, bob.public_key, "test-context"
        )
        key2, _ = derive_key_from_ecdh(
            bob.private_key, alice.public_key, "test-context", salt=salt
        )

        assert key1 == key2

    def test_different_contexts_different_keys(self):
        """Different context strings produce different keys."""
        alice = generate_x25519_keypair()
        bob = generate_x25519_keypair()
        salt = generate_salt()

        key1, _ = derive_key_from_ecdh(
            alice.private_key, bob.public_key, "context-a", salt=salt
        )
        key2, _ = derive_key_from_ecdh(
            alice.private_key, bob.public_key, "context-b", salt=salt
        )

        assert key1 != key2

    def test_different_salts_different_keys(self):
        """Different salts produce different keys."""
        alice = generate_x25519_keypair()
        bob = generate_x25519_keypair()

        key1, _ = derive_key_from_ecdh(
            alice.private_key, bob.public_key, "test", salt=generate_salt()
        )
        key2, _ = derive_key_from_ecdh(
            alice.private_key, bob.public_key, "test", salt=generate_salt()
        )

        assert key1 != key2

    def test_generates_random_salt_if_not_provided(self):
        """When salt is None, generates random salt."""
        alice = generate_x25519_keypair()
        bob = generate_x25519_keypair()

        _, salt1 = derive_key_from_ecdh(alice.private_key, bob.public_key, "test")
        _, salt2 = derive_key_from_ecdh(alice.private_key, bob.public_key, "test")

        assert salt1 != salt2

    def test_rejects_wrong_key_lengths(self):
        """Rejects keys that aren't 32 bytes."""
        with pytest.raises(ValueError, match="Private key must be 32 bytes"):
            derive_key_from_ecdh(b'\x00' * 31, b'\x01' * 32, "test")

        with pytest.raises(ValueError, match="Public key must be 32 bytes"):
            derive_key_from_ecdh(b'\x00' * 32, b'\x01' * 31, "test")

    def test_rejects_wrong_salt_length(self):
        """Rejects salt that isn't 32 bytes."""
        alice = generate_x25519_keypair()
        bob = generate_x25519_keypair()

        with pytest.raises(ValueError, match="Salt must be 32 bytes"):
            derive_key_from_ecdh(
                alice.private_key, bob.public_key, "test", salt=b'\x00' * 16
            )


# =============================================================================
# AES-GCM Tests
# =============================================================================

class TestAESGCM:
    """Tests for AES-256-GCM encryption/decryption."""

    @pytest.fixture
    def key(self):
        return generate_salt(32)

    def test_encrypt_decrypt_roundtrip(self, key):
        """Encrypted data can be decrypted."""
        plaintext = b"Hello, World!"
        iv, ciphertext, tag = encrypt_aes_gcm(key, plaintext)
        result = decrypt_aes_gcm(key, iv, ciphertext, tag)
        assert result == plaintext

    def test_encrypt_produces_different_iv_each_time(self, key):
        """Each encryption uses a unique IV."""
        plaintext = b"test"
        ivs = {encrypt_aes_gcm(key, plaintext)[0] for _ in range(100)}
        assert len(ivs) == 100

    def test_ciphertext_length_matches_plaintext(self, key):
        """Ciphertext is same length as plaintext."""
        plaintext = b"x" * 100
        _, ciphertext, _ = encrypt_aes_gcm(key, plaintext)
        assert len(ciphertext) == len(plaintext)

    def test_iv_is_16_bytes(self, key):
        """IV is 16 bytes."""
        iv, _, _ = encrypt_aes_gcm(key, b"test")
        assert len(iv) == 16

    def test_tag_is_16_bytes(self, key):
        """Auth tag is 16 bytes."""
        _, _, tag = encrypt_aes_gcm(key, b"test")
        assert len(tag) == 16

    def test_wrong_key_fails(self, key):
        """Decryption with wrong key fails."""
        plaintext = b"secret"
        iv, ciphertext, tag = encrypt_aes_gcm(key, plaintext)
        wrong_key = generate_salt(32)

        with pytest.raises(InvalidTag):
            decrypt_aes_gcm(wrong_key, iv, ciphertext, tag)

    def test_tampered_ciphertext_fails(self, key):
        """Tampered ciphertext fails authentication."""
        plaintext = b"secret"
        iv, ciphertext, tag = encrypt_aes_gcm(key, plaintext)
        tampered = bytes([ciphertext[0] ^ 1]) + ciphertext[1:]

        with pytest.raises(InvalidTag):
            decrypt_aes_gcm(key, iv, tampered, tag)

    def test_tampered_tag_fails(self, key):
        """Tampered auth tag fails authentication."""
        plaintext = b"secret"
        iv, ciphertext, tag = encrypt_aes_gcm(key, plaintext)
        tampered_tag = bytes([tag[0] ^ 1]) + tag[1:]

        with pytest.raises(InvalidTag):
            decrypt_aes_gcm(key, iv, ciphertext, tampered_tag)

    def test_associated_data_authenticated(self, key):
        """AAD is authenticated but not encrypted."""
        plaintext = b"message"
        aad = b"metadata"

        iv, ciphertext, tag = encrypt_aes_gcm(key, plaintext, aad)
        result = decrypt_aes_gcm(key, iv, ciphertext, tag, aad)
        assert result == plaintext

    def test_wrong_aad_fails(self, key):
        """Wrong AAD fails authentication."""
        plaintext = b"message"
        aad = b"metadata"

        iv, ciphertext, tag = encrypt_aes_gcm(key, plaintext, aad)

        with pytest.raises(InvalidTag):
            decrypt_aes_gcm(key, iv, ciphertext, tag, b"wrong")

    def test_missing_aad_fails(self, key):
        """Missing AAD when required fails authentication."""
        plaintext = b"message"
        aad = b"metadata"

        iv, ciphertext, tag = encrypt_aes_gcm(key, plaintext, aad)

        with pytest.raises(InvalidTag):
            decrypt_aes_gcm(key, iv, ciphertext, tag, None)

    def test_rejects_wrong_key_length(self):
        """Key must be exactly 32 bytes."""
        with pytest.raises(ValueError, match="Key must be 32 bytes"):
            encrypt_aes_gcm(b'\x00' * 16, b"test")

    def test_decrypt_rejects_wrong_iv_length(self, key):
        """Decrypt rejects wrong IV length."""
        with pytest.raises(ValueError, match="IV must be 16 bytes"):
            decrypt_aes_gcm(key, b'\x00' * 15, b"test", b'\x00' * 16)

    def test_decrypt_rejects_wrong_tag_length(self, key):
        """Decrypt rejects wrong auth tag length."""
        with pytest.raises(ValueError, match="Auth tag must be 16 bytes"):
            decrypt_aes_gcm(key, b'\x00' * 16, b"test", b'\x00' * 15)


# =============================================================================
# Public Key Encryption Tests
# =============================================================================

class TestPublicKeyEncryption:
    """Tests for ephemeral ECDH encrypt-to-public-key."""

    def test_encrypt_decrypt_roundtrip(self):
        """Message encrypted to public key can be decrypted with private key."""
        recipient = generate_x25519_keypair()
        plaintext = b"Hello, recipient!"
        context = "test-context"

        payload = encrypt_to_public_key(recipient.public_key, plaintext, context)
        result = decrypt_with_private_key(recipient.private_key, payload, context)

        assert result == plaintext

    def test_different_ephemeral_keys_each_time(self):
        """Each encryption uses a unique ephemeral key."""
        recipient = generate_x25519_keypair()
        plaintext = b"test"
        context = "test"

        payloads = [
            encrypt_to_public_key(recipient.public_key, plaintext, context)
            for _ in range(100)
        ]
        ephemeral_keys = {p.ephemeral_public_key for p in payloads}

        assert len(ephemeral_keys) == 100

    def test_payload_fields_correct_lengths(self):
        """Payload has correct field lengths."""
        recipient = generate_x25519_keypair()
        payload = encrypt_to_public_key(recipient.public_key, b"test", "ctx")

        assert len(payload.ephemeral_public_key) == 32
        assert len(payload.iv) == 16
        assert len(payload.auth_tag) == 16
        assert len(payload.hkdf_salt) == 32

    def test_wrong_private_key_fails(self):
        """Wrong private key fails to decrypt."""
        recipient = generate_x25519_keypair()
        attacker = generate_x25519_keypair()

        payload = encrypt_to_public_key(recipient.public_key, b"secret", "ctx")

        with pytest.raises(InvalidTag):
            decrypt_with_private_key(attacker.private_key, payload, "ctx")

    def test_wrong_context_fails(self):
        """Wrong context fails to decrypt."""
        recipient = generate_x25519_keypair()

        payload = encrypt_to_public_key(recipient.public_key, b"secret", "context-a")

        with pytest.raises(InvalidTag):
            decrypt_with_private_key(recipient.private_key, payload, "context-b")

    def test_large_message(self):
        """Can encrypt large messages."""
        recipient = generate_x25519_keypair()
        plaintext = b"x" * 1_000_000  # 1 MB

        payload = encrypt_to_public_key(recipient.public_key, plaintext, "large")
        result = decrypt_with_private_key(recipient.private_key, payload, "large")

        assert result == plaintext

    def test_empty_message(self):
        """Can encrypt empty message."""
        recipient = generate_x25519_keypair()

        payload = encrypt_to_public_key(recipient.public_key, b"", "empty")
        result = decrypt_with_private_key(recipient.private_key, payload, "empty")

        assert result == b""

    def test_rejects_wrong_public_key_length(self):
        """encrypt_to_public_key rejects wrong key length."""
        with pytest.raises(ValueError, match="Recipient public key must be 32 bytes"):
            encrypt_to_public_key(b'\x00' * 31, b"test", "ctx")

    def test_decrypt_rejects_wrong_private_key_length(self):
        """decrypt_with_private_key rejects wrong key length."""
        recipient = generate_x25519_keypair()
        payload = encrypt_to_public_key(recipient.public_key, b"test", "ctx")

        with pytest.raises(ValueError, match="Private key must be 32 bytes"):
            decrypt_with_private_key(b'\x00' * 31, payload, "ctx")


# =============================================================================
# Utility Tests
# =============================================================================

class TestSecureCompare:
    """Tests for constant-time comparison."""

    def test_equal_returns_true(self):
        """Equal byte strings return True."""
        a = b"hello"
        assert secure_compare(a, a) is True
        assert secure_compare(b"test", b"test") is True

    def test_different_returns_false(self):
        """Different byte strings return False."""
        assert secure_compare(b"hello", b"world") is False
        assert secure_compare(b"abc", b"abd") is False

    def test_different_lengths_returns_false(self):
        """Different length strings return False."""
        assert secure_compare(b"short", b"longer") is False


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """End-to-end integration tests simulating real usage."""

    def test_user_key_encryption_flow(self):
        """
        Simulate user key creation and unlock flow:
        1. Generate keypair
        2. Encrypt private key with passcode
        3. Store encrypted key (simulated)
        4. Later: derive key from passcode
        5. Decrypt private key
        6. Use to decrypt messages
        """
        # Use fast parameters for test
        params = {'time_cost': 1, 'memory_cost': 16384, 'parallelism': 1}

        # 1. Generate user keypair
        user = generate_x25519_keypair()
        passcode = "123456"
        salt = generate_salt()

        # 2. Encrypt private key with passcode
        passcode_key = derive_key_from_passcode(passcode, salt, **params)
        iv, encrypted_private_key, tag = encrypt_aes_gcm(passcode_key, user.private_key)

        # 3. "Store" in database (just keep references)
        stored_public = user.public_key
        stored_encrypted_private = encrypted_private_key
        stored_iv = iv
        stored_tag = tag
        stored_salt = salt

        # 4-5. Later: unlock with passcode
        derived_key = derive_key_from_passcode(passcode, stored_salt, **params)
        recovered_private = decrypt_aes_gcm(derived_key, stored_iv, stored_encrypted_private, stored_tag)

        assert recovered_private == user.private_key

        # 6. Use to decrypt a message
        message = b"Secret message for user"
        payload = encrypt_to_public_key(stored_public, message, "test")
        decrypted = decrypt_with_private_key(recovered_private, payload, "test")

        assert decrypted == message

    def test_org_key_distribution_flow(self):
        """
        Simulate org key distribution:
        1. Admin creates org keypair
        2. Admin encrypts org private key with org passcode
        3. New member joins (has their own keypair)
        4. Admin re-encrypts org private key to member's public key
        5. Member decrypts org key with their private key
        6. Member can now decrypt org messages
        """
        params = {'time_cost': 1, 'memory_cost': 16384, 'parallelism': 1}

        # 1. Create org keypair
        org = generate_x25519_keypair()
        org_passcode = "admin123"
        org_salt = generate_salt()

        # 2. Encrypt org private key with org passcode (admin escrow)
        passcode_key = derive_key_from_passcode(org_passcode, org_salt, **params)
        org_iv, encrypted_org_key, org_tag = encrypt_aes_gcm(passcode_key, org.private_key)

        # 3. New member has their own keypair
        member = generate_x25519_keypair()

        # 4. Admin distributes org key to member (encrypted to member's public key)
        member_org_key_payload = encrypt_to_public_key(
            member.public_key,
            org.private_key,  # Admin has this decrypted in memory
            "org-key-distribution"
        )

        # 5. Member decrypts org key with their private key
        recovered_org_private = decrypt_with_private_key(
            member.private_key,
            member_org_key_payload,
            "org-key-distribution"
        )

        assert recovered_org_private == org.private_key

        # 6. Member can now decrypt org messages
        org_message = b"Confidential org data"
        payload = encrypt_to_public_key(org.public_key, org_message, "org-storage")
        decrypted = decrypt_with_private_key(recovered_org_private, payload, "org-storage")

        assert decrypted == org_message

    def test_recovery_code_flow(self):
        """
        Simulate recovery code usage:
        1. Generate keypair and recovery code
        2. Encrypt private key with both passcode AND recovery code
        3. User forgets passcode
        4. Use recovery code to recover private key
        """
        params = {'time_cost': 1, 'memory_cost': 16384, 'parallelism': 1}

        # 1. Generate keypair and recovery code
        user = generate_x25519_keypair()
        passcode = "123456"
        recovery_code = generate_recovery_code()

        # 2. Encrypt with both passcode and recovery code
        passcode_salt = generate_salt()
        recovery_salt = generate_salt()

        passcode_key = derive_key_from_passcode(passcode, passcode_salt, **params)
        recovery_key = derive_key_from_passcode(recovery_code, recovery_salt, **params)

        # Encrypt with passcode (primary)
        p_iv, p_encrypted, p_tag = encrypt_aes_gcm(passcode_key, user.private_key)

        # Encrypt with recovery code (backup)
        r_iv, r_encrypted, r_tag = encrypt_aes_gcm(recovery_key, user.private_key)

        # 3. User forgets passcode, uses recovery code
        # 4. Recover private key
        derived_recovery_key = derive_key_from_passcode(recovery_code, recovery_salt, **params)
        recovered_private = decrypt_aes_gcm(derived_recovery_key, r_iv, r_encrypted, r_tag)

        assert recovered_private == user.private_key

    def test_multi_recipient_encryption(self):
        """
        Simulate encrypting to multiple recipients:
        1. Sender wants to send same message to multiple users
        2. Encrypt separately to each recipient's public key
        3. Each recipient can decrypt with their private key
        """
        # Create multiple recipients
        recipients = [generate_x25519_keypair() for _ in range(5)]
        message = b"Announcement for all users"

        # Encrypt to each recipient
        payloads = [
            encrypt_to_public_key(r.public_key, message, "announcement")
            for r in recipients
        ]

        # Each recipient can decrypt
        for recipient, payload in zip(recipients, payloads):
            decrypted = decrypt_with_private_key(
                recipient.private_key, payload, "announcement"
            )
            assert decrypted == message

        # Payloads are all different (different ephemeral keys)
        ephemeral_keys = {p.ephemeral_public_key for p in payloads}
        assert len(ephemeral_keys) == 5
