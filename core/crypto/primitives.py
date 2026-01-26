"""
Cryptographic primitives for the zero-trust LLM platform.

Security Properties:
- All randomness from secrets module (CSPRNG)
- X25519 for key exchange (ephemeral ECDH pattern)
- HKDF-SHA512 with random salt for key derivation
- AES-256-GCM for authenticated encryption

Note: Passcode derivation (Argon2id) is NOT included here because passcodes
never reach the server in production - they stay client-side. The Argon2id
function exists only in tests/utils/crypto_test_utils.py for test vector
generation and cross-platform compatibility testing.

Usage Contexts (must match between encrypt/decrypt):
- "client-to-enclave-transport": Messages from client to enclave
- "enclave-to-client-transport": Responses from enclave to client
- "user-message-storage": User messages stored in database
- "assistant-message-storage": Assistant messages stored in database
- "org-key-distribution": Org private key encrypted to member
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import secrets
import hmac

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from nacl.public import PrivateKey
from nacl.bindings import crypto_scalarmult


# =============================================================================
# Data Structures
# =============================================================================


@dataclass(frozen=True)
class KeyPair:
    """
    X25519 keypair container.

    Attributes:
        private_key: 32-byte private key (KEEP SECRET)
        public_key: 32-byte public key (safe to share)
    """

    private_key: bytes  # 32 bytes
    public_key: bytes  # 32 bytes

    def __post_init__(self):
        if len(self.private_key) != 32:
            raise ValueError("Private key must be 32 bytes")
        if len(self.public_key) != 32:
            raise ValueError("Public key must be 32 bytes")


@dataclass(frozen=True)
class EncryptedPayload:
    """
    Standard encrypted payload structure for storage/transmission.

    This structure is used for all encrypt-to-public-key operations.
    The ephemeral ECDH pattern provides forward secrecy per-message.

    Attributes:
        ephemeral_public_key: Sender's ephemeral public key for ECDH (32 bytes)
        iv: AES-GCM initialization vector (16 bytes)
        ciphertext: Encrypted data (variable length)
        auth_tag: AES-GCM authentication tag (16 bytes)
        hkdf_salt: Random salt used in HKDF derivation (32 bytes)
    """

    ephemeral_public_key: bytes  # 32 bytes
    iv: bytes  # 16 bytes
    ciphertext: bytes  # variable
    auth_tag: bytes  # 16 bytes
    hkdf_salt: bytes  # 32 bytes

    def __post_init__(self):
        if len(self.ephemeral_public_key) != 32:
            raise ValueError("Ephemeral public key must be 32 bytes")
        if len(self.iv) != 16:
            raise ValueError("IV must be 16 bytes")
        if len(self.auth_tag) != 16:
            raise ValueError("Auth tag must be 16 bytes")
        if len(self.hkdf_salt) != 32:
            raise ValueError("HKDF salt must be 32 bytes")


# =============================================================================
# Key Generation
# =============================================================================


def generate_x25519_keypair() -> KeyPair:
    """
    Generate a new X25519 keypair for key exchange.

    Uses cryptographically secure random number generation.

    Returns:
        KeyPair with 32-byte private and public keys

    Example:
        >>> keypair = generate_x25519_keypair()
        >>> len(keypair.private_key)
        32
        >>> len(keypair.public_key)
        32
    """
    private_key = PrivateKey.generate()
    public_key = private_key.public_key

    return KeyPair(private_key=bytes(private_key), public_key=bytes(public_key))


def generate_salt(length: int = 32) -> bytes:
    """
    Generate cryptographically secure random bytes.

    Args:
        length: Number of bytes to generate (default: 32)

    Returns:
        Random bytes of specified length
    """
    return secrets.token_bytes(length)


def generate_recovery_code(length: int = 20) -> str:
    """
    Generate a numeric recovery code.

    Recovery codes are used as a backup to recover encrypted private keys
    if the user forgets their passcode.

    Args:
        length: Number of digits (default: 20)

    Returns:
        String of random digits (e.g., "12345678901234567890")

    Security Note:
        20 digits = ~66 bits of entropy, sufficient for recovery codes
        that are stored offline by users.
    """
    return "".join(str(secrets.randbelow(10)) for _ in range(length))


# =============================================================================
# Key Derivation
# =============================================================================


def derive_key_from_ecdh(
    private_key: bytes,
    public_key: bytes,
    context: str,
    salt: Optional[bytes] = None,
) -> Tuple[bytes, bytes]:
    """
    Derive a symmetric key from X25519 ECDH shared secret using HKDF-SHA512.

    This function:
    1. Computes the X25519 shared secret
    2. Generates a random salt if not provided
    3. Derives a 32-byte key using HKDF-SHA512

    The context string ensures different keys for different purposes
    even with the same keypair combination.

    Args:
        private_key: Our X25519 private key (32 bytes)
        public_key: Their X25519 public key (32 bytes)
        context: Context string for domain separation (e.g., "client-to-enclave-transport")
        salt: Optional HKDF salt. If None, generates random 32-byte salt.

    Returns:
        Tuple of (derived_key, salt) - both 32 bytes.
        The salt MUST be stored alongside ciphertext for decryption.

    Raises:
        ValueError: If keys are wrong length

    Example:
        >>> alice = generate_x25519_keypair()
        >>> bob = generate_x25519_keypair()
        >>> key1, salt = derive_key_from_ecdh(alice.private_key, bob.public_key, "test")
        >>> key2, _ = derive_key_from_ecdh(bob.private_key, alice.public_key, "test", salt)
        >>> key1 == key2
        True
    """
    if len(private_key) != 32:
        raise ValueError("Private key must be 32 bytes")
    if len(public_key) != 32:
        raise ValueError("Public key must be 32 bytes")

    # Generate random salt if not provided
    if salt is None:
        salt = generate_salt(32)
    elif len(salt) != 32:
        raise ValueError("Salt must be 32 bytes")

    # Compute X25519 shared secret
    shared_secret = crypto_scalarmult(private_key, public_key)

    # Derive key using HKDF-SHA512
    hkdf = HKDF(
        algorithm=hashes.SHA512(),
        length=32,
        salt=salt,
        info=context.encode("utf-8"),
    )
    derived_key = hkdf.derive(shared_secret)

    return derived_key, salt


# =============================================================================
# Symmetric Encryption (AES-256-GCM)
# =============================================================================


def encrypt_aes_gcm(
    key: bytes,
    plaintext: bytes,
    associated_data: Optional[bytes] = None,
) -> Tuple[bytes, bytes, bytes]:
    """
    Encrypt data using AES-256-GCM.

    AES-GCM provides both confidentiality and authenticity.
    A random 16-byte IV is generated for each encryption.

    Args:
        key: 32-byte encryption key
        plaintext: Data to encrypt
        associated_data: Optional additional authenticated data (AAD).
                        AAD is authenticated but not encrypted.

    Returns:
        Tuple of (iv, ciphertext, auth_tag):
        - iv: 16 bytes (must be stored for decryption)
        - ciphertext: Same length as plaintext
        - auth_tag: 16 bytes (must be stored for decryption)

    Raises:
        ValueError: If key is not 32 bytes

    Security Note:
        The IV is randomly generated for each call. Never reuse an IV
        with the same key - this is handled automatically.
    """
    if len(key) != 32:
        raise ValueError("Key must be 32 bytes")

    iv = secrets.token_bytes(16)
    aesgcm = AESGCM(key)

    # AESGCM.encrypt returns ciphertext + tag concatenated
    ciphertext_with_tag = aesgcm.encrypt(iv, plaintext, associated_data)

    # Split into ciphertext and tag (tag is last 16 bytes)
    ciphertext = ciphertext_with_tag[:-16]
    auth_tag = ciphertext_with_tag[-16:]

    return iv, ciphertext, auth_tag


def decrypt_aes_gcm(
    key: bytes,
    iv: bytes,
    ciphertext: bytes,
    auth_tag: bytes,
    associated_data: Optional[bytes] = None,
) -> bytes:
    """
    Decrypt data using AES-256-GCM.

    Verifies the authentication tag before returning plaintext.

    Args:
        key: 32-byte decryption key
        iv: 16-byte initialization vector (from encryption)
        ciphertext: Encrypted data
        auth_tag: 16-byte authentication tag (from encryption)
        associated_data: Optional AAD (must match encryption)

    Returns:
        Decrypted plaintext

    Raises:
        ValueError: If key/iv/tag are wrong length
        cryptography.exceptions.InvalidTag: If authentication fails
            (ciphertext was tampered with or wrong key/AAD)
    """
    if len(key) != 32:
        raise ValueError("Key must be 32 bytes")
    if len(iv) != 16:
        raise ValueError("IV must be 16 bytes")
    if len(auth_tag) != 16:
        raise ValueError("Auth tag must be 16 bytes")

    aesgcm = AESGCM(key)

    # Reconstruct ciphertext + tag for decryption
    ciphertext_with_tag = ciphertext + auth_tag

    return aesgcm.decrypt(iv, ciphertext_with_tag, associated_data)


# =============================================================================
# High-Level Encryption (Ephemeral ECDH Pattern)
# =============================================================================


def encrypt_to_public_key(
    recipient_public_key: bytes,
    plaintext: bytes,
    context: str,
) -> EncryptedPayload:
    """
    Encrypt data to a recipient's public key using ephemeral ECDH.

    This implements the ephemeral ECDH pattern:
    1. Generate ephemeral X25519 keypair
    2. Compute shared secret with recipient's public key
    3. Derive symmetric key via HKDF with random salt
    4. Encrypt with AES-256-GCM
    5. Discard ephemeral private key

    The result can only be decrypted by the holder of the recipient's
    private key.

    Args:
        recipient_public_key: Recipient's X25519 public key (32 bytes)
        plaintext: Data to encrypt
        context: Context string for domain separation

    Returns:
        EncryptedPayload containing all data needed for decryption

    Example:
        >>> recipient = generate_x25519_keypair()
        >>> payload = encrypt_to_public_key(
        ...     recipient.public_key,
        ...     b"secret message",
        ...     "user-message-storage"
        ... )
        >>> plaintext = decrypt_with_private_key(
        ...     recipient.private_key,
        ...     payload,
        ...     "user-message-storage"
        ... )
        >>> plaintext
        b'secret message'
    """
    if len(recipient_public_key) != 32:
        raise ValueError("Recipient public key must be 32 bytes")

    if not plaintext:
        raise ValueError("Plaintext cannot be empty")

    # 1. Generate ephemeral keypair
    ephemeral = generate_x25519_keypair()

    # 2-3. ECDH + HKDF with random salt
    symmetric_key, hkdf_salt = derive_key_from_ecdh(
        ephemeral.private_key,
        recipient_public_key,
        context,
    )

    # 4. Encrypt
    iv, ciphertext, auth_tag = encrypt_aes_gcm(symmetric_key, plaintext)

    # 5. Ephemeral private key is discarded (goes out of scope)

    return EncryptedPayload(
        ephemeral_public_key=ephemeral.public_key,
        iv=iv,
        ciphertext=ciphertext,
        auth_tag=auth_tag,
        hkdf_salt=hkdf_salt,
    )


def decrypt_with_private_key(
    private_key: bytes,
    payload: EncryptedPayload,
    context: str,
) -> bytes:
    """
    Decrypt data encrypted with encrypt_to_public_key.

    This performs the inverse of the ephemeral ECDH pattern:
    1. Compute shared secret using our private key and sender's ephemeral public key
    2. Derive the same symmetric key via HKDF (using stored salt)
    3. Decrypt with AES-256-GCM

    Args:
        private_key: Our X25519 private key (32 bytes)
        payload: EncryptedPayload from encrypt_to_public_key
        context: Context string (MUST match encryption context)

    Returns:
        Decrypted plaintext

    Raises:
        ValueError: If private key is wrong length or context doesn't match
        cryptography.exceptions.InvalidTag: If decryption fails
    """
    if len(private_key) != 32:
        raise ValueError("Private key must be 32 bytes")

    # Derive the same symmetric key
    symmetric_key, _ = derive_key_from_ecdh(
        private_key,
        payload.ephemeral_public_key,
        context,
        salt=payload.hkdf_salt,  # Use stored salt
    )

    # Decrypt
    return decrypt_aes_gcm(
        symmetric_key,
        payload.iv,
        payload.ciphertext,
        payload.auth_tag,
    )


# =============================================================================
# Utilities
# =============================================================================


def secure_compare(a: bytes, b: bytes) -> bool:
    """
    Compare two byte strings in constant time.

    Prevents timing attacks when comparing secrets (like auth tags).

    Args:
        a: First byte string
        b: Second byte string

    Returns:
        True if equal, False otherwise
    """
    return hmac.compare_digest(a, b)
