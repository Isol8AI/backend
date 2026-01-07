from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.fernet import Fernet
from core.config import settings
import os
import base64

class EncryptionService:
    def __init__(self):
        # Master key is used to encrypt/decrypt user keys.
        # It must be a valid Fernet key (32 bytes url-safe base64)
        try:
            self.master_cipher = Fernet(settings.MASTER_KEY)
        except Exception as e:
            print(f"Error initializing Master Cipher: {e}")
            self.master_cipher = None

    def generate_user_key(self) -> bytes:
        """Generates a new AES-256 key (32 bytes) for a user."""
        return AESGCM.generate_key(bit_length=256)

    def encrypt_user_key_for_storage(self, user_key: bytes) -> str:
        """Encrypts the raw user key using the Master Key for DB storage."""
        if not self.master_cipher:
            raise ValueError("Master Cipher not initialized")
        return self.master_cipher.encrypt(user_key).decode('utf-8')

    def decrypt_user_key_from_storage(self, encrypted_user_key: str) -> bytes:
        """Decrypts the stored user key using the Master Key."""
        if not self.master_cipher:
            raise ValueError("Master Cipher not initialized")
        return self.master_cipher.decrypt(encrypted_user_key.encode('utf-8'))

    def encrypt_message(self, message: str, user_key: bytes) -> bytes:
        """
        Encrypts a message using the user's key via AES-256-GCM.
        Returns the IV + Ciphertext concatenated.
        """
        aesgcm = AESGCM(user_key)
        nonce = os.urandom(12)  # NIST recommends 96-bit (12 byte) IV for GCM
        ciphertext = aesgcm.encrypt(nonce, message.encode('utf-8'), None)
        return nonce + ciphertext

    def decrypt_message(self, encrypted_data: bytes, user_key: bytes) -> str:
        """
        Decrypts a message using the user's key via AES-256-GCM.
        Expects input to be IV (12 bytes) + Ciphertext.
        """
        aesgcm = AESGCM(user_key)
        nonce = encrypted_data[:12]
        ciphertext = encrypted_data[12:]
        plaintext = aesgcm.decrypt(nonce, ciphertext, None)
        return plaintext.decode('utf-8')

encryption_service = EncryptionService()
