"""
Debug endpoint for encryption verification.

DEVELOPMENT ONLY - Do not expose in production.

This endpoint provides all encryption data in a format suitable
for manual verification using online crypto tools like:
- Argon2id: https://argon2.online/
- AES-GCM: CyberChef (Operations -> Encryption -> AES Decrypt)
- X25519: https://paulmillr.com/noble/
- HKDF: CyberChef (Operations -> HKDF)
"""
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from core.auth import get_current_user, AuthContext
from core.database import get_db
from core.enclave import get_enclave, EncryptionContext
from models.user import User
from models.organization import Organization
from models.organization_membership import OrganizationMembership
from models.session import Session
from models.message import Message

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/debug/encryption", tags=["debug"])


# =============================================================================
# Response Models
# =============================================================================

class KeyDerivationInfo(BaseModel):
    """Key derivation algorithm parameters."""
    algorithm: str
    time_cost: int
    memory_cost_kb: int
    parallelism: int
    output_length: int


class EncryptionAlgorithmInfo(BaseModel):
    """Encryption algorithm parameters."""
    cipher: str
    key_length: int
    iv_length: int
    tag_length: int


class KeyExchangeInfo(BaseModel):
    """Key exchange algorithm parameters."""
    algorithm: str
    kdf: str
    kdf_salt_length: int
    derived_key_length: int


class UserKeyInfo(BaseModel):
    """User encryption key information."""
    user_id: str
    public_key_hex: Optional[str]
    encrypted_private_key_hex: Optional[str]
    encrypted_private_key_iv_hex: Optional[str]
    encrypted_private_key_tag_hex: Optional[str]
    key_salt_hex: Optional[str]
    key_derivation: KeyDerivationInfo
    has_encryption_keys: bool
    has_recovery_keys: bool


class EnclaveKeyInfo(BaseModel):
    """Enclave transport key information."""
    transport_public_key_hex: str


class EncryptionContexts(BaseModel):
    """All HKDF context strings."""
    client_to_enclave: str
    enclave_to_client: str
    user_message_storage: str
    assistant_message_storage: str
    org_key_distribution: str


class MembershipKeyInfo(BaseModel):
    """Organization membership key distribution info."""
    role: str
    has_org_key: bool
    encrypted_org_key_ephemeral_hex: Optional[str]
    encrypted_org_key_iv_hex: Optional[str]
    encrypted_org_key_ciphertext_hex: Optional[str]
    encrypted_org_key_tag_hex: Optional[str]
    encrypted_org_key_hkdf_salt_hex: Optional[str]


class OrgKeyInfo(BaseModel):
    """Organization encryption key information."""
    org_id: str
    org_name: str
    org_public_key_hex: Optional[str]
    admin_encrypted_private_key_hex: Optional[str]
    admin_encrypted_private_key_iv_hex: Optional[str]
    admin_encrypted_private_key_tag_hex: Optional[str]
    admin_key_salt_hex: Optional[str]
    has_encryption_keys: bool
    membership: Optional[MembershipKeyInfo]


class SampleMessageInfo(BaseModel):
    """Sample encrypted message structure."""
    message_id: str
    session_id: str
    role: str
    ephemeral_public_key_hex: str
    iv_hex: str
    ciphertext_hex: str
    ciphertext_length: int
    auth_tag_hex: str
    hkdf_salt_hex: str
    storage_context: str


class VerificationSteps(BaseModel):
    """Step-by-step verification instructions."""
    to_decrypt_user_private_key: list[str]
    to_decrypt_message: list[str]
    to_decrypt_org_key: list[str]


class EncryptionReport(BaseModel):
    """Complete encryption report for verification."""
    user: UserKeyInfo
    enclave: EnclaveKeyInfo
    encryption_contexts: EncryptionContexts
    encryption_algorithm: EncryptionAlgorithmInfo
    key_exchange: KeyExchangeInfo
    organizations: list[OrgKeyInfo]
    sample_messages: list[SampleMessageInfo]
    verification_steps: VerificationSteps
    online_tools: dict[str, str]


# =============================================================================
# Debug Endpoint
# =============================================================================

@router.get("/report", response_model=EncryptionReport)
async def get_encryption_report(
    auth: AuthContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Returns all encryption data for the current user.

    Designed for manual verification with online crypto tools.

    DEVELOPMENT ONLY - Do not expose in production.
    """
    # 1. Get user with encryption keys
    user_query = select(User).where(User.id == auth.user_id)
    result = await db.execute(user_query)
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # 2. Build user key info
    user_info = UserKeyInfo(
        user_id=user.id,
        public_key_hex=user.public_key,
        encrypted_private_key_hex=user.encrypted_private_key,
        encrypted_private_key_iv_hex=user.encrypted_private_key_iv,
        encrypted_private_key_tag_hex=user.encrypted_private_key_tag,
        key_salt_hex=user.key_salt,
        key_derivation=KeyDerivationInfo(
            algorithm="Argon2id",
            time_cost=4,
            memory_cost_kb=131072,  # 128MB
            parallelism=2,
            output_length=32,
        ),
        has_encryption_keys=user.has_encryption_keys,
        has_recovery_keys=user.recovery_encrypted_private_key is not None,
    )

    # 3. Get enclave transport key
    enclave = get_enclave()
    enclave_info = EnclaveKeyInfo(
        transport_public_key_hex=enclave.get_transport_public_key(),
    )

    # 4. Encryption contexts
    contexts = EncryptionContexts(
        client_to_enclave=EncryptionContext.CLIENT_TO_ENCLAVE.value,
        enclave_to_client=EncryptionContext.ENCLAVE_TO_CLIENT.value,
        user_message_storage=EncryptionContext.USER_MESSAGE_STORAGE.value,
        assistant_message_storage=EncryptionContext.ASSISTANT_MESSAGE_STORAGE.value,
        org_key_distribution=EncryptionContext.ORG_KEY_DISTRIBUTION.value,
    )

    # 5. Static algorithm info
    encryption_algorithm = EncryptionAlgorithmInfo(
        cipher="AES-256-GCM",
        key_length=32,
        iv_length=16,
        tag_length=16,
    )

    key_exchange = KeyExchangeInfo(
        algorithm="X25519 ECDH",
        kdf="HKDF-SHA512",
        kdf_salt_length=32,
        derived_key_length=32,
    )

    # 6. Get user's organization memberships with org keys
    memberships_query = (
        select(OrganizationMembership)
        .options(selectinload(OrganizationMembership.organization))
        .where(OrganizationMembership.user_id == auth.user_id)
    )
    memberships_result = await db.execute(memberships_query)
    memberships = memberships_result.scalars().all()

    organizations = []
    for membership in memberships:
        org = membership.organization
        if org:
            membership_info = MembershipKeyInfo(
                role=membership.role.value if membership.role else "unknown",
                has_org_key=membership.has_org_key,
                encrypted_org_key_ephemeral_hex=membership.encrypted_org_key_ephemeral,
                encrypted_org_key_iv_hex=membership.encrypted_org_key_iv,
                encrypted_org_key_ciphertext_hex=membership.encrypted_org_key_ciphertext,
                encrypted_org_key_tag_hex=membership.encrypted_org_key_tag,
                encrypted_org_key_hkdf_salt_hex=membership.encrypted_org_key_hkdf_salt,
            )

            org_info = OrgKeyInfo(
                org_id=org.id,
                org_name=org.name,
                org_public_key_hex=org.org_public_key,
                admin_encrypted_private_key_hex=org.admin_encrypted_private_key,
                admin_encrypted_private_key_iv_hex=org.admin_encrypted_private_key_iv,
                admin_encrypted_private_key_tag_hex=org.admin_encrypted_private_key_tag,
                admin_key_salt_hex=org.admin_key_salt,
                has_encryption_keys=org.has_encryption_keys,
                membership=membership_info,
            )
            organizations.append(org_info)

    # 7. Get sample messages from user's sessions (up to 5)
    sessions_query = (
        select(Session)
        .where(Session.user_id == auth.user_id)
        .order_by(Session.updated_at.desc())
        .limit(3)
    )
    sessions_result = await db.execute(sessions_query)
    sessions = sessions_result.scalars().all()

    sample_messages = []
    for session in sessions:
        messages_query = (
            select(Message)
            .where(Message.session_id == session.id)
            .order_by(Message.created_at.desc())
            .limit(2)
        )
        messages_result = await db.execute(messages_query)
        messages = messages_result.scalars().all()

        for msg in messages:
            # Determine storage context based on role
            storage_context = (
                EncryptionContext.ASSISTANT_MESSAGE_STORAGE.value
                if msg.role == "assistant"
                else EncryptionContext.USER_MESSAGE_STORAGE.value
            )

            sample_messages.append(SampleMessageInfo(
                message_id=msg.id,
                session_id=msg.session_id,
                role=msg.role,
                ephemeral_public_key_hex=msg.ephemeral_public_key,
                iv_hex=msg.iv,
                ciphertext_hex=msg.ciphertext[:100] + "..." if len(msg.ciphertext) > 100 else msg.ciphertext,
                ciphertext_length=len(msg.ciphertext) // 2,  # hex chars / 2 = bytes
                auth_tag_hex=msg.auth_tag,
                hkdf_salt_hex=msg.hkdf_salt,
                storage_context=storage_context,
            ))

    # 8. Verification steps
    verification_steps = VerificationSteps(
        to_decrypt_user_private_key=[
            "1. Get your passcode (6+ digits)",
            "2. Derive key: Argon2id(passcode, key_salt_hex, t=4, m=128MB, p=2) -> 32-byte key",
            "3. Decrypt: AES-256-GCM.decrypt(derived_key, iv_hex, encrypted_private_key_hex, tag_hex) -> private_key",
            "4. Result: 32-byte X25519 private key (64 hex chars)",
        ],
        to_decrypt_message=[
            "1. Get your decrypted private key (from passcode unlock)",
            "2. ECDH: X25519(your_private_key, message.ephemeral_public_key_hex) -> shared_secret",
            "3. KDF: HKDF-SHA512(shared_secret, hkdf_salt_hex, info=storage_context) -> 32-byte key",
            "4. Decrypt: AES-256-GCM.decrypt(derived_key, iv_hex, ciphertext_hex, auth_tag_hex) -> plaintext",
        ],
        to_decrypt_org_key=[
            "1. Get your decrypted personal private key (from passcode unlock)",
            "2. ECDH: X25519(personal_private_key, encrypted_org_key_ephemeral_hex) -> shared_secret",
            "3. KDF: HKDF-SHA512(shared_secret, encrypted_org_key_hkdf_salt_hex, info='org-key-distribution') -> 32-byte key",
            "4. Decrypt: AES-256-GCM.decrypt(derived_key, encrypted_org_key_iv_hex, encrypted_org_key_ciphertext_hex, encrypted_org_key_tag_hex) -> org_private_key",
            "5. Use org_private_key to decrypt org messages (same process as personal messages)",
        ],
    )

    # 9. Online tools for verification
    online_tools = {
        "argon2id": "https://argon2.online/",
        "aes_gcm": "https://gchq.github.io/CyberChef/ (Operations -> Encryption -> AES Decrypt)",
        "x25519": "https://paulmillr.com/noble/",
        "hkdf": "https://gchq.github.io/CyberChef/ (Operations -> HKDF)",
        "hex_conversion": "https://gchq.github.io/CyberChef/ (Operations -> From Hex / To Hex)",
    }

    return EncryptionReport(
        user=user_info,
        enclave=enclave_info,
        encryption_contexts=contexts,
        encryption_algorithm=encryption_algorithm,
        key_exchange=key_exchange,
        organizations=organizations,
        sample_messages=sample_messages,
        verification_steps=verification_steps,
        online_tools=online_tools,
    )
