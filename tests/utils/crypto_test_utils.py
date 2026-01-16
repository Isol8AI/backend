"""
Cryptographic test utilities.

This module contains crypto functions used only for testing, not in production.
The passcode derivation function lives here because passcodes never reach the
server in production - they stay client-side. This function exists solely for:
1. Generating cross-platform test vectors
2. Unit testing the derivation algorithm
3. Integration tests simulating the full client-side flow
"""
from argon2.low_level import hash_secret_raw, Type


def derive_key_from_passcode(
    passcode: str,
    salt: bytes,
    time_cost: int = 4,
    memory_cost: int = 131072,  # 128 MB
    parallelism: int = 2,
) -> bytes:
    """
    Derive a 32-byte key from a passcode using Argon2id.

    NOTE: This function is TEST-ONLY. In production, passcode derivation
    happens exclusively on the client (browser). The server never sees
    the user's passcode.

    Argon2id is memory-hard and resistant to GPU/ASIC attacks.
    The default parameters (t=4, m=128MB, p=2) provide strong protection
    even for low-entropy passcodes like 6 digits.

    Args:
        passcode: User's passcode (6+ digits recommended)
        salt: Random 32-byte salt (must be stored for later derivation)
        time_cost: Number of iterations (default: 4)
        memory_cost: Memory in KB (default: 131072 = 128MB)
        parallelism: Number of threads (default: 2)

    Returns:
        32-byte derived key

    Raises:
        ValueError: If passcode is empty or salt is wrong length

    Security Note:
        With these parameters, even a 6-digit passcode (1M combinations)
        requires significant resources to brute-force offline.
    """
    if not passcode:
        raise ValueError("Passcode cannot be empty")
    if len(salt) != 32:
        raise ValueError("Salt must be 32 bytes")

    return hash_secret_raw(
        secret=passcode.encode('utf-8'),
        salt=salt,
        time_cost=time_cost,
        memory_cost=memory_cost,
        parallelism=parallelism,
        hash_len=32,
        type=Type.ID,  # Argon2id
    )
