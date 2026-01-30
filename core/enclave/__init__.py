"""
Enclave package for secure message processing.

This package provides enclave implementations:
- MockEnclave: In-process for development (ENCLAVE_MODE=mock)
- NitroEnclaveClient: Real Nitro Enclave via vsock (ENCLAVE_MODE=nitro)
"""

import logging
import subprocess
import json
from enum import Enum
from typing import Union

logger = logging.getLogger(__name__)


class EncryptionContext(str, Enum):
    """
    HKDF context strings for domain separation.

    These context strings MUST match between encryption and decryption.
    They ensure that keys derived for different purposes cannot be
    confused or misused.
    """

    # Transport contexts (ephemeral per-request)
    CLIENT_TO_ENCLAVE = "client-to-enclave-transport"
    ENCLAVE_TO_CLIENT = "enclave-to-client-transport"

    # Storage contexts (long-term storage encryption)
    USER_MESSAGE_STORAGE = "user-message-storage"
    ASSISTANT_MESSAGE_STORAGE = "assistant-message-storage"

    # Key distribution contexts
    ORG_KEY_DISTRIBUTION = "org-key-distribution"
    RECOVERY_KEY_ENCRYPTION = "recovery-key-encryption"


# Import types from mock_enclave (used by both implementations)
from .mock_enclave import (
    MockEnclave,
    EnclaveInterface,
    ProcessedMessage,
    StreamChunk,
    EnclaveInfo,
    DecryptedMessage,
)

# Singleton instance
_enclave_instance: Union[EnclaveInterface, None] = None


def _discover_enclave_cid() -> int:
    """Discover running enclave's CID using nitro-cli."""
    try:
        result = subprocess.run(
            ["nitro-cli", "describe-enclaves"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        enclaves = json.loads(result.stdout)
        if enclaves and len(enclaves) > 0:
            cid = enclaves[0].get("EnclaveCID")
            if cid:
                logger.info(f"Discovered enclave CID: {cid}")
                return cid
    except FileNotFoundError:
        logger.warning("nitro-cli not found - not running on Nitro-enabled instance")
    except subprocess.TimeoutExpired:
        logger.warning("nitro-cli timed out")
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse nitro-cli output: {e}")
    except Exception as e:
        logger.warning(f"Could not discover enclave CID: {e}")

    raise RuntimeError(
        "No running enclave found. "
        "Start enclave with: sudo nitro-cli run-enclave --eif-path /path/to/enclave.eif --cpu-count 2 --memory 512"
    )


def get_enclave() -> EnclaveInterface:
    """
    Get the enclave instance based on ENCLAVE_MODE config.

    - ENCLAVE_MODE=mock: Returns MockEnclave (in-process, for dev)
    - ENCLAVE_MODE=nitro: Returns NitroEnclaveClient (vsock to real enclave)

    Returns:
        EnclaveInterface implementation
    """
    global _enclave_instance

    if _enclave_instance is None:
        from core.config import settings

        if settings.ENCLAVE_MODE == "nitro":
            from .nitro_enclave_client import NitroEnclaveClient

            # Discover enclave CID if not configured
            cid = settings.ENCLAVE_CID
            if cid == 0:
                cid = _discover_enclave_cid()

            _enclave_instance = NitroEnclaveClient(
                enclave_cid=cid,
                enclave_port=settings.ENCLAVE_PORT,
            )
            logger.info(f"Using NitroEnclaveClient (CID={cid}, port={settings.ENCLAVE_PORT})")
        else:
            _enclave_instance = MockEnclave(
                aws_region=settings.AWS_REGION,
                inference_timeout=settings.ENCLAVE_INFERENCE_TIMEOUT,
            )
            logger.info("Using MockEnclave (development mode)")

    return _enclave_instance


def reset_enclave() -> None:
    """
    Reset the enclave singleton (for testing only).

    This forces a new instance to be created on next get_enclave() call.
    """
    global _enclave_instance
    _enclave_instance = None


async def startup_enclave() -> None:
    """
    Initialize enclave on application startup.

    For NitroEnclaveClient, starts the credential refresh background task.
    """
    from core.config import settings

    enclave = get_enclave()

    if settings.ENCLAVE_MODE == "nitro":
        from .nitro_enclave_client import NitroEnclaveClient

        if isinstance(enclave, NitroEnclaveClient):
            await enclave.start_credential_refresh()
            logger.info("Started enclave credential refresh task")


async def shutdown_enclave() -> None:
    """
    Cleanup enclave on application shutdown.

    For NitroEnclaveClient, stops the credential refresh background task.
    """
    global _enclave_instance

    if _enclave_instance is not None:
        from .nitro_enclave_client import NitroEnclaveClient

        if isinstance(_enclave_instance, NitroEnclaveClient):
            await _enclave_instance.stop_credential_refresh()
            logger.info("Stopped enclave credential refresh task")

    _enclave_instance = None


__all__ = [
    "EncryptionContext",
    "MockEnclave",
    "EnclaveInterface",
    "ProcessedMessage",
    "StreamChunk",
    "EnclaveInfo",
    "DecryptedMessage",
    "get_enclave",
    "reset_enclave",
    "startup_enclave",
    "shutdown_enclave",
]
