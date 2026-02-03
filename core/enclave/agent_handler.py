"""
Agent handler for enclave integration.

This module bridges the AgentRunner with the enclave's
cryptographic operations, handling:
1. Decrypting incoming messages
2. Decrypting/extracting agent state
3. Running OpenClaw via AgentRunner
4. Packing and encrypting updated state
5. Encrypting responses for transport
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from core.crypto import EncryptedPayload
from core.enclave.agent_runner import AgentRunner, AgentConfig

logger = logging.getLogger(__name__)


@dataclass
class AgentMessageRequest:
    """Request to process a message through an agent."""

    user_id: str
    agent_name: str
    encrypted_message: EncryptedPayload
    encrypted_state: Optional[EncryptedPayload]  # None for new users
    user_public_key: bytes
    model: str


@dataclass
class AgentMessageResponse:
    """Response from processing an agent message."""

    success: bool
    encrypted_response: Optional[EncryptedPayload] = None
    encrypted_state: Optional[EncryptedPayload] = None  # Updated state for storage
    error: str = ""


class AgentHandler:
    """
    Handles agent message processing in the enclave.

    This class orchestrates:
    1. Decrypting messages and state
    2. Managing tmpfs directories
    3. Running OpenClaw via AgentRunner
    4. Re-encrypting state and responses
    """

    def __init__(
        self,
        runner: Optional[AgentRunner] = None,
        enclave=None,
    ):
        """
        Initialize the handler.

        Args:
            runner: AgentRunner instance (created if not provided)
            enclave: Enclave instance for crypto operations
        """
        self.runner = runner or AgentRunner()
        self.enclave = enclave

    async def process_message(
        self,
        request: AgentMessageRequest,
    ) -> AgentMessageResponse:
        """
        Process a message through an agent.

        Flow:
        1. Get/create tmpfs directory for user
        2. If existing state: decrypt and extract to tmpfs
        3. If new user: create fresh agent directory
        4. Decrypt the message
        5. Run OpenClaw CLI via AgentRunner
        6. Pack tmpfs into tarball
        7. Encrypt tarball for storage (to enclave's key)
        8. Encrypt response for transport (to user's key)
        9. Cleanup tmpfs
        """
        tmpfs_path = Path(self.runner.get_user_tmpfs_path(request.user_id))

        try:
            # 1. Prepare tmpfs directory
            if request.encrypted_state:
                # Existing user: decrypt and extract state
                state_bytes = self.enclave.decrypt_transport_message(
                    request.encrypted_state
                )
                self.runner.unpack_tarball(state_bytes, tmpfs_path)
                logger.info(f"Extracted existing state for user {request.user_id}")
            else:
                # New user: create fresh agent
                config = AgentConfig(
                    agent_name=request.agent_name,
                    model=request.model,
                )
                self.runner.create_fresh_agent(tmpfs_path, config)
                logger.info(f"Created fresh agent for user {request.user_id}")

            # 2. Decrypt the user message
            message_bytes = self.enclave.decrypt_transport_message(
                request.encrypted_message
            )
            message = message_bytes.decode("utf-8")
            logger.info(f"Processing message for agent {request.agent_name}")

            # 3. Run OpenClaw CLI via AgentRunner
            result = self.runner.run_agent(
                agent_dir=tmpfs_path,
                message=message,
                agent_name=request.agent_name,
                model=request.model,
            )

            if not result.success:
                return AgentMessageResponse(
                    success=False,
                    error=result.error,
                )

            # 4. Pack updated state
            tarball_bytes = self.runner.pack_directory(tmpfs_path)

            # 5. Encrypt state for storage (to enclave's key for future decryption)
            # Note: We encrypt to enclave's key so we can decrypt it on next request
            encrypted_state = self.enclave.encrypt_for_storage(
                tarball_bytes,
                self.enclave.get_info().enclave_public_key,
                is_assistant=False,  # Using user context for state
            )

            # 6. Encrypt response for transport (to user's key)
            encrypted_response = self.enclave.encrypt_for_transport(
                result.response.encode("utf-8"),
                request.user_public_key,
            )

            return AgentMessageResponse(
                success=True,
                encrypted_response=encrypted_response,
                encrypted_state=encrypted_state,
            )

        except Exception as e:
            logger.exception(f"Error processing agent message: {e}")
            return AgentMessageResponse(
                success=False,
                error=str(e),
            )

        finally:
            # Always cleanup tmpfs
            self.runner.cleanup_directory(tmpfs_path)
