"""
Agent handler for enclave integration.

This module provides the interface for agent operations.
The actual agent execution (decryption, OpenClaw CLI, re-encryption)
happens inside the enclave. This handler delegates to the enclave's
run_agent method.

In production (Nitro Enclave): Agent runs in isolated enclave via vsock
In development (MockEnclave): Agent runs in-process with fallback response
"""

import logging
from dataclasses import dataclass
from typing import AsyncGenerator, Optional

from core.crypto import EncryptedPayload

logger = logging.getLogger(__name__)


@dataclass
class AgentStreamRequest:
    """Request to process a streaming agent chat message."""

    user_id: str
    agent_name: str
    encrypted_message: EncryptedPayload
    encrypted_state: Optional[EncryptedPayload]
    client_public_key: bytes
    encrypted_soul_content: Optional[EncryptedPayload] = None
    encryption_mode: str = "zero_trust"  # "zero_trust" or "background"


@dataclass
class AgentMessageRequest:
    """Request to process a message through an agent."""

    user_id: str
    agent_name: str
    encrypted_message: EncryptedPayload
    encrypted_state: Optional[EncryptedPayload]  # None for new users
    user_public_key: bytes
    model: str
    encryption_mode: str = "zero_trust"  # "zero_trust" or "background"


@dataclass
class AgentMessageResponse:
    """Response from processing an agent message."""

    success: bool
    encrypted_response: Optional[EncryptedPayload] = None
    encrypted_state: Optional[EncryptedPayload] = None  # Updated state for storage
    error: str = ""


class AgentHandler:
    """
    Handles agent message processing by delegating to the enclave.

    The enclave is responsible for:
    1. Decrypting messages and state
    2. Managing tmpfs directories (inside enclave)
    3. Running OpenClaw CLI
    4. Re-encrypting state and responses

    This handler simply forwards requests to the enclave's run_agent
    method, which handles all the secure operations.
    """

    def __init__(self, enclave=None):
        """
        Initialize the handler.

        Args:
            enclave: Enclave instance (MockEnclave or NitroEnclaveClient)
        """
        self.enclave = enclave

    async def process_message(
        self,
        request: AgentMessageRequest,
    ) -> AgentMessageResponse:
        """
        Process a message through an agent.

        Delegates to enclave.run_agent() which handles all secure operations:
        1. Decrypting the message and state
        2. Running OpenClaw CLI
        3. Re-encrypting response and updated state

        Args:
            request: Agent message request with encrypted data

        Returns:
            AgentMessageResponse with encrypted response and state
        """
        if self.enclave is None:
            return AgentMessageResponse(
                success=False,
                error="Enclave not configured",
            )

        try:
            logger.info(f"Processing agent message for user {request.user_id}, agent {request.agent_name}")

            # Delegate to enclave's run_agent method
            # The enclave handles decryption, agent execution, and re-encryption
            result = await self.enclave.run_agent(
                encrypted_message=request.encrypted_message,
                encrypted_state=request.encrypted_state,
                user_public_key=request.user_public_key,
                agent_name=request.agent_name,
                model=request.model,
                encryption_mode=request.encryption_mode,
            )

            return AgentMessageResponse(
                success=result.success,
                encrypted_response=result.encrypted_response,
                encrypted_state=result.encrypted_state,
                error=result.error,
            )

        except Exception as e:
            logger.exception(f"Error processing agent message: {e}")
            return AgentMessageResponse(
                success=False,
                error=str(e),
            )

    async def process_message_streaming(
        self,
        request: AgentStreamRequest,
    ) -> AsyncGenerator:
        """
        Process a streaming agent chat message.

        Delegates to enclave's agent_chat_streaming method which handles
        all secure operations: decryption, Bedrock streaming, state update,
        and re-encryption.

        Args:
            request: Agent stream request with encrypted data

        Yields:
            AgentStreamChunk objects with encrypted content or final state
        """
        from .enclave_types import AgentStreamChunk

        if self.enclave is None:
            yield AgentStreamChunk(error="Enclave not configured", is_final=True)
            return

        try:
            logger.info(f"Processing streaming agent message for user {request.user_id}, agent {request.agent_name}")

            async for chunk in self.enclave.agent_chat_streaming(
                encrypted_message=request.encrypted_message,
                encrypted_state=request.encrypted_state,
                client_public_key=request.client_public_key,
                agent_name=request.agent_name,
                encrypted_soul_content=request.encrypted_soul_content,
                encryption_mode=request.encryption_mode,
            ):
                yield chunk

        except Exception as e:
            logger.exception(f"Error in agent streaming: {e}")
            yield AgentStreamChunk(error=str(e), is_final=True)
