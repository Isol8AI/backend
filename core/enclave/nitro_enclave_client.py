"""
Nitro Enclave client for production deployment.

This client implements EnclaveInterface and communicates with the
real Nitro Enclave via vsock. It's used when ENCLAVE_MODE=nitro.
"""

import asyncio
import json
import logging
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import AsyncGenerator, List, Optional

from core.config import settings
from core.crypto import EncryptedPayload
from .enclave_types import (
    EnclaveInterface,
    EnclaveInfo,
    StreamChunk,
    AgentStreamChunk,
    AgentRunResponse,
)

logger = logging.getLogger(__name__)

# vsock constants
AF_VSOCK = 40


class EnclaveConnectionError(Exception):
    """Raised when cannot connect to enclave."""

    pass


class EnclaveTimeoutError(Exception):
    """Raised when enclave request times out."""

    pass


class NitroEnclaveClient(EnclaveInterface):
    """
    Client for communicating with real Nitro Enclave via vsock.

    Implements the same interface as MockEnclave so ChatService
    and routes work unchanged.
    """

    def __init__(self, enclave_cid: int, enclave_port: int = 5000):
        """
        Initialize the Nitro Enclave client.

        Args:
            enclave_cid: The enclave's CID (Context Identifier)
            enclave_port: The vsock port the enclave listens on
        """
        self._cid = enclave_cid
        self._port = enclave_port
        self._enclave_public_key: Optional[bytes] = None
        self._credentials_task: Optional[asyncio.Task] = None
        self._credentials_expiration: Optional[datetime] = None

        logger.info(f"NitroEnclaveClient initializing (CID={enclave_cid}, port={enclave_port})")

        # Fetch enclave's public key
        self._refresh_public_key()

        # Push initial credentials
        self._push_credentials_sync()

        logger.info("NitroEnclaveClient initialized successfully")

    # =========================================================================
    # vsock Communication
    # =========================================================================

    def _send_command(self, command: dict, timeout: float = 120.0) -> dict:
        """Send command to enclave via vsock, return response."""
        sock = socket.socket(AF_VSOCK, socket.SOCK_STREAM)
        sock.settimeout(timeout)

        try:
            sock.connect((self._cid, self._port))
            sock.sendall(json.dumps(command).encode("utf-8"))
            response = sock.recv(1048576)  # 1MB buffer
            return json.loads(response.decode("utf-8"))

        except socket.timeout:
            logger.error(f"Enclave timeout (CID={self._cid})")
            raise EnclaveTimeoutError("Enclave request timed out")

        except ConnectionRefusedError:
            logger.error(f"Enclave connection refused (CID={self._cid})")
            raise EnclaveConnectionError("Enclave not running or not accepting connections")

        except OSError as e:
            logger.error(f"Enclave socket error: {e}")
            raise EnclaveConnectionError(f"Socket error: {e}")

        finally:
            try:
                sock.close()
            except Exception:
                pass

    def _send_command_stream(self, command: dict, timeout: float = 120.0):
        """Send command and yield streaming response events."""
        sock = socket.socket(AF_VSOCK, socket.SOCK_STREAM)
        sock.settimeout(timeout)

        try:
            sock.connect((self._cid, self._port))
            sock.sendall(json.dumps(command).encode("utf-8"))

            # Read streaming JSON events (newline-delimited)
            buffer = b""
            while True:
                try:
                    chunk = sock.recv(65536)
                except socket.timeout:
                    logger.warning("Socket timeout during streaming, continuing...")
                    continue

                if not chunk:
                    break
                buffer += chunk

                # Parse complete JSON objects (newline-delimited)
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if line:
                        try:
                            event = json.loads(line.decode("utf-8"))
                            yield event
                            if event.get("is_final"):
                                return
                        except json.JSONDecodeError as e:
                            logger.warning(f"Invalid JSON in stream: {e}")

        except socket.timeout:
            logger.error("Enclave stream timeout")
            yield {"error": "Stream timeout", "is_final": True}

        except Exception as e:
            logger.error(f"Enclave stream error: {e}")
            yield {"error": str(e), "is_final": True}

        finally:
            try:
                sock.close()
            except Exception:
                pass

    # =========================================================================
    # EnclaveInterface Implementation
    # =========================================================================

    def get_info(self) -> EnclaveInfo:
        """Get enclave's public key and attestation."""
        if self._enclave_public_key is None:
            self._refresh_public_key()

        return EnclaveInfo(
            enclave_public_key=self._enclave_public_key,
            attestation_document=None,  # M6 will add attestation
        )

    def get_transport_public_key(self) -> str:
        """Get enclave's transport public key as hex string."""
        if self._enclave_public_key is None:
            self._refresh_public_key()
        return self._enclave_public_key.hex()

    def _refresh_public_key(self) -> None:
        """Fetch enclave's public key."""
        response = self._send_command({"command": "GET_PUBLIC_KEY"}, timeout=10.0)
        if response.get("status") != "success":
            raise EnclaveConnectionError(f"Failed to get public key: {response}")
        self._enclave_public_key = bytes.fromhex(response["public_key"])
        logger.info(f"Enclave public key: {response['public_key'][:16]}...")

    def decrypt_transport_message(self, payload: EncryptedPayload) -> bytes:
        """Not implemented - decryption happens inside enclave during CHAT_STREAM."""
        raise NotImplementedError(
            "NitroEnclaveClient does not expose decrypt_transport_message. Use process_message_streaming instead."
        )

    def encrypt_for_storage(
        self,
        plaintext: bytes,
        storage_public_key: bytes,
        is_assistant: bool,
    ) -> EncryptedPayload:
        """Not implemented - encryption happens inside enclave during CHAT_STREAM."""
        raise NotImplementedError(
            "NitroEnclaveClient does not expose encrypt_for_storage. Use process_message_streaming instead."
        )

    def encrypt_for_transport(
        self,
        plaintext: bytes,
        recipient_public_key: bytes,
    ) -> EncryptedPayload:
        """Not implemented - encryption happens inside enclave during CHAT_STREAM."""
        raise NotImplementedError(
            "NitroEnclaveClient does not expose encrypt_for_transport. Use process_message_streaming instead."
        )

    async def process_message(
        self,
        encrypted_message: EncryptedPayload,
        encrypted_history: List[EncryptedPayload],
        storage_public_key: bytes,
        transport_public_key: bytes,
        model: str,
    ):
        """Not implemented - use process_message_streaming instead."""
        raise NotImplementedError(
            "NitroEnclaveClient does not support non-streaming. Use process_message_streaming instead."
        )

    async def process_message_stream(
        self,
        encrypted_message: EncryptedPayload,
        encrypted_history: List[EncryptedPayload],
        storage_public_key: bytes,
        transport_public_key: bytes,
        model: str,
    ):
        """Not implemented - use process_message_streaming instead."""
        raise NotImplementedError(
            "NitroEnclaveClient does not support process_message_stream. Use process_message_streaming instead."
        )

    async def process_message_streaming(
        self,
        encrypted_message: EncryptedPayload,
        encrypted_history: List[EncryptedPayload],
        facts_context: Optional[str],
        storage_public_key: bytes,
        client_public_key: bytes,
        session_id: str,
        model: str,
        user_id: str = "",
        org_id: Optional[str] = None,
        user_metadata: Optional[dict] = None,
        org_metadata: Optional[dict] = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Process message through Nitro Enclave with streaming.

        Sends CHAT_STREAM command, yields StreamChunk objects as
        enclave streams back encrypted response chunks.
        """
        # Check if credentials need refresh
        if self._credentials_expiring_soon():
            logger.info("Credentials expiring soon, refreshing...")
            await self._push_credentials_async()

        command = {
            "command": "CHAT_STREAM",
            "encrypted_message": encrypted_message.to_dict(),
            "encrypted_history": [h.to_dict() for h in encrypted_history],
            "storage_public_key": storage_public_key.hex(),
            "client_public_key": client_public_key.hex(),
            "model_id": model,
            "session_id": session_id,
        }

        logger.debug(f"Sending CHAT_STREAM command for session {session_id}")

        # Use asyncio.Queue for proper async waiting (no polling/sleeping)
        event_queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def stream_in_thread():
            """Run sync generator in thread, put events in async queue."""
            try:
                event_num = 0
                for event in self._send_command_stream(command):
                    event_num += 1
                    print(f"[Parent] Thread: received event #{event_num} at {time.time():.3f}", flush=True)
                    # Thread-safe way to put into asyncio.Queue
                    loop.call_soon_threadsafe(event_queue.put_nowait, ("event", event))
                loop.call_soon_threadsafe(event_queue.put_nowait, ("done", None))
                print(f"[Parent] Thread: done, total events={event_num}", flush=True)
            except Exception as e:
                print(f"[Parent] Thread: error {e}", flush=True)
                loop.call_soon_threadsafe(event_queue.put_nowait, ("error", e))

        # Start the sync generator in a thread pool
        executor = ThreadPoolExecutor(max_workers=1)
        loop.run_in_executor(executor, stream_in_thread)  # Fire and forget

        try:
            while True:
                # Properly await the async queue - no polling, no sleeping
                item_type, item = await event_queue.get()

                if item_type == "done":
                    break

                if item_type == "error":
                    if isinstance(item, EnclaveConnectionError):
                        logger.error(f"Enclave connection error: {item}")
                        yield StreamChunk(error="Service temporarily unavailable", is_final=True)
                    elif isinstance(item, EnclaveTimeoutError):
                        logger.error("Enclave timeout during streaming")
                        yield StreamChunk(error="Request timed out", is_final=True)
                    else:
                        logger.exception(f"Unexpected error in enclave streaming: {item}")
                        yield StreamChunk(error="Internal error", is_final=True)
                    break

                event = item
                if event.get("error"):
                    logger.error(f"Enclave error: {event['error']}")
                    yield StreamChunk(error=event["error"], is_final=True)
                    break

                if event.get("encrypted_content"):
                    print(f"[Parent] Async: yielding encrypted_content at {time.time():.3f}", flush=True)
                    yield StreamChunk(encrypted_content=EncryptedPayload.from_dict(event["encrypted_content"]))

                if event.get("is_final"):
                    print(f"[Parent] Async: yielding is_final at {time.time():.3f}", flush=True)
                    yield StreamChunk(
                        stored_user_message=EncryptedPayload.from_dict(event["stored_user_message"]),
                        stored_assistant_message=EncryptedPayload.from_dict(event["stored_assistant_message"]),
                        model_used=event.get("model_used", model),
                        input_tokens=event.get("input_tokens", 0),
                        output_tokens=event.get("output_tokens", 0),
                        is_final=True,
                    )
                    break

        except Exception:
            logger.exception("Unexpected error in enclave streaming")
            yield StreamChunk(error="Internal error", is_final=True)
        finally:
            executor.shutdown(wait=False)

    # =========================================================================
    # Credential Management
    # =========================================================================

    def _get_iam_credentials(self) -> dict:
        """Fetch IAM role credentials from EC2 IMDS."""
        import requests

        # IMDSv2 - get token first
        token_response = requests.put(
            "http://169.254.169.254/latest/api/token",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
            timeout=5,
        )
        token = token_response.text

        # Get IAM role name
        role_response = requests.get(
            "http://169.254.169.254/latest/meta-data/iam/security-credentials/",
            headers={"X-aws-ec2-metadata-token": token},
            timeout=5,
        )
        role_name = role_response.text.strip()

        # Get credentials
        creds_response = requests.get(
            f"http://169.254.169.254/latest/meta-data/iam/security-credentials/{role_name}",
            headers={"X-aws-ec2-metadata-token": token},
            timeout=5,
        )
        creds = creds_response.json()

        return {
            "access_key_id": creds["AccessKeyId"],
            "secret_access_key": creds["SecretAccessKey"],
            "session_token": creds["Token"],
            "expiration": creds["Expiration"],
        }

    def _push_credentials_sync(self) -> None:
        """Push credentials to enclave (sync version)."""
        logger.info("Pushing credentials to enclave...")
        creds = self._get_iam_credentials()

        response = self._send_command(
            {
                "command": "SET_CREDENTIALS",
                "credentials": creds,
            },
            timeout=10.0,
        )

        if response.get("status") != "success":
            raise RuntimeError(f"Failed to set enclave credentials: {response}")

        # Parse expiration time
        exp_str = creds["expiration"]
        # Handle both formats: with and without timezone
        if exp_str.endswith("Z"):
            exp_str = exp_str[:-1] + "+00:00"
        self._credentials_expiration = datetime.fromisoformat(exp_str)

        logger.info(f"Credentials pushed, expire at {self._credentials_expiration}")

    async def _push_credentials_async(self) -> None:
        """Push credentials to enclave (async version)."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._push_credentials_sync)

    def _credentials_expiring_soon(self) -> bool:
        """Check if credentials expire within 5 minutes."""
        if self._credentials_expiration is None:
            return True
        # Use UTC for comparison
        now = datetime.utcnow()
        expiry = self._credentials_expiration.replace(tzinfo=None)
        return now + timedelta(minutes=5) > expiry

    async def start_credential_refresh(self) -> None:
        """Start background task to refresh enclave credentials."""
        if self._credentials_task is None:
            self._credentials_task = asyncio.create_task(self._credential_refresh_loop())
            logger.info("Started credential refresh task")

    async def stop_credential_refresh(self) -> None:
        """Stop credential refresh task."""
        if self._credentials_task:
            self._credentials_task.cancel()
            try:
                await self._credentials_task
            except asyncio.CancelledError:
                pass
            self._credentials_task = None
            logger.info("Stopped credential refresh task")

    async def _credential_refresh_loop(self) -> None:
        """Refresh credentials periodically."""
        while True:
            try:
                await asyncio.sleep(settings.ENCLAVE_CREDENTIAL_REFRESH_SECONDS)
                await self._push_credentials_async()
                logger.info("Refreshed enclave credentials")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Credential refresh failed: {e}")
                # Retry sooner on failure
                await asyncio.sleep(60)

    # =========================================================================
    # Health Check
    # =========================================================================

    def health_check(self) -> dict:
        """Check enclave health."""
        try:
            response = self._send_command({"command": "HEALTH"}, timeout=5.0)
            return {
                "status": "healthy",
                "mode": "nitro",
                "enclave_cid": self._cid,
                "has_credentials": response.get("has_credentials", False),
                "region": response.get("region"),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "mode": "nitro",
                "enclave_cid": self._cid,
                "error": str(e),
            }

    # =========================================================================
    # Agent Execution
    # =========================================================================

    async def run_agent(
        self,
        encrypted_message: EncryptedPayload,
        encrypted_state: Optional[EncryptedPayload],
        user_public_key: bytes,
        agent_name: str,
        model: str,
        encryption_mode: str = "zero_trust",
        kms_envelope: Optional[dict] = None,
    ) -> AgentRunResponse:
        """
        Run an OpenClaw agent inside the Nitro Enclave (non-streaming).

        This sends the RUN_AGENT command to the real enclave via vsock.
        The enclave handles:
        1. Decrypting the message and state
        2. Unpacking state to tmpfs
        3. Running OpenClaw CLI
        4. Re-encrypting the response and updated state

        Args:
            encrypted_message: User's message encrypted to enclave
            encrypted_state: Existing agent state tarball (for zero_trust: re-encrypted by client)
            user_public_key: User's public key for response and state encryption
            agent_name: Name of the agent to run
            model: LLM model to use
            encryption_mode: "zero_trust" (default) or "background"
            kms_envelope: KMS envelope dict for background mode (optional)

        Returns:
            AgentRunResponse with encrypted response and state
        """
        # Check if credentials need refresh
        if self._credentials_expiring_soon():
            logger.info("Credentials expiring soon, refreshing...")
            await self._push_credentials_async()

        # For background mode, use kms_envelope instead of encrypted_state
        encrypted_state_dict = None
        if encryption_mode == "background" and kms_envelope:
            # Serialize bytes to hex strings for JSON transmission
            encrypted_state_dict = {
                "encrypted_dek": kms_envelope["encrypted_dek"].hex(),
                "iv": kms_envelope["iv"].hex(),
                "ciphertext": kms_envelope["ciphertext"].hex(),
                "auth_tag": kms_envelope["auth_tag"].hex(),
            }
        elif encrypted_state:
            encrypted_state_dict = encrypted_state.to_dict()

        command = {
            "command": "RUN_AGENT",
            "encrypted_message": encrypted_message.to_dict(),
            "encrypted_state": encrypted_state_dict,
            "user_public_key": user_public_key.hex(),
            "agent_name": agent_name,
            "model": model,
            "encryption_mode": encryption_mode,
        }

        logger.info(f"Sending RUN_AGENT command for agent {agent_name}")

        try:
            # Run in thread pool since vsock is blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._send_command(command, timeout=settings.ENCLAVE_INFERENCE_TIMEOUT),
            )

            if response.get("status") != "success":
                error_msg = response.get("error", "Unknown enclave error")
                logger.error(f"Enclave RUN_AGENT failed: {error_msg}")
                return AgentRunResponse(
                    success=False,
                    error=error_msg,
                )

            # Parse encrypted payloads from response
            encrypted_response = None
            if response.get("encrypted_response"):
                encrypted_response = EncryptedPayload.from_dict(response["encrypted_response"])

            # Handle encrypted_state based on encryption_mode
            encrypted_state_result = None
            kms_envelope_result = None

            if response.get("encrypted_state"):
                if encryption_mode == "background":
                    # Background mode: encrypted_state is a raw KMS envelope dict
                    kms_envelope_result = response["encrypted_state"]
                else:
                    # Zero trust mode: encrypted_state is an EncryptedPayload
                    encrypted_state_result = EncryptedPayload.from_dict(response["encrypted_state"])

            encrypted_dek = response.get("encrypted_dek")  # Deprecated, kms_envelope contains it

            return AgentRunResponse(
                success=True,
                encrypted_response=encrypted_response,
                encrypted_state=encrypted_state_result,
                encrypted_dek=encrypted_dek,
                kms_envelope=kms_envelope_result,
            )

        except EnclaveConnectionError as e:
            logger.error(f"Enclave connection error: {e}")
            return AgentRunResponse(
                success=False,
                error="Service temporarily unavailable",
            )

        except EnclaveTimeoutError:
            logger.error("Enclave timeout during agent execution")
            return AgentRunResponse(
                success=False,
                error="Request timed out",
            )

    async def agent_chat_streaming(
        self,
        encrypted_message: EncryptedPayload,
        encrypted_state: Optional[EncryptedPayload],
        client_public_key: bytes,
        user_public_key: bytes,
        agent_name: str,
        encrypted_soul_content: Optional[EncryptedPayload] = None,
        encryption_mode: str = "zero_trust",
    ) -> AsyncGenerator[AgentStreamChunk, None]:
        """
        Process agent chat through Nitro Enclave with streaming.

        Sends AGENT_CHAT_STREAM command, yields AgentStreamChunk objects
        as enclave streams back encrypted response chunks.

        Args:
            encrypted_message: User message encrypted to enclave transport key
            encrypted_state: Agent state (for zero_trust: re-encrypted by client to enclave key)
            client_public_key: Ephemeral transport key for response chunk encryption
            user_public_key: User's long-term public key for state encryption
            agent_name: Name of the agent
            encrypted_soul_content: Optional SOUL.md content for new agents
            encryption_mode: "zero_trust" (default) or "background"
        """
        # Check if credentials need refresh
        if self._credentials_expiring_soon():
            logger.info("Credentials expiring soon, refreshing...")
            await self._push_credentials_async()

        command = {
            "command": "AGENT_CHAT_STREAM",
            "encrypted_message": encrypted_message.to_dict(),
            "encrypted_state": encrypted_state.to_dict() if encrypted_state else None,
            "client_public_key": client_public_key.hex(),
            "user_public_key": user_public_key.hex(),
            "agent_name": agent_name,
            "encrypted_soul_content": encrypted_soul_content.to_dict() if encrypted_soul_content else None,
            "encryption_mode": encryption_mode,
        }

        logger.debug(f"Sending AGENT_CHAT_STREAM command for agent {agent_name}")

        # Use asyncio.Queue for proper async waiting
        event_queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def stream_in_thread():
            """Run sync generator in thread, put events in async queue."""
            try:
                for event in self._send_command_stream(command):
                    loop.call_soon_threadsafe(event_queue.put_nowait, ("event", event))
                loop.call_soon_threadsafe(event_queue.put_nowait, ("done", None))
            except Exception as e:
                loop.call_soon_threadsafe(event_queue.put_nowait, ("error", e))

        executor = ThreadPoolExecutor(max_workers=1)
        loop.run_in_executor(executor, stream_in_thread)

        try:
            while True:
                item_type, item = await event_queue.get()

                if item_type == "done":
                    break

                if item_type == "error":
                    yield AgentStreamChunk(error="Internal error", is_final=True)
                    break

                event = item

                # Log diagnostic info from enclave (not yielded to caller)
                if event.get("diagnostic"):
                    diag = event["diagnostic"]
                    print(
                        f"ENCLAVE_DIAG: chunk_count={diag.get('chunk_count')}, event_types={diag.get('event_types')}, block_fallback={diag.get('has_block_fallback')}, block_len={diag.get('block_text_len')}",
                        flush=True,
                    )
                    continue

                if event.get("error"):
                    logger.error(f"Enclave agent error: {event['error']}")
                    yield AgentStreamChunk(error=event["error"], is_final=True)
                    break

                if event.get("encrypted_content"):
                    yield AgentStreamChunk(encrypted_content=EncryptedPayload.from_dict(event["encrypted_content"]))

                if event.get("is_final"):
                    encrypted_state_result = None
                    if event.get("encrypted_state"):
                        encrypted_state_result = EncryptedPayload.from_dict(event["encrypted_state"])
                        # TRACE_CRYPTO: Log what we received from vsock
                        import hashlib as _hl

                        _rd = event["encrypted_state"]
                        print(f"TRACE_CRYPTO:VSOCK_RECV eph_pub={_rd['ephemeral_public_key']}", flush=True)
                        print(f"TRACE_CRYPTO:VSOCK_RECV iv={_rd['iv']}", flush=True)
                        print(
                            f"TRACE_CRYPTO:VSOCK_RECV ct_sha256={_hl.sha256(bytes.fromhex(_rd['ciphertext'])).hexdigest()} ct_hex_len={len(_rd['ciphertext'])}",
                            flush=True,
                        )
                        print(f"TRACE_CRYPTO:VSOCK_RECV auth_tag={_rd['auth_tag']}", flush=True)
                        print(f"TRACE_CRYPTO:VSOCK_RECV hkdf_salt={_rd['hkdf_salt']}", flush=True)

                    encrypted_dek = event.get("encrypted_dek")  # None for zero_trust, KMS-encrypted DEK for background

                    yield AgentStreamChunk(
                        encrypted_state=encrypted_state_result,
                        encrypted_dek=encrypted_dek,
                        is_final=True,
                        input_tokens=event.get("input_tokens", 0),
                        output_tokens=event.get("output_tokens", 0),
                    )
                    break

        except Exception:
            logger.exception("Unexpected error in agent streaming")
            yield AgentStreamChunk(error="Internal error", is_final=True)
        finally:
            executor.shutdown(wait=False)
