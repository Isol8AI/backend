# M5: Full Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace MockEnclave with real Nitro Enclave while keeping FastAPI backend code unchanged.

**Architecture:** NitroEnclaveClient implements EnclaveInterface and communicates with the Nitro enclave via vsock. The `get_enclave()` factory returns either MockEnclave or NitroEnclaveClient based on ENCLAVE_MODE config. Real streaming is implemented via AWS event stream parsing in the enclave.

**Tech Stack:** Python 3.8+, FastAPI, vsock (AF_VSOCK=40), botocore, AWS Bedrock Converse API

---

## Phase 1: Enclave Streaming

### Task 1.1: Add request_stream() to VsockHttpClient

**Files:**
- Modify: `enclave/vsock_http_client.py`

**Step 1: Add helper method _read_exact()**z    
    

Add after line 178 (after `_parse_response` method):

```python
def _read_exact(self, sock: ssl.SSLSocket, num_bytes: int) -> bytes:
    """Read exactly num_bytes from socket."""
    data = b""
    while len(data) < num_bytes:
        chunk = sock.recv(num_bytes - len(data))
        if not chunk:
            break
        data += chunk
    return data
```

**Step 2: Run existing tests to verify no regression**

Run: `cd /Users/prasiddhaparthsarthy/Desktop/freebird/backend && python -c "from enclave.vsock_http_client import VsockHttpClient; print('Import OK')"`
Expected: `Import OK`

**Step 3: Add request_stream() method**

Add after `_read_exact` method:

```python
def request_stream(
    self,
    method: str,
    url: str,
    headers: Optional[Dict[str, str]] = None,
    body: Optional[bytes] = None,
) -> Generator[bytes, None, None]:
    """
    Make streaming HTTP request. Yields raw AWS event stream messages.

    AWS event stream format per message:
    - 4 bytes: total byte length (big-endian)
    - 4 bytes: headers byte length (big-endian)
    - 4 bytes: prelude CRC
    - Headers (variable)
    - Payload (variable)
    - 4 bytes: message CRC
    """
    headers = headers or {}

    # Parse URL
    parsed = urlparse(url)
    host = parsed.netloc
    path = parsed.path or "/"
    if parsed.query:
        path += f"?{parsed.query}"

    port = 443 if parsed.scheme == "https" else 80
    if ":" in host:
        host, port = host.rsplit(":", 1)
        port = int(port)

    use_tls = parsed.scheme == "https"

    # Connect through proxy
    sock = self._create_vsock()

    try:
        if not self._send_connect(sock, host, port):
            raise ConnectionError("Proxy refused connection")

        if use_tls:
            sock = self._wrap_tls(sock, host)

        # Build and send request
        request = self._build_request(method, path, host, headers, body)
        sock.sendall(request)

        # Read HTTP response headers
        header_data = b""
        while b"\r\n\r\n" not in header_data:
            chunk = sock.recv(4096)
            if not chunk:
                raise ConnectionError("Connection closed while reading headers")
            header_data += chunk

        header_part, body_start = header_data.split(b"\r\n\r\n", 1)

        # Check status
        status_line = header_part.split(b"\r\n")[0].decode("utf-8")
        if "200" not in status_line:
            raise Exception(f"HTTP error: {status_line}")

        # Buffer any body data we already received
        buffer = body_start

        # Stream AWS event stream messages
        while True:
            # Need at least 4 bytes for message length
            while len(buffer) < 4:
                chunk = sock.recv(65536)
                if not chunk:
                    return  # End of stream
                buffer += chunk

            # Parse message length (first 4 bytes, big-endian)
            msg_len = int.from_bytes(buffer[:4], "big")

            if msg_len == 0:
                return  # End marker

            # Read full message
            while len(buffer) < msg_len:
                chunk = sock.recv(65536)
                if not chunk:
                    return
                buffer += chunk

            # Yield complete message
            message = buffer[:msg_len]
            buffer = buffer[msg_len:]
            yield message

    finally:
        sock.close()
```

**Step 4: Add Generator import**

At top of file, modify import line:

```python
from typing import Dict, Optional, Generator
```

**Step 5: Verify syntax**

Run: `cd /Users/prasiddhaparthsarthy/Desktop/freebird/backend && python -c "from enclave.vsock_http_client import VsockHttpClient; print('Import OK')"`
Expected: `Import OK`

**Step 6: Run linting**

Run: `cd /Users/prasiddhaparthsarthy/Desktop/freebird/backend && ruff check enclave/vsock_http_client.py && ruff format enclave/vsock_http_client.py`
Expected: No errors

---

### Task 1.2: Add Event Stream Parser to BedrockClient

**Files:**
- Modify: `enclave/bedrock_client.py`

**Step 1: Add event stream parsing methods**

Add after `_parse_converse_response` method (around line 199):

```python
def _parse_event_stream_message(self, data: bytes) -> dict:
    """
    Parse single AWS event stream message into event dict.

    Format:
    - Bytes 0-3: total length (big-endian)
    - Bytes 4-7: headers length (big-endian)
    - Bytes 8-11: prelude CRC
    - Bytes 12 to 12+headers_length: headers
    - Remaining (minus 4 byte CRC): payload
    """
    if len(data) < 16:
        return {}

    headers_length = int.from_bytes(data[4:8], "big")
    headers_end = 12 + headers_length

    # Extract payload (skip 4-byte message CRC at end)
    payload = data[headers_end:-4]

    if payload:
        try:
            return json.loads(payload.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}
    return {}
```

**Step 2: Update converse_stream() for real streaming**

Replace the entire `converse_stream` method (lines 287-367) with:

```python
def converse_stream(
    self,
    model_id: str,
    messages: List[Dict[str, Any]],
    system: Optional[List[Dict[str, str]]] = None,
    inference_config: Optional[Dict[str, Any]] = None,
) -> Generator[Dict[str, Any], None, None]:
    """
    Call Bedrock Converse API with real streaming.

    Yields event dictionaries:
    - {"type": "content", "text": "chunk"}
    - {"type": "stop", "reason": "end_turn"}
    - {"type": "metadata", "usage": {"inputTokens": N, "outputTokens": N}}
    """
    # Build request body
    body = {"modelId": model_id, "messages": messages}
    if system:
        body["system"] = system
    if inference_config:
        body["inferenceConfig"] = inference_config

    body_bytes = json.dumps(body).encode("utf-8")

    # Streaming endpoint
    path = f"/model/{model_id}/converse-stream"
    url = f"https://{self.host}{path}"

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/vnd.amazon.eventstream",
    }

    # Sign request
    signed_headers = self._sign_request("POST", url, headers, body_bytes)

    # Stream response
    try:
        for raw_message in self.http_client.request_stream("POST", url, signed_headers, body_bytes):
            event = self._parse_event_stream_message(raw_message)

            if not event:
                continue

            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"].get("delta", {})
                text = delta.get("text", "")
                if text:
                    yield {"type": "content", "text": text}

            elif "messageStop" in event:
                reason = event["messageStop"].get("stopReason", "end_turn")
                yield {"type": "stop", "reason": reason}

            elif "metadata" in event:
                usage = event["metadata"].get("usage", {})
                yield {"type": "metadata", "usage": usage}

    except Exception as e:
        yield {"type": "error", "message": str(e)}
```

**Step 3: Verify syntax**

Run: `cd /Users/prasiddhaparthsarthy/Desktop/freebird/backend && python -c "from enclave.bedrock_client import BedrockClient; print('Import OK')"`
Expected: `Import OK`

**Step 4: Run linting**

Run: `cd /Users/prasiddhaparthsarthy/Desktop/freebird/backend && ruff check enclave/bedrock_client.py && ruff format enclave/bedrock_client.py`
Expected: No errors

---

### Task 1.3: Add CHAT_STREAM Command to Enclave Server

**Files:**
- Modify: `enclave/bedrock_server.py`

**Step 1: Add _send_event helper method**

Add after `handle_run_tests` method (around line 297):

```python
def _send_event(self, conn: socket.socket, event: dict) -> None:
    """Send newline-delimited JSON event."""
    conn.sendall(json.dumps(event).encode("utf-8") + b"\n")

def _decrypt_history(self, encrypted_history: list) -> list:
    """Decrypt conversation history from EncryptedPayload dicts."""
    history = []
    for i, payload_dict in enumerate(encrypted_history):
        is_assistant = i % 2 == 1
        payload = EncryptedPayload.from_dict(payload_dict)
        plaintext = decrypt_with_private_key(
            self.keypair.private_key,
            payload,
            "client-to-enclave-transport",
        )
        history.append(ConverseTurn(
            role="assistant" if is_assistant else "user",
            content=plaintext.decode("utf-8"),
        ))
    return history
```

**Step 2: Add handle_chat_stream method**

Add after `_decrypt_history`:

```python
def handle_chat_stream(self, data: dict, conn: socket.socket) -> None:
    """
    Process encrypted chat with streaming response.

    Streams newline-delimited JSON events:
    - {"encrypted_content": {...}}  - Encrypted chunk for client
    - {"is_final": true, ...}       - Final event with stored messages
    - {"error": "...", "is_final": true} - Error event
    """
    try:
        # Check credentials
        if not self.bedrock.has_credentials():
            self._send_event(conn, {"error": "No AWS credentials", "is_final": True})
            return

        # Extract parameters
        encrypted_message = EncryptedPayload.from_dict(data["encrypted_message"])
        encrypted_history = data.get("encrypted_history", [])
        storage_public_key = hex_to_bytes(data["storage_public_key"])
        client_public_key = hex_to_bytes(data["client_public_key"])
        model_id = data["model_id"]

        print(f"[Enclave] CHAT_STREAM: model={model_id}", flush=True)

        # Decrypt user message
        user_plaintext = decrypt_with_private_key(
            self.keypair.private_key,
            encrypted_message,
            "client-to-enclave-transport",
        )
        user_content = user_plaintext.decode("utf-8")
        print(f"[Enclave] User message: {user_content[:50]}...", flush=True)

        # Decrypt history
        history = self._decrypt_history(encrypted_history)
        print(f"[Enclave] History: {len(history)} messages", flush=True)

        # Build messages for Bedrock
        messages = build_converse_messages(history, user_content)
        system = [{"text": "You are a helpful AI assistant."}]
        inference_config = {"maxTokens": 4096, "temperature": 0.7}

        # Stream from Bedrock
        full_response = ""
        input_tokens = 0
        output_tokens = 0
        chunk_count = 0

        print("[Enclave] Starting Bedrock stream...", flush=True)

        for event in self.bedrock.converse_stream(model_id, messages, system, inference_config):
            if event["type"] == "content":
                chunk_text = event["text"]
                full_response += chunk_text
                chunk_count += 1

                # Encrypt chunk for transport to client
                encrypted_chunk = encrypt_to_public_key(
                    client_public_key,
                    chunk_text.encode("utf-8"),
                    "enclave-to-client-transport",
                )
                self._send_event(conn, {"encrypted_content": encrypted_chunk.to_dict()})

            elif event["type"] == "metadata":
                input_tokens = event["usage"].get("inputTokens", 0)
                output_tokens = event["usage"].get("outputTokens", 0)

            elif event["type"] == "error":
                self._send_event(conn, {"error": event["message"], "is_final": True})
                return

        print(f"[Enclave] Stream complete: {chunk_count} chunks, {len(full_response)} chars", flush=True)

        # Encrypt final messages for storage
        stored_user = encrypt_to_public_key(
            storage_public_key,
            user_content.encode("utf-8"),
            "user-message-storage",
        )
        stored_assistant = encrypt_to_public_key(
            storage_public_key,
            full_response.encode("utf-8"),
            "assistant-message-storage",
        )

        # Send final event
        self._send_event(conn, {
            "is_final": True,
            "stored_user_message": stored_user.to_dict(),
            "stored_assistant_message": stored_assistant.to_dict(),
            "model_used": model_id,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        })

        print("[Enclave] CHAT_STREAM complete", flush=True)

    except KeyError as e:
        print(f"[Enclave] CHAT_STREAM missing field: {e}", flush=True)
        self._send_event(conn, {"error": f"Missing field: {e}", "is_final": True})

    except Exception as e:
        print(f"[Enclave] CHAT_STREAM error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        self._send_event(conn, {"error": str(e), "is_final": True})
```

**Step 3: Update handle_request to route CHAT_STREAM**

Replace the `handle_request` method:

```python
def handle_request(self, request: dict, conn: socket.socket) -> dict:
    """Route request to appropriate handler."""
    command = request.get("command", "").upper()

    # Streaming commands handle their own response
    if command == "CHAT_STREAM":
        self.handle_chat_stream(request, conn)
        return None  # Response already sent

    # Non-streaming commands
    handlers = {
        "GET_PUBLIC_KEY": self.handle_get_public_key,
        "SET_CREDENTIALS": lambda: self.handle_set_credentials(request),
        "HEALTH": self.handle_health,
        "CHAT": lambda: self.handle_chat(request),
        "RUN_TESTS": self.handle_run_tests,
    }

    handler = handlers.get(command)
    if handler:
        return handler()
    else:
        return {
            "status": "error",
            "error": f"Unknown command: {command}",
            "available_commands": list(handlers.keys()) + ["CHAT_STREAM"],
        }
```

**Step 4: Update handle_client for streaming support**

Replace the `handle_client` function:

```python
def handle_client(server: BedrockServer, conn: socket.socket, addr: tuple):
    """Handle a single client connection."""
    cid, port = addr
    print(f"[Enclave] Connection from CID={cid}, port={port}", flush=True)

    try:
        # Receive data (up to 1MB for large payloads)
        data = conn.recv(1048576)
        if not data:
            print("[Enclave] Client disconnected", flush=True)
            return

        try:
            request = json.loads(data.decode("utf-8"))
            command = request.get("command", "unknown")
            print(f"[Enclave] Received command: {command}", flush=True)

            response = server.handle_request(request, conn)

            # Only send response if handler returned one (non-streaming)
            if response is not None:
                response["source"] = "nitro-enclave-bedrock"
                conn.sendall(json.dumps(response).encode("utf-8"))
                print("[Enclave] Sent response", flush=True)

        except json.JSONDecodeError as e:
            response = {
                "status": "error",
                "source": "nitro-enclave-bedrock",
                "error": f"Invalid JSON: {e}",
            }
            conn.sendall(json.dumps(response).encode("utf-8"))

    except Exception as e:
        print(f"[Enclave] Error handling client: {e}", flush=True)
    finally:
        conn.close()
```

**Step 5: Update typing import**

Ensure `List` is imported (should already be there from existing code).

**Step 6: Verify syntax**

Run: `cd /Users/prasiddhaparthsarthy/Desktop/freebird/backend && python -c "from enclave.bedrock_server import BedrockServer; print('Import OK')"`
Expected: `Import OK`

**Step 7: Run linting**

Run: `cd /Users/prasiddhaparthsarthy/Desktop/freebird/backend && ruff check enclave/bedrock_server.py && ruff format enclave/bedrock_server.py`
Expected: No errors

**Step 8: Commit Phase 1**

Run:
```bash
cd /Users/prasiddhaparthsarthy/Desktop/freebird/backend
git add enclave/vsock_http_client.py enclave/bedrock_client.py enclave/bedrock_server.py
git commit -m "feat(enclave): add real streaming support for CHAT_STREAM command

- Add request_stream() to VsockHttpClient for streaming HTTP responses
- Add AWS event stream parser to BedrockClient
- Add CHAT_STREAM command handler with encrypted chunk streaming
- Update handle_client for streaming response handling

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Phase 2: Parent-Side Client

### Task 2.1: Add Configuration Settings

**Files:**
- Modify: `core/config.py`

**Step 1: Add enclave settings**

Find the `Settings` class and add after existing settings:

```python
    # Enclave mode: "mock" for development, "nitro" for production
    ENCLAVE_MODE: str = "mock"

    # Nitro enclave settings (only used when ENCLAVE_MODE=nitro)
    ENCLAVE_CID: int = 0  # 0 = auto-discover
    ENCLAVE_PORT: int = 5000

    # Credential refresh interval (seconds) - creds expire after 1 hour
    ENCLAVE_CREDENTIAL_REFRESH_SECONDS: int = 2700  # 45 minutes
```

**Step 2: Verify syntax**

Run: `cd /Users/prasiddhaparthsarthy/Desktop/freebird/backend && python -c "from core.config import settings; print(f'ENCLAVE_MODE={settings.ENCLAVE_MODE}')"`
Expected: `ENCLAVE_MODE=mock`

**Step 3: Run linting**

Run: `cd /Users/prasiddhaparthsarthy/Desktop/freebird/backend && ruff check core/config.py && ruff format core/config.py`
Expected: No errors

---

### Task 2.2: Create NitroEnclaveClient

**Files:**
- Create: `core/enclave/nitro_enclave_client.py`

**Step 1: Create the file with full implementation**

```python
"""
Nitro Enclave client for production deployment.

This client implements EnclaveInterface and communicates with the
real Nitro Enclave via vsock. It's used when ENCLAVE_MODE=nitro.
"""

import asyncio
import json
import logging
import socket
from datetime import datetime, timedelta
from typing import AsyncGenerator, Dict, List, Optional, Union

from core.config import settings
from core.crypto import EncryptedPayload
from .mock_enclave import (
    EnclaveInterface,
    EnclaveInfo,
    StreamChunk,
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
            "NitroEnclaveClient does not expose decrypt_transport_message. "
            "Use process_message_streaming instead."
        )

    def encrypt_for_storage(
        self,
        plaintext: bytes,
        storage_public_key: bytes,
        is_assistant: bool,
    ) -> EncryptedPayload:
        """Not implemented - encryption happens inside enclave during CHAT_STREAM."""
        raise NotImplementedError(
            "NitroEnclaveClient does not expose encrypt_for_storage. "
            "Use process_message_streaming instead."
        )

    def encrypt_for_transport(
        self,
        plaintext: bytes,
        recipient_public_key: bytes,
    ) -> EncryptedPayload:
        """Not implemented - encryption happens inside enclave during CHAT_STREAM."""
        raise NotImplementedError(
            "NitroEnclaveClient does not expose encrypt_for_transport. "
            "Use process_message_streaming instead."
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
            "NitroEnclaveClient does not support non-streaming. "
            "Use process_message_streaming instead."
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
            "NitroEnclaveClient does not support process_message_stream. "
            "Use process_message_streaming instead."
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

        try:
            for event in self._send_command_stream(command):
                if event.get("error"):
                    logger.error(f"Enclave error: {event['error']}")
                    yield StreamChunk(error=event["error"], is_final=True)
                    return

                if event.get("encrypted_content"):
                    yield StreamChunk(
                        encrypted_content=EncryptedPayload.from_dict(event["encrypted_content"])
                    )

                if event.get("is_final"):
                    yield StreamChunk(
                        stored_user_message=EncryptedPayload.from_dict(event["stored_user_message"]),
                        stored_assistant_message=EncryptedPayload.from_dict(event["stored_assistant_message"]),
                        model_used=event.get("model_used", model),
                        input_tokens=event.get("input_tokens", 0),
                        output_tokens=event.get("output_tokens", 0),
                        is_final=True,
                    )

        except EnclaveConnectionError as e:
            logger.error(f"Enclave connection error: {e}")
            yield StreamChunk(error="Service temporarily unavailable", is_final=True)

        except EnclaveTimeoutError:
            logger.error("Enclave timeout during streaming")
            yield StreamChunk(error="Request timed out", is_final=True)

        except Exception as e:
            logger.exception("Unexpected error in enclave streaming")
            yield StreamChunk(error="Internal error", is_final=True)

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

        response = self._send_command({
            "command": "SET_CREDENTIALS",
            "credentials": creds,
        }, timeout=10.0)

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
```

**Step 2: Verify syntax**

Run: `cd /Users/prasiddhaparthsarthy/Desktop/freebird/backend && python -c "from core.enclave.nitro_enclave_client import NitroEnclaveClient; print('Import OK')"`
Expected: `Import OK`

**Step 3: Run linting**

Run: `cd /Users/prasiddhaparthsarthy/Desktop/freebird/backend && ruff check core/enclave/nitro_enclave_client.py && ruff format core/enclave/nitro_enclave_client.py`
Expected: No errors

---

### Task 2.3: Update Enclave Factory

**Files:**
- Modify: `core/enclave/__init__.py`

**Step 1: Add imports and update factory**

Replace the entire file content:

```python
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
from typing import Union, Optional

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
_enclave_instance: Union[MockEnclave, "NitroEnclaveClient", None] = None


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
```

**Step 2: Verify syntax**

Run: `cd /Users/prasiddhaparthsarthy/Desktop/freebird/backend && python -c "from core.enclave import get_enclave; print('Import OK')"`
Expected: `Import OK`

**Step 3: Run linting**

Run: `cd /Users/prasiddhaparthsarthy/Desktop/freebird/backend && ruff check core/enclave/__init__.py && ruff format core/enclave/__init__.py`
Expected: No errors

**Step 4: Commit Phase 2**

Run:
```bash
cd /Users/prasiddhaparthsarthy/Desktop/freebird/backend
git add core/config.py core/enclave/nitro_enclave_client.py core/enclave/__init__.py
git commit -m "feat(enclave): add NitroEnclaveClient and ENCLAVE_MODE config

- Add ENCLAVE_MODE, ENCLAVE_CID, ENCLAVE_PORT settings
- Create NitroEnclaveClient implementing EnclaveInterface
- Update get_enclave() factory to support mock/nitro toggle
- Add credential refresh background task
- Add startup_enclave/shutdown_enclave lifecycle hooks

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Phase 3: Integration

### Task 3.1: Add Startup/Shutdown Hooks to FastAPI

**Files:**
- Modify: `main.py`

**Step 1: Add enclave lifecycle hooks**

Find the existing startup/shutdown event handlers and add enclave hooks. If they don't exist, add them:

```python
from core.enclave import startup_enclave, shutdown_enclave

@app.on_event("startup")
async def startup():
    """Application startup handler."""
    await startup_enclave()

@app.on_event("shutdown")
async def shutdown():
    """Application shutdown handler."""
    await shutdown_enclave()
```

If there are existing startup/shutdown handlers, add the enclave calls to them.

**Step 2: Verify syntax**

Run: `cd /Users/prasiddhaparthsarthy/Desktop/freebird/backend && python -c "from main import app; print('Import OK')"`
Expected: `Import OK`

**Step 3: Run linting**

Run: `cd /Users/prasiddhaparthsarthy/Desktop/freebird/backend && ruff check main.py && ruff format main.py`
Expected: No errors

---

### Task 3.2: Add Enclave Health Endpoint

**Files:**
- Modify: `routers/chat.py`

**Step 1: Add health check endpoint**

Add after the existing `/enclave/info` endpoint:

```python
@router.get("/enclave/health")
async def get_enclave_health(
    auth: AuthContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Check enclave health and connectivity.

    Returns enclave status, mode (mock/nitro), and credential status.
    """
    from core.enclave import get_enclave
    from core.config import settings

    try:
        enclave = get_enclave()

        # For NitroEnclaveClient, use health_check method
        if hasattr(enclave, 'health_check'):
            return enclave.health_check()

        # For MockEnclave, return basic info
        info = enclave.get_info()
        return {
            "status": "healthy",
            "mode": "mock",
            "public_key": info.enclave_public_key.hex()[:16] + "...",
        }

    except Exception as e:
        logger.error(f"Enclave health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Enclave not available",
        )
```

**Step 2: Verify syntax**

Run: `cd /Users/prasiddhaparthsarthy/Desktop/freebird/backend && python -c "from routers.chat import router; print('Import OK')"`
Expected: `Import OK`

**Step 3: Run linting**

Run: `cd /Users/prasiddhaparthsarthy/Desktop/freebird/backend && ruff check routers/chat.py && ruff format routers/chat.py`
Expected: No errors

**Step 4: Commit Phase 3**

Run:
```bash
cd /Users/prasiddhaparthsarthy/Desktop/freebird/backend
git add main.py routers/chat.py
git commit -m "feat(api): add enclave lifecycle hooks and health endpoint

- Add startup_enclave/shutdown_enclave to FastAPI lifecycle
- Add /enclave/health endpoint for monitoring

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Phase 4: Testing & Deployment

### Task 4.1: Run Local Tests (Mock Mode)

**Step 1: Run existing tests**

Run: `cd /Users/prasiddhaparthsarthy/Desktop/freebird/backend && ./run_tests.sh`
Expected: All tests pass (MockEnclave still works)

**Step 2: Verify mock mode still works**

Run: `cd /Users/prasiddhaparthsarthy/Desktop/freebird/backend && ENCLAVE_MODE=mock python -c "from core.enclave import get_enclave; e = get_enclave(); print(f'Mode: mock, Key: {e.get_transport_public_key()[:16]}...')"`
Expected: `Mode: mock, Key: <hex>...`

---

### Task 4.2: Deploy and Test on EC2

**Step 1: Push changes**

Run:
```bash
cd /Users/prasiddhaparthsarthy/Desktop/freebird/backend
git push
```

**Step 2: Watch CI/CD**

Run: `gh run watch --repo Isol8AI/backend --exit-status`
Expected: All jobs pass

**Step 3: SSH to EC2 and verify enclave is running**

Run on EC2:
```bash
nitro-cli describe-enclaves
```
Expected: Shows running enclave with CID

**Step 4: Test enclave health via API**

Run on EC2:
```bash
# Set ENCLAVE_MODE in .env
echo "ENCLAVE_MODE=nitro" >> /home/ec2-user/.env
echo "ENCLAVE_CID=0" >> /home/ec2-user/.env

# Restart service
sudo systemctl restart isol8-backend

# Test health endpoint
curl -s http://localhost:8000/api/v1/chat/enclave/health | jq .
```
Expected: `{"status": "healthy", "mode": "nitro", ...}`

**Step 5: Test streaming chat**

Use the frontend or a test script to verify streaming works with real enclave.

---

## Success Criteria Checklist

- [ ] All existing tests pass with ENCLAVE_MODE=mock
- [ ] NitroEnclaveClient connects to enclave on EC2
- [ ] GET_PUBLIC_KEY works via NitroEnclaveClient
- [ ] CHAT_STREAM returns encrypted streaming chunks
- [ ] Credentials refresh automatically
- [ ] Health endpoint returns correct status
- [ ] ChatService/routes code unchanged
- [ ] Rollback works (set ENCLAVE_MODE=mock)
