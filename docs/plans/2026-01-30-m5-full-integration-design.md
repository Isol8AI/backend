# M5: Full Integration Design

**Date:** 2026-01-30
**Status:** Approved
**Goal:** Replace MockEnclave with real Nitro Enclave while keeping FastAPI backend code unchanged

## Overview

M5 integrates the Nitro Enclave (proven in M4) with the production FastAPI backend. The key principle is that `ChatService` and routes don't change - they call `get_enclave()` which returns either `MockEnclave` or `NitroEnclaveClient` based on configuration.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PARENT (EC2 Instance)                           │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      FastAPI Backend                             │   │
│  │                                                                  │   │
│  │  ChatService ──► get_enclave() ──► EnclaveInterface             │   │
│  │                                           │                      │   │
│  │                        ┌──────────────────┴──────────────────┐  │   │
│  │                        │                                      │  │   │
│  │              ENCLAVE_MODE=mock              ENCLAVE_MODE=nitro│  │   │
│  │                        │                                      │  │   │
│  │                        ▼                                      ▼  │   │
│  │                  MockEnclave                    NitroEnclaveClient  │
│  │                  (in-process)                   (vsock to enclave)  │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                         │               │
│  ┌──────────────┐                                       │ vsock        │
│  │ vsock-proxy  │◄──────────────────────────────────────┼──────────┐   │
│  └──────────────┘                                       │          │   │
└─────────────────────────────────────────────────────────┼──────────┼───┘
                                                          │          │
                    ┌─────────────────────────────────────┼──────────┼───┐
                    │              NITRO ENCLAVE          │          │   │
                    │                                     ▼          │   │
                    │  ┌─────────────────────────────────────────┐   │   │
                    │  │           bedrock_server.py             │   │   │
                    │  │  - Receives commands via vsock          │   │   │
                    │  │  - Decrypts messages                    │   │   │
                    │  │  - Calls Bedrock (streaming) ───────────┼───┘   │
                    │  │  - Re-encrypts responses                │       │
                    │  └─────────────────────────────────────────┘       │
                    └────────────────────────────────────────────────────┘
```

---

## Section 1: Enclave-Side Changes (Streaming)

**Files to modify:** `enclave/vsock_http_client.py`, `enclave/bedrock_client.py`

### 1a. VsockHttpClient Streaming

Add a `request_stream()` method that yields response chunks instead of buffering:

```python
# vsock_http_client.py

def request_stream(
    self,
    method: str,
    url: str,
    headers: Dict[str, str],
    body: bytes,
) -> Generator[bytes, None, None]:
    """
    Make streaming HTTP request. Yields raw response chunks.

    For AWS event stream format, each chunk is:
    - 4 bytes: total byte length (big-endian)
    - 4 bytes: headers byte length (big-endian)
    - 4 bytes: prelude CRC
    - Headers (variable)
    - Payload (variable)
    - 4 bytes: message CRC
    """
    # Same setup as request() - connect, CONNECT proxy, TLS wrap
    sock = self._create_vsock()
    # ... setup code ...

    # Send request
    sock.sendall(request)

    # Read headers first
    headers = self._read_headers(sock)

    # Stream body chunks as they arrive
    while True:
        # Read AWS event stream message length (4 bytes)
        length_bytes = self._read_exact(sock, 4)
        if not length_bytes:
            break
        total_length = int.from_bytes(length_bytes, 'big')

        # Read rest of message
        message = length_bytes + self._read_exact(sock, total_length - 4)
        yield message
```

### 1b. Event Stream Parser

Add a simple parser for AWS event stream format:

```python
# bedrock_client.py

def _parse_event_stream_message(self, data: bytes) -> dict:
    """Parse single AWS event stream message into event dict."""
    # Skip prelude (12 bytes: length + headers_length + crc)
    headers_length = int.from_bytes(data[4:8], 'big')
    headers_end = 12 + headers_length

    # Parse headers to get event type
    headers = self._parse_event_headers(data[12:headers_end])

    # Extract payload (skip message CRC at end)
    payload = data[headers_end:-4]

    # Payload is JSON for Bedrock events
    if payload:
        return json.loads(payload.decode('utf-8'))
    return {}
```

### 1c. Real Streaming in converse_stream()

```python
# bedrock_client.py

def converse_stream(self, model_id, messages, system, inference_config):
    """Real streaming - yields events as they arrive."""
    # ... build request ...

    for raw_message in self.http_client.request_stream("POST", url, headers, body):
        event = self._parse_event_stream_message(raw_message)

        if "contentBlockDelta" in event:
            yield {"type": "content", "text": event["contentBlockDelta"]["delta"]["text"]}
        elif "messageStop" in event:
            yield {"type": "stop", "reason": event["messageStop"]["stopReason"]}
        elif "metadata" in event:
            yield {"type": "metadata", "usage": event["metadata"]["usage"]}
```

---

## Section 2: Parent-Side Changes (NitroEnclaveClient)

**Files to create/modify:** `core/enclave/nitro_enclave_client.py`, `core/enclave/__init__.py`

### 2a. NitroEnclaveClient Class

```python
# core/enclave/nitro_enclave_client.py

class NitroEnclaveClient(EnclaveInterface):
    """
    Client for communicating with real Nitro Enclave via vsock.

    Implements the same interface as MockEnclave so ChatService
    and routes work unchanged.
    """

    def __init__(self, enclave_cid: int, enclave_port: int = 5000):
        self._cid = enclave_cid
        self._port = enclave_port
        self._enclave_public_key: Optional[bytes] = None

        # Fetch enclave's public key on init
        self._refresh_public_key()

    def _send_command(self, command: dict, timeout: float = 120.0) -> dict:
        """Send command to enclave via vsock, return response."""
        sock = socket.socket(AF_VSOCK, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        try:
            sock.connect((self._cid, self._port))
            sock.sendall(json.dumps(command).encode('utf-8'))
            response = sock.recv(1048576)  # 1MB buffer
            return json.loads(response.decode('utf-8'))
        finally:
            sock.close()

    def _send_command_stream(self, command: dict) -> Generator[dict, None, None]:
        """Send command and stream response events."""
        sock = socket.socket(AF_VSOCK, socket.SOCK_STREAM)
        sock.settimeout(120.0)
        try:
            sock.connect((self._cid, self._port))
            sock.sendall(json.dumps(command).encode('utf-8'))

            # Read streaming JSON events (newline-delimited)
            buffer = b""
            while True:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                buffer += chunk

                # Parse complete JSON objects
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if line:
                        event = json.loads(line.decode('utf-8'))
                        yield event
                        if event.get("is_final"):
                            return
        finally:
            sock.close()
```

### 2b. Implementing EnclaveInterface Methods

```python
# core/enclave/nitro_enclave_client.py (continued)

    def get_info(self) -> EnclaveInfo:
        """Get enclave's public key and attestation."""
        response = self._send_command({"command": "GET_PUBLIC_KEY"})
        return EnclaveInfo(
            enclave_public_key=bytes.fromhex(response["public_key"]),
            attestation_document=None,  # M6 will add attestation
        )

    def decrypt_transport_message(self, payload: EncryptedPayload) -> bytes:
        """Decrypt message - happens inside enclave, we just get result."""
        raise NotImplementedError("Use process_message_streaming instead")

    def encrypt_for_storage(self, plaintext: bytes, storage_public_key: bytes, is_assistant: bool) -> EncryptedPayload:
        """Encrypt for storage - happens inside enclave."""
        raise NotImplementedError("Use process_message_streaming instead")

    def encrypt_for_transport(self, plaintext: bytes, recipient_public_key: bytes) -> EncryptedPayload:
        """Encrypt for transport - happens inside enclave."""
        raise NotImplementedError("Use process_message_streaming instead")
```

### 2c. The Main Streaming Method

```python
    async def process_message_streaming(
        self,
        encrypted_message: EncryptedPayload,
        encrypted_history: List[EncryptedPayload],
        facts_context: Optional[str],
        storage_public_key: bytes,
        client_public_key: bytes,
        session_id: str,
        model: str,
        **kwargs,
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Process message through Nitro Enclave with streaming.

        Sends CHAT_STREAM command, yields StreamChunk objects as
        enclave streams back encrypted response chunks.
        """
        command = {
            "command": "CHAT_STREAM",
            "encrypted_message": encrypted_message.to_dict(),
            "encrypted_history": [h.to_dict() for h in encrypted_history],
            "storage_public_key": storage_public_key.hex(),
            "client_public_key": client_public_key.hex(),
            "model_id": model,
            "session_id": session_id,
        }

        for event in self._send_command_stream(command):
            if event.get("error"):
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
```

---

## Section 3: Configuration & Enclave Toggle

**Files to modify:** `core/config.py`, `core/enclave/__init__.py`

### 3a. Configuration Settings

```python
# core/config.py

class Settings(BaseSettings):
    # ... existing settings ...

    # Enclave mode: "mock" for development, "nitro" for production
    ENCLAVE_MODE: str = "mock"

    # Nitro enclave settings (only used when ENCLAVE_MODE=nitro)
    ENCLAVE_CID: int = 0  # 0 = auto-discover
    ENCLAVE_PORT: int = 5000

    # Credential refresh interval (enclave needs fresh IAM creds)
    ENCLAVE_CREDENTIAL_REFRESH_SECONDS: int = 2700  # 45 min
```

### 3b. Updated get_enclave() Factory

```python
# core/enclave/__init__.py

_enclave_instance: Union[MockEnclave, "NitroEnclaveClient", None] = None

def get_enclave() -> EnclaveInterface:
    """
    Get the enclave instance based on ENCLAVE_MODE config.

    - ENCLAVE_MODE=mock: Returns MockEnclave (in-process, for dev)
    - ENCLAVE_MODE=nitro: Returns NitroEnclaveClient (vsock to real enclave)
    """
    global _enclave_instance

    if _enclave_instance is None:
        if settings.ENCLAVE_MODE == "nitro":
            from .nitro_enclave_client import NitroEnclaveClient

            cid = settings.ENCLAVE_CID
            if cid == 0:
                cid = _discover_enclave_cid()

            _enclave_instance = NitroEnclaveClient(
                enclave_cid=cid,
                enclave_port=settings.ENCLAVE_PORT,
            )
            logger.info(f"Using NitroEnclaveClient (CID={cid})")
        else:
            _enclave_instance = MockEnclave(
                aws_region=settings.AWS_REGION,
                inference_timeout=settings.ENCLAVE_INFERENCE_TIMEOUT,
            )
            logger.info("Using MockEnclave (development mode)")

    return _enclave_instance


def _discover_enclave_cid() -> int:
    """Discover running enclave's CID using nitro-cli."""
    import subprocess
    import json

    try:
        result = subprocess.run(
            ["nitro-cli", "describe-enclaves"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        enclaves = json.loads(result.stdout)
        if enclaves:
            return enclaves[0]["EnclaveCID"]
    except Exception as e:
        logger.warning(f"Could not discover enclave CID: {e}")

    raise RuntimeError("No running enclave found. Start enclave first.")
```

### 3c. Environment-Based Defaults

| Environment | ENCLAVE_MODE | Reason |
|-------------|--------------|--------|
| Local dev | `mock` | No Nitro hardware available |
| CI/CD tests | `mock` | Tests run in containers |
| Dev EC2 | `nitro` | Testing real enclave |
| Staging | `nitro` | Pre-prod validation |
| Production | `nitro` | Full security |

---

## Section 4: Enclave Protocol Changes

**Files to modify:** `enclave/bedrock_server.py`

### 4a. New CHAT_STREAM Command

```python
# enclave/bedrock_server.py

def handle_chat_stream(self, data: dict, conn: socket.socket) -> None:
    """
    Process encrypted chat with streaming response.

    Streams newline-delimited JSON events back to the parent:
    - {"encrypted_content": {...}}  - Encrypted chunk for client
    - {"is_final": true, "stored_user_message": {...}, ...}  - Final event
    - {"error": "...", "is_final": true}  - Error event
    """
    try:
        if not self.bedrock.has_credentials():
            self._send_event(conn, {"error": "No credentials", "is_final": True})
            return

        # Extract parameters
        encrypted_message = EncryptedPayload.from_dict(data["encrypted_message"])
        encrypted_history = [EncryptedPayload.from_dict(h) for h in data.get("encrypted_history", [])]
        storage_public_key = hex_to_bytes(data["storage_public_key"])
        client_public_key = hex_to_bytes(data["client_public_key"])
        model_id = data["model_id"]

        # Decrypt user message
        user_plaintext = decrypt_with_private_key(
            self.keypair.private_key,
            encrypted_message,
            "client-to-enclave-transport",
        )
        user_content = user_plaintext.decode("utf-8")

        # Decrypt history
        history = self._decrypt_history(encrypted_history)

        # Build messages for Bedrock
        messages = build_converse_messages(history, user_content)
        system = [{"text": "You are a helpful AI assistant."}]

        # Stream from Bedrock, encrypt and forward each chunk
        full_response = ""
        input_tokens = 0
        output_tokens = 0

        for event in self.bedrock.converse_stream(model_id, messages, system):
            if event["type"] == "content":
                chunk_text = event["text"]
                full_response += chunk_text

                encrypted_chunk = encrypt_to_public_key(
                    client_public_key,
                    chunk_text.encode("utf-8"),
                    "enclave-to-client-transport",
                )
                self._send_event(conn, {"encrypted_content": encrypted_chunk.to_dict()})

            elif event["type"] == "metadata":
                input_tokens = event["usage"].get("inputTokens", 0)
                output_tokens = event["usage"].get("outputTokens", 0)

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

    except Exception as e:
        self._send_event(conn, {"error": str(e), "is_final": True})

def _send_event(self, conn: socket.socket, event: dict) -> None:
    """Send newline-delimited JSON event."""
    conn.sendall(json.dumps(event).encode("utf-8") + b"\n")
```

### 4b. Updated Request Handler

```python
def handle_request(self, request: dict, conn: socket.socket) -> Optional[dict]:
    """Route request to appropriate handler."""
    command = request.get("command", "").upper()

    # Streaming commands handle their own response
    if command == "CHAT_STREAM":
        self.handle_chat_stream(request, conn)
        return None

    # Non-streaming commands return dict
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
    return {"status": "error", "error": f"Unknown command: {command}"}
```

---

## Section 5: Credential Management

**Problem:** IAM role credentials expire after ~1 hour. The enclave needs fresh credentials.

### 5a. Credential Refresh on Parent

```python
# core/enclave/nitro_enclave_client.py

class NitroEnclaveClient(EnclaveInterface):
    def __init__(self, enclave_cid: int, enclave_port: int = 5000):
        self._cid = enclave_cid
        self._port = enclave_port
        self._credentials_task: Optional[asyncio.Task] = None
        self._credentials_expiration: Optional[datetime] = None

        self._refresh_public_key()
        self._push_credentials_sync()

    async def start_credential_refresh(self):
        """Start background task to refresh enclave credentials."""
        if self._credentials_task is None:
            self._credentials_task = asyncio.create_task(self._credential_refresh_loop())

    async def _credential_refresh_loop(self):
        """Refresh credentials every 45 minutes."""
        while True:
            try:
                await asyncio.sleep(settings.ENCLAVE_CREDENTIAL_REFRESH_SECONDS)
                await self._push_credentials_async()
                logger.info("Refreshed enclave credentials")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Credential refresh failed: {e}")
                await asyncio.sleep(60)
```

### 5b. Fetching Credentials from IMDS

```python
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
```

### 5c. Startup/Shutdown Hooks

```python
# core/enclave/__init__.py

async def startup_enclave() -> None:
    """Initialize enclave on application startup."""
    enclave = get_enclave()
    if isinstance(enclave, NitroEnclaveClient):
        await enclave.start_credential_refresh()

async def shutdown_enclave() -> None:
    """Cleanup enclave on application shutdown."""
    global _enclave_instance
    if isinstance(_enclave_instance, NitroEnclaveClient):
        await _enclave_instance.stop_credential_refresh()
    _enclave_instance = None
```

---

## Section 6: Error Handling & Resilience

### 6a. Error Categories

| Error Type | Where | Handling |
|------------|-------|----------|
| Enclave not running | Parent | Return 503, log error |
| vsock connection failed | Parent | Return 503, attempt reconnect |
| Credential expired | Enclave | Parent pushes fresh creds, retry |
| Bedrock API error | Enclave | Stream error event to client |
| Decryption failed | Enclave | Stream error event |
| Timeout | Both | Clean up connection, return 504 |

### 6b. NitroEnclaveClient Error Handling

```python
class EnclaveConnectionError(Exception):
    """Raised when cannot connect to enclave."""
    pass

class EnclaveTimeoutError(Exception):
    """Raised when enclave request times out."""
    pass

def _send_command(self, command: dict, timeout: float = 120.0) -> dict:
    """Send command with error handling."""
    try:
        sock = socket.socket(AF_VSOCK, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((self._cid, self._port))
        sock.sendall(json.dumps(command).encode('utf-8'))
        response = sock.recv(1048576)
        return json.loads(response.decode('utf-8'))

    except socket.timeout:
        raise EnclaveTimeoutError("Enclave request timed out")

    except ConnectionRefusedError:
        raise EnclaveConnectionError("Enclave not running")

    finally:
        sock.close()
```

---

## Section 7: Testing Strategy

### 7a. Testing Layers

| Layer | What | How | Where |
|-------|------|-----|-------|
| Unit | NitroEnclaveClient logic | Mock vsock | CI/CD |
| Unit | Enclave streaming parser | Test event parsing | CI/CD |
| Integration | Mock↔Nitro parity | Same tests, both modes | CI/CD |
| E2E | Full flow on EC2 | Real enclave, real Bedrock | Dev EC2 |

### 7b. Parity Tests

```python
@pytest.fixture(params=["mock", "nitro"])
def enclave_mode(request, monkeypatch):
    """Run test with both enclave modes."""
    monkeypatch.setenv("ENCLAVE_MODE", request.param)
    reset_enclave()

    if request.param == "nitro":
        pytest.importorskip("check_enclave_available")

    yield request.param
    reset_enclave()
```

---

## Section 8: Implementation Summary

### Files to Create

| File | Purpose | Lines (est.) |
|------|---------|--------------|
| `core/enclave/nitro_enclave_client.py` | Parent-side client | ~300 |
| `tests/unit/enclave/test_nitro_enclave_client.py` | Unit tests | ~150 |
| `tests/integration/test_enclave_parity.py` | Parity tests | ~100 |
| `scripts/test-m5-e2e.sh` | E2E test script | ~50 |

### Files to Modify

| File | Changes |
|------|---------|
| `enclave/vsock_http_client.py` | Add `request_stream()` (~100 lines) |
| `enclave/bedrock_client.py` | Real streaming (~50 lines) |
| `enclave/bedrock_server.py` | `CHAT_STREAM` command (~150 lines) |
| `core/enclave/__init__.py` | Factory update, hooks (~50 lines) |
| `core/config.py` | New settings (~10 lines) |
| `main.py` | Startup/shutdown handlers (~10 lines) |
| `routers/chat.py` | Health endpoint (~20 lines) |

### Implementation Order

```
Phase 1: Enclave Streaming
├── 1.1 vsock_http_client.py - request_stream()
├── 1.2 bedrock_client.py - real converse_stream()
└── 1.3 bedrock_server.py - CHAT_STREAM command

Phase 2: Parent Client
├── 2.1 core/config.py - new settings
├── 2.2 nitro_enclave_client.py - full implementation
└── 2.3 core/enclave/__init__.py - factory update

Phase 3: Integration
├── 3.1 main.py - startup/shutdown hooks
├── 3.2 routers/chat.py - health endpoint
└── 3.3 CI/CD - add ENCLAVE_MODE to EC2 .env

Phase 4: Testing
├── 4.1 Unit tests
├── 4.2 Parity tests
└── 4.3 E2E on EC2
```

### Success Criteria

| Criteria | Verification |
|----------|--------------|
| MockEnclave still works | Existing tests pass with `ENCLAVE_MODE=mock` |
| NitroEnclaveClient connects | Health check returns success on EC2 |
| Streaming works | Response chunks appear incrementally in UI |
| Credentials refresh | No auth errors after 1+ hour |
| Error handling | Graceful errors when enclave stopped |
| No code duplication | ChatService/routes unchanged |

### Rollback Plan

If issues arise in production:
1. Set `ENCLAVE_MODE=mock` in EC2 .env
2. Restart FastAPI service
3. Service continues with MockEnclave

---

## Total Estimated Work

~800 lines of new/modified code

## Key Decisions

1. NitroEnclaveClient implements EnclaveInterface (no backend code changes)
2. Real streaming via event stream parsing in enclave
3. Newline-delimited JSON for vsock streaming protocol
4. Automatic credential refresh every 45 minutes
5. CID auto-discovery via `nitro-cli describe-enclaves`
