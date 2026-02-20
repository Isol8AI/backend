#!/usr/bin/env python3
"""
M4: Bedrock Server for Nitro Enclave
=====================================

A vsock server that provides secure LLM inference via AWS Bedrock.
The model is selected by the frontend and passed through the request.

Flow:
1. Parent sends SET_CREDENTIALS command with IAM role credentials
2. Client encrypts message to enclave's transport key
3. Enclave decrypts, calls Bedrock via vsock-proxy
4. Enclave re-encrypts response to user's storage key
5. Response returned to parent

Commands:
- GET_PUBLIC_KEY: Returns enclave's transport public key
- SET_CREDENTIALS: Sets AWS credentials for Bedrock API calls
- CHAT: Send encrypted message with model_id, get LLM response
- CHAT_STREAM: Send encrypted message with streaming response (newline-delimited JSON)
- HEALTH: Check enclave and Bedrock connectivity status
- RUN_TESTS: Execute crypto test vectors

Security properties:
- Plaintext messages only exist inside enclave memory
- TLS to Bedrock terminates inside enclave
- Parent cannot read message content
"""

import socket
import sys
import json
import os
import time
import io
import subprocess
import tarfile
import shutil
import tempfile
from pathlib import Path
from typing import List

from crypto_primitives import (
    generate_x25519_keypair,
    encrypt_to_public_key,
    decrypt_with_private_key,
    EncryptedPayload,
    KeyPair,
    bytes_to_hex,
    hex_to_bytes,
)
from bedrock_client import BedrockClient, BedrockResponse, build_converse_messages, ConverseTurn
from kms_encryption import encrypt_with_kms, decrypt_with_kms, set_kms_credentials
from agent_bridge import run_agent_streaming, collect_response_text

# vsock constants
VSOCK_PORT = 5000
AF_VSOCK = 40


class BedrockServer:
    """Secure Bedrock inference server for Nitro Enclave."""

    def __init__(self, region: str = "us-east-1"):
        """Initialize server with transport keypair."""
        self.keypair: KeyPair = generate_x25519_keypair()
        self.bedrock = BedrockClient(region=region)
        self.region = region

        print("[Enclave] Generated transport keypair", flush=True)
        print(f"[Enclave] Public key: {bytes_to_hex(self.keypair.public_key)}", flush=True)
        print(f"[Enclave] Bedrock region: {region}", flush=True)

    def handle_get_public_key(self) -> dict:
        """Return the enclave's transport public key."""
        return {
            "status": "success",
            "command": "GET_PUBLIC_KEY",
            "public_key": bytes_to_hex(self.keypair.public_key),
            "region": self.region,
        }

    def handle_set_credentials(self, data: dict) -> dict:
        """Set AWS credentials for Bedrock API calls (from parent's IAM role)."""
        try:
            credentials = data.get("credentials", {})

            self.bedrock.set_credentials(
                access_key_id=credentials["access_key_id"],
                secret_access_key=credentials["secret_access_key"],
                session_token=credentials["session_token"],
                expiration=credentials.get("expiration"),
            )

            # Also set KMS credentials for envelope encryption
            set_kms_credentials(
                access_key_id=credentials["access_key_id"],
                secret_access_key=credentials["secret_access_key"],
                session_token=credentials["session_token"],
                region=self.region,
            )

            # Set KMS_KEY_ID if provided by parent (needed for background-mode encryption)
            kms_key_id = data.get("kms_key_id")
            if kms_key_id:
                os.environ["KMS_KEY_ID"] = kms_key_id
                print(f"[Enclave] KMS_KEY_ID set: {kms_key_id[:8]}...", flush=True)

            # Store service API keys (e.g., BRAVE_API_KEY for web search)
            service_keys = data.get("service_keys", {})
            for key, value in service_keys.items():
                if key.startswith(("BRAVE_", "FIRECRAWL_")):  # allowlist prefixes
                    os.environ[key] = value
            if service_keys:
                print(f"[Enclave] Service keys set: {list(service_keys.keys())}", flush=True)

            print("[Enclave] AWS credentials set (Bedrock + KMS)", flush=True)
            if credentials.get("expiration"):
                print(f"[Enclave] Credentials expire: {credentials['expiration']}", flush=True)

            return {
                "status": "success",
                "command": "SET_CREDENTIALS",
                "has_credentials": True,
                "expiration": credentials.get("expiration"),
            }
        except KeyError as e:
            return {
                "status": "error",
                "command": "SET_CREDENTIALS",
                "error": f"Missing credential field: {e}",
            }
        except Exception as e:
            return {
                "status": "error",
                "command": "SET_CREDENTIALS",
                "error": str(e),
            }

    def handle_health(self) -> dict:
        """Check enclave health and Bedrock connectivity."""
        return {
            "status": "success",
            "command": "HEALTH",
            "enclave": "running",
            "has_credentials": self.bedrock.has_credentials(),
            "region": self.region,
            "public_key": bytes_to_hex(self.keypair.public_key),
        }

    def _get_aws_env(self) -> dict:
        """
        Build environment variables dict with AWS credentials for subprocess.

        The enclave receives credentials via SET_CREDENTIALS command from the
        parent instance. These are stored in self.bedrock but not in os.environ.
        This method extracts them so they can be passed to the Node.js bridge
        subprocess (which uses AWS SDK and reads standard env vars).
        """
        env = self.bedrock.get_credentials_env()
        env["AWS_REGION"] = self.region
        env["AWS_DEFAULT_REGION"] = self.region
        return env

    def handle_chat(self, data: dict) -> dict:
        """
        Process an encrypted chat message through Bedrock.

        Required fields:
        - encrypted_message: EncryptedPayload (encrypted to enclave key)
        - user_public_key: Hex string of user's storage public key
        - model_id: Model identifier from frontend (e.g., "anthropic.claude-3-5-haiku-20241022-v1:0")

        Optional fields:
        - history: List of prior messages [{role, content}]
        - system_prompt: Optional system prompt
        """
        try:
            # Check credentials
            if not self.bedrock.has_credentials():
                return {
                    "status": "error",
                    "command": "CHAT",
                    "error": "No AWS credentials. Parent must send SET_CREDENTIALS first.",
                }

            # Get required parameters
            user_public_key = hex_to_bytes(data["user_public_key"])
            model_id = data["model_id"]  # Required - comes from frontend

            if not model_id:
                return {
                    "status": "error",
                    "command": "CHAT",
                    "error": "model_id is required",
                }

            system_prompt = data.get("system_prompt")

            # Decrypt incoming message
            incoming_payload = EncryptedPayload.from_dict(data["encrypted_message"])
            plaintext = decrypt_with_private_key(
                self.keypair.private_key,
                incoming_payload,
                "client-to-enclave-transport",
            )
            user_message = plaintext.decode("utf-8")
            print(f"[Enclave] Decrypted message: {user_message[:50]}...", flush=True)
            print(f"[Enclave] Using model: {model_id}", flush=True)

            # Build conversation history
            history: List[ConverseTurn] = []
            for msg in data.get("history", []):
                history.append(ConverseTurn(role=msg["role"], content=msg["content"]))

            # Build Converse API messages
            messages = build_converse_messages(history, user_message)

            # Build system prompts
            system = None
            if system_prompt:
                system = [{"text": system_prompt}]
            else:
                system = [{"text": "You are a helpful AI assistant."}]

            # Call Bedrock Converse API
            print("[Enclave] Calling Bedrock Converse API...", flush=True)
            bedrock_response: BedrockResponse = self.bedrock.converse(
                model_id=model_id,
                messages=messages,
                system=system,
                inference_config={"maxTokens": 4096, "temperature": 0.7},
            )
            print(f"[Enclave] Response: {len(bedrock_response.content)} chars", flush=True)
            print(
                f"[Enclave] Tokens: in={bedrock_response.input_tokens}, out={bedrock_response.output_tokens}",
                flush=True,
            )

            # Re-encrypt response for storage (to user's key)
            response_payload = encrypt_to_public_key(
                user_public_key,
                bedrock_response.content.encode("utf-8"),
                "assistant-message-storage",
            )

            # Also encrypt the user's message for storage
            user_msg_payload = encrypt_to_public_key(
                user_public_key,
                user_message.encode("utf-8"),
                "user-message-storage",
            )

            return {
                "status": "success",
                "command": "CHAT",
                "encrypted_response": response_payload.to_dict(),
                "encrypted_user_message": user_msg_payload.to_dict(),
                "model_id": model_id,
                "usage": {
                    "input_tokens": bedrock_response.input_tokens,
                    "output_tokens": bedrock_response.output_tokens,
                },
                "stop_reason": bedrock_response.stop_reason,
            }

        except Exception as e:
            print(f"[Enclave] CHAT error: {e}", flush=True)
            import traceback

            traceback.print_exc()
            return {
                "status": "error",
                "command": "CHAT",
                "error": str(e),
            }

    def handle_run_tests(self) -> dict:
        """Run crypto test vectors (from M3)."""
        try:
            import test_crypto_vectors

            results = {"ecdh_tests": [], "aes_gcm_tests": []}

            from crypto_primitives import derive_key_from_ecdh, decrypt_aes_gcm

            for vector in test_crypto_vectors.TEST_VECTORS["ecdh_derivation"]:
                derived_key, _ = derive_key_from_ecdh(
                    hex_to_bytes(vector["private_key_hex"]),
                    hex_to_bytes(vector["public_key_hex"]),
                    vector["context"],
                    hex_to_bytes(vector["salt_hex"]),
                )
                passed = bytes_to_hex(derived_key) == vector["expected_key_hex"]
                results["ecdh_tests"].append(
                    {
                        "description": vector["description"],
                        "passed": passed,
                    }
                )

            for vector in test_crypto_vectors.TEST_VECTORS["aes_gcm"]:
                aad = hex_to_bytes(vector["aad_hex"]) if "aad_hex" in vector else None
                plaintext = decrypt_aes_gcm(
                    hex_to_bytes(vector["key_hex"]),
                    hex_to_bytes(vector["iv_hex"]),
                    hex_to_bytes(vector["ciphertext_hex"]),
                    hex_to_bytes(vector["auth_tag_hex"]),
                    aad,
                )
                passed = bytes_to_hex(plaintext) == vector["plaintext_hex"]
                results["aes_gcm_tests"].append(
                    {
                        "description": vector["description"],
                        "passed": passed,
                    }
                )

            ecdh_passed = sum(1 for t in results["ecdh_tests"] if t["passed"])
            aes_passed = sum(1 for t in results["aes_gcm_tests"] if t["passed"])

            return {
                "status": "success",
                "command": "RUN_TESTS",
                "results": results,
                "summary": {
                    "ecdh_passed": ecdh_passed,
                    "ecdh_total": len(results["ecdh_tests"]),
                    "aes_gcm_passed": aes_passed,
                    "aes_gcm_total": len(results["aes_gcm_tests"]),
                    "total_passed": ecdh_passed + aes_passed,
                    "total_tests": len(results["ecdh_tests"]) + len(results["aes_gcm_tests"]),
                    "all_passed": (
                        ecdh_passed == len(results["ecdh_tests"]) and aes_passed == len(results["aes_gcm_tests"])
                    ),
                },
            }
        except Exception as e:
            return {
                "status": "error",
                "command": "RUN_TESTS",
                "error": str(e),
            }

    def handle_run_agent(self, data: dict) -> dict:
        """
        Run an OpenClaw agent with an encrypted message (non-streaming).

        Supports dual encryption modes:
        - zero_trust: State encrypted to user's public key (default)
        - background: State encrypted with KMS envelope encryption (opt-in)

        Required fields:
        - encrypted_message: EncryptedPayload (user's message, encrypted to enclave key)
        - user_public_key: Hex string of user's public key (for response encryption)
        - agent_name: Name of the agent to run
        - model: LLM model to use

        Optional fields:
        - encrypted_state: EncryptedPayload (existing agent state tarball)
          If not provided, creates a fresh agent.
        - encryption_mode: "zero_trust" (default) or "background"

        Returns:
        - encrypted_response: Agent's response (encrypted to user's key)
        - encrypted_state: Updated agent state tarball
        - encrypted_dek: KMS-encrypted DEK (background mode only)
        """
        tmpfs_path = None
        try:
            # Extract parameters
            user_public_key = hex_to_bytes(data["user_public_key"])
            agent_name = data["agent_name"]
            model = data["model"]
            encrypted_state_dict = data.get("encrypted_state")
            encryption_mode = data.get("encryption_mode", "zero_trust")

            print(f"[Enclave] RUN_AGENT: agent={agent_name}, model={model}, mode={encryption_mode}", flush=True)

            # Create tmpfs directory for this request
            tmpfs_base = os.environ.get("OPENCLAW_TMPFS", "/tmp/openclaw")
            tmpfs_path = Path(tempfile.mkdtemp(dir=tmpfs_base, prefix=f"agent_{agent_name}_"))

            # Decrypt and extract existing state, or create fresh agent
            if encrypted_state_dict:
                state_bytes = self._decrypt_state(encrypted_state_dict, encryption_mode)
                self._unpack_tarball(state_bytes, tmpfs_path)
                print(f"[Enclave] Extracted existing state ({len(state_bytes)} bytes)", flush=True)
                self._log_tarball_contents(tmpfs_path)
            else:
                self._create_fresh_agent(tmpfs_path, agent_name, model)
                print("[Enclave] Created fresh agent directory", flush=True)
                self._log_tarball_contents(tmpfs_path)

            # Decrypt user message
            encrypted_message = EncryptedPayload.from_dict(data["encrypted_message"])
            message_bytes = decrypt_with_private_key(
                self.keypair.private_key,
                encrypted_message,
                "client-to-enclave-transport",
            )
            message = message_bytes.decode("utf-8")
            print(f"[Enclave] Decrypted message: {message[:50]}...", flush=True)

            # Run OpenClaw via Node.js bridge (same path as streaming)
            response_text = collect_response_text(
                run_agent_streaming(
                    state_dir=str(tmpfs_path),
                    agent_name=agent_name,
                    message=message,
                    provider="amazon-bedrock",
                    env=self._get_aws_env(),
                )
            )

            if not response_text:
                return {
                    "status": "error",
                    "command": "RUN_AGENT",
                    "error": "No response from agent bridge",
                }

            print(f"[Enclave] OpenClaw response: {response_text[:50]}...", flush=True)

            # Pack updated state
            tarball_bytes = self._pack_directory(tmpfs_path)
            print(f"[Enclave] Packed state: {len(tarball_bytes)} bytes", flush=True)

            # Encrypt state for storage
            state_result = self._encrypt_state(tarball_bytes, user_public_key, encryption_mode)

            # Encrypt response for transport (to user's key)
            encrypted_response = encrypt_to_public_key(
                user_public_key,
                response_text.encode("utf-8"),
                "enclave-to-client-transport",
            )

            return {
                "status": "success",
                "command": "RUN_AGENT",
                "encrypted_response": encrypted_response.to_dict(),
                "encrypted_state": state_result["encrypted_state"],
                "encrypted_dek": state_result["encrypted_dek"],
            }

        except KeyError as e:
            print(f"[Enclave] RUN_AGENT missing field: {e}", flush=True)
            return {
                "status": "error",
                "command": "RUN_AGENT",
                "error": f"Missing required field: {e}",
            }
        except Exception as e:
            print(f"[Enclave] RUN_AGENT error: {e}", flush=True)
            import traceback

            traceback.print_exc()
            return {
                "status": "error",
                "command": "RUN_AGENT",
                "error": str(e),
            }
        finally:
            # Always cleanup tmpfs
            if tmpfs_path and tmpfs_path.exists():
                shutil.rmtree(tmpfs_path, ignore_errors=True)
                print(f"[Enclave] Cleaned up tmpfs: {tmpfs_path}", flush=True)

    def _decrypt_state(self, encrypted_state_dict: dict, encryption_mode: str) -> bytes:
        """Decrypt agent state based on encryption mode.

        Args:
            encrypted_state_dict: Serialized encrypted state from vsock
            encryption_mode: "zero_trust" or "background"

        Returns:
            Decrypted state bytes (tarball)
        """
        if encryption_mode == "zero_trust":
            encrypted_state = EncryptedPayload.from_dict(encrypted_state_dict)
            return decrypt_with_private_key(
                self.keypair.private_key,
                encrypted_state,
                "client-to-enclave-transport",
            )
        else:
            kms_key_id = os.environ.get("KMS_KEY_ID", "")
            if not kms_key_id:
                raise ValueError("KMS_KEY_ID environment variable required for background mode")
            kms_envelope = {
                "encrypted_dek": hex_to_bytes(encrypted_state_dict["encrypted_dek"]),
                "iv": hex_to_bytes(encrypted_state_dict["iv"]),
                "ciphertext": hex_to_bytes(encrypted_state_dict["ciphertext"]),
                "auth_tag": hex_to_bytes(encrypted_state_dict["auth_tag"]),
            }
            state_bytes = decrypt_with_kms(kms_envelope, kms_key_id)
            print(f"[Enclave] Decrypted state from KMS ({len(state_bytes)} bytes)", flush=True)
            return state_bytes

    def _encrypt_state(self, tarball_bytes: bytes, user_public_key: bytes, encryption_mode: str) -> dict:
        """Encrypt agent state based on encryption mode.

        Args:
            tarball_bytes: Packed agent state tarball
            user_public_key: User's X25519 public key
            encryption_mode: "zero_trust" or "background"

        Returns:
            Dict with "encrypted_state" (serializable) and "encrypted_dek" (None)
        """
        if encryption_mode == "zero_trust":
            encrypted = encrypt_to_public_key(
                user_public_key,
                tarball_bytes,
                "agent-state-storage",
            )
            return {"encrypted_state": encrypted.to_dict(), "encrypted_dek": None}
        else:
            kms_key_id = os.environ.get("KMS_KEY_ID", "")
            if not kms_key_id:
                raise ValueError("KMS_KEY_ID environment variable required for background mode")
            kms_envelope = encrypt_with_kms(tarball_bytes, kms_key_id)
            return {
                "encrypted_state": {
                    "encrypted_dek": kms_envelope["encrypted_dek"].hex(),
                    "iv": kms_envelope["iv"].hex(),
                    "ciphertext": kms_envelope["ciphertext"].hex(),
                    "auth_tag": kms_envelope["auth_tag"].hex(),
                },
                "encrypted_dek": None,
            }

    def _unpack_tarball(self, tarball_bytes: bytes, target_dir: Path) -> None:
        """Unpack a gzip tarball to a directory."""
        target_dir.mkdir(parents=True, exist_ok=True)
        buffer = io.BytesIO(tarball_bytes)
        with tarfile.open(fileobj=buffer, mode="r:gz") as tar:
            # Security: Check for path traversal
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    raise ValueError(f"Unsafe path in tarball: {member.name}")
            tar.extractall(target_dir)

    def _pack_directory(self, directory: Path) -> bytes:
        """Pack a directory into a gzip tarball."""
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            for item in directory.rglob("*"):
                if item.is_file():
                    arcname = item.relative_to(directory)
                    tar.add(item, arcname=str(arcname))
        buffer.seek(0)
        return buffer.read()

    def _create_fresh_agent(self, agent_dir: Path, agent_name: str, model: str, soul_content: str = None) -> None:
        """Create a fresh OpenClaw agent directory structure."""
        agent_dir.mkdir(parents=True, exist_ok=True)

        # Use provided soul content or default
        if not soul_content:
            soul_content = f"""# {agent_name}

You are {agent_name}, a personal AI companion.

## Personality
- Friendly and helpful
- Remember past conversations
- Learn user preferences over time

## Guidelines
- Be concise but thorough
- Ask clarifying questions when needed
- Respect user privacy
"""

        # Create openclaw.json config
        config = {
            "version": "1.0",
            "agents": {agent_name: {"model": model}},
            "defaults": {"model": model, "agent": agent_name},
        }
        config_file = agent_dir / "openclaw.json"
        config_file.write_text(json.dumps(config, indent=2))

        # Create agent directory structure
        agent_subdir = agent_dir / "agents" / agent_name
        agent_subdir.mkdir(parents=True, exist_ok=True)

        # Create SOUL.md
        (agent_subdir / "SOUL.md").write_text(soul_content)

        # Create memory directory
        memory_dir = agent_subdir / "memory"
        memory_dir.mkdir(exist_ok=True)
        (memory_dir / "MEMORY.md").write_text("# Memories\n\nNo memories yet.\n")

        # Create sessions directory
        (agent_subdir / "sessions").mkdir(exist_ok=True)

    def _log_tarball_contents(self, agent_dir: Path) -> None:
        """Log the contents of an extracted agent tarball for diagnostics."""
        file_count = 0
        total_size = 0
        for root, dirs, files in os.walk(agent_dir):
            for f in files:
                filepath = os.path.join(root, f)
                size = os.path.getsize(filepath)
                rel = os.path.relpath(filepath, agent_dir)
                print(f"[Enclave] Tarball: {rel} ({size} bytes)", flush=True)
                file_count += 1
                total_size += size
        print(f"[Enclave] Tarball total: {file_count} files, {total_size} bytes", flush=True)

    def _read_agent_state(self, agent_dir: Path, agent_name: str) -> dict:
        """
        Read OpenClaw agent state files.

        Returns:
            {
                "model": str,
                "system_prompt": str,       # SOUL.md + MEMORY.md + daily memories
                "history": List[ConverseTurn],
                "session_file": Path,       # file to append new messages to
            }
        """
        from datetime import datetime, timedelta

        agent_subdir = agent_dir / "agents" / agent_name

        # --- Model resolution ---
        model = "us.anthropic.claude-opus-4-5-20251101-v1:0"
        config_file = agent_dir / "openclaw.json"
        if config_file.exists():
            try:
                config = json.loads(config_file.read_text())
                agent_config = config.get("agents", {}).get(agent_name, {})
                model = agent_config.get("model") or config.get("defaults", {}).get("model") or model
            except (json.JSONDecodeError, KeyError):
                pass

        # --- System prompt composition ---
        # 1. SOUL.md
        soul_content = ""
        soul_file = agent_subdir / "SOUL.md"
        if soul_file.exists():
            soul_content = soul_file.read_text().strip()

        # 2. MEMORY.md (long-term memories)
        memory_content = ""
        memory_file = agent_subdir / "memory" / "MEMORY.md"
        if memory_file.exists():
            memory_content = memory_file.read_text().strip()

        # 3. Daily memories (today + yesterday)
        daily_memories = ""
        memory_dir = agent_subdir / "memory"
        if memory_dir.exists():
            today = datetime.now()
            yesterday = today - timedelta(days=1)
            for day in [yesterday, today]:
                daily_file = memory_dir / f"{day.strftime('%Y-%m-%d')}.md"
                if daily_file.exists():
                    content = daily_file.read_text().strip()
                    if content:
                        daily_memories += f"\n### {day.strftime('%Y-%m-%d')}\n{content}\n"

        # Combine system prompt
        system_parts = []
        if soul_content:
            system_parts.append(soul_content)
        if memory_content:
            system_parts.append(f"## Memories\n{memory_content}")
        if daily_memories:
            system_parts.append(f"## Recent Notes{daily_memories}")

        system_prompt = "\n\n".join(system_parts) if system_parts else f"You are {agent_name}, a helpful AI assistant."

        # --- Session history parsing ---
        history: List[ConverseTurn] = []
        sessions_dir = agent_subdir / "sessions"
        session_file = None

        if sessions_dir.exists():
            # Find session files, sorted by name (timestamp-based)
            session_files = sorted(sessions_dir.glob("*.jsonl"))
            if session_files:
                # Use most recent session file
                latest_session = session_files[-1]
                session_file = latest_session

                try:
                    for line in latest_session.read_text().strip().split("\n"):
                        if not line.strip():
                            continue
                        try:
                            record = json.loads(line)
                            if record.get("type") == "message":
                                msg = record.get("message", {})
                                role = msg.get("role", "")
                                content_blocks = msg.get("content", [])
                                text = ""
                                for block in content_blocks:
                                    if isinstance(block, dict) and block.get("type") == "text":
                                        text += block.get("text", "")
                                    elif isinstance(block, str):
                                        text += block
                                if role in ("user", "assistant") and text:
                                    history.append(ConverseTurn(role=role, content=text))
                        except json.JSONDecodeError:
                            continue
                except Exception as e:
                    print(f"[Enclave] Error reading session file: {e}", flush=True)

        # If no session file, create one
        if session_file is None:
            sessions_dir.mkdir(parents=True, exist_ok=True)
            from datetime import datetime as dt

            timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
            session_file = sessions_dir / f"{timestamp}.jsonl"
            # Write session header
            header = json.dumps({"type": "session", "timestamp": timestamp, "agent": agent_name})
            session_file.write_text(header + "\n")

        return {
            "model": model,
            "system_prompt": system_prompt,
            "history": history,
            "session_file": session_file,
        }

    def _append_to_session(self, session_file: Path, role: str, content: str) -> None:
        """Append a message to a session JSONL file in OpenClaw format."""
        from datetime import datetime as dt

        record = {
            "type": "message",
            "timestamp": dt.now().isoformat(),
            "message": {
                "role": role,
                "content": [{"type": "text", "text": content}],
            },
        }
        with open(session_file, "a") as f:
            f.write(json.dumps(record) + "\n")

    def handle_extract_agent_files(self, data: dict) -> dict:
        """
        Extract files from a KMS-encrypted agent tarball.

        Decrypts the KMS envelope, extracts files, encrypts the file manifest
        to the user's transport key.

        Required fields:
        - encrypted_state: KMS envelope dict (hex strings)
        - user_public_key: Client's ephemeral transport public key (hex)
        """
        tmpfs_path = None
        try:
            encrypted_state_dict = data["encrypted_state"]
            user_public_key = hex_to_bytes(data["user_public_key"])

            print("[Enclave] EXTRACT_AGENT_FILES: decrypting KMS state", flush=True)

            # Decrypt state from KMS
            state_bytes = self._decrypt_state(encrypted_state_dict, "background")

            # Extract tarball to tmpfs
            tmpfs_base = os.environ.get("OPENCLAW_TMPFS", "/tmp/openclaw")
            os.makedirs(tmpfs_base, exist_ok=True)
            tmpfs_path = Path(tempfile.mkdtemp(dir=tmpfs_base, prefix="extract_"))
            self._unpack_tarball(state_bytes, tmpfs_path)

            # Read all files
            file_list = []
            for item in tmpfs_path.rglob("*"):
                if item.is_file():
                    rel_path = str(item.relative_to(tmpfs_path))
                    try:
                        content = item.read_text(encoding="utf-8")
                    except UnicodeDecodeError:
                        content = item.read_bytes().hex()
                    file_list.append({"path": rel_path, "content": content})

            print(f"[Enclave] Extracted {len(file_list)} files", flush=True)

            # Encrypt file manifest to user's transport key
            manifest_json = json.dumps(file_list).encode("utf-8")
            encrypted_files = encrypt_to_public_key(
                user_public_key,
                manifest_json,
                "enclave-to-client-transport",
            )

            return {
                "status": "success",
                "command": "EXTRACT_AGENT_FILES",
                "encrypted_files": encrypted_files.to_dict(),
            }

        except Exception as e:
            print(f"[Enclave] EXTRACT_AGENT_FILES error: {e}", flush=True)
            import traceback

            traceback.print_exc()
            return {
                "status": "error",
                "command": "EXTRACT_AGENT_FILES",
                "error": str(e),
            }
        finally:
            if tmpfs_path and tmpfs_path.exists():
                shutil.rmtree(tmpfs_path, ignore_errors=True)

    def handle_pack_agent_files(self, data: dict) -> dict:
        """
        Pack files into a new KMS-encrypted agent tarball.

        Decrypts each file's content (encrypted to enclave transport key),
        writes them to a tmpfs directory, packs into a tarball, and
        KMS-encrypts the result.

        Required fields:
        - files: List of {path, encrypted_content (EncryptedPayload dict)}
        """
        tmpfs_path = None
        try:
            files = data["files"]

            print(f"[Enclave] PACK_AGENT_FILES: packing {len(files)} files", flush=True)

            # Create tmpfs directory
            tmpfs_base = os.environ.get("OPENCLAW_TMPFS", "/tmp/openclaw")
            os.makedirs(tmpfs_base, exist_ok=True)
            tmpfs_path = Path(tempfile.mkdtemp(dir=tmpfs_base, prefix="pack_"))

            # Decrypt and write each file
            for f in files:
                rel_path = f["path"]
                encrypted_content = EncryptedPayload.from_dict(f["encrypted_content"])
                content_bytes = decrypt_with_private_key(
                    self.keypair.private_key,
                    encrypted_content,
                    "client-to-enclave-transport",
                )

                file_path = tmpfs_path / rel_path
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_bytes(content_bytes)

            # Pack directory into tarball
            tarball_bytes = self._pack_directory(tmpfs_path)
            print(f"[Enclave] Packed tarball: {len(tarball_bytes)} bytes", flush=True)

            # KMS-encrypt the tarball
            kms_key_id = os.environ.get("KMS_KEY_ID", "")
            if not kms_key_id:
                raise ValueError("KMS_KEY_ID required for background mode")

            kms_envelope = encrypt_with_kms(tarball_bytes, kms_key_id)
            kms_envelope_hex = {
                "encrypted_dek": kms_envelope["encrypted_dek"].hex(),
                "iv": kms_envelope["iv"].hex(),
                "ciphertext": kms_envelope["ciphertext"].hex(),
                "auth_tag": kms_envelope["auth_tag"].hex(),
            }

            return {
                "status": "success",
                "command": "PACK_AGENT_FILES",
                "kms_envelope": kms_envelope_hex,
            }

        except Exception as e:
            print(f"[Enclave] PACK_AGENT_FILES error: {e}", flush=True)
            import traceback

            traceback.print_exc()
            return {
                "status": "error",
                "command": "PACK_AGENT_FILES",
                "error": str(e),
            }
        finally:
            if tmpfs_path and tmpfs_path.exists():
                shutil.rmtree(tmpfs_path, ignore_errors=True)

    def handle_agent_chat_stream(self, data: dict, conn: socket.socket) -> None:
        """
        Process encrypted agent chat with streaming response.

        Supports dual encryption modes:
        - zero_trust: State encrypted to user's public key (default)
        - background: State encrypted with KMS envelope encryption (opt-in)

        Streams newline-delimited JSON events:
        - {"encrypted_content": {...}}  - Encrypted chunk for client
        - {"is_final": true, "encrypted_state": {...}, "encrypted_dek": {...}} - Final event
        - {"error": "...", "is_final": true} - Error event
        """
        tmpfs_path = None
        try:
            # Check credentials
            if not self.bedrock.has_credentials():
                self._send_event(conn, {"error": "No AWS credentials", "is_final": True})
                return

            # Extract parameters
            encrypted_message_dict = data["encrypted_message"]
            encrypted_state_dict = data.get("encrypted_state")
            client_public_key = hex_to_bytes(data["client_public_key"])
            user_public_key = hex_to_bytes(data["user_public_key"])
            agent_name = data["agent_name"]
            encrypted_soul_dict = data.get("encrypted_soul_content")
            encryption_mode = data.get("encryption_mode", "zero_trust")  # Default to zero_trust

            print(f"[Enclave] AGENT_CHAT_STREAM: agent={agent_name}, mode={encryption_mode}", flush=True)

            # Create tmpfs directory
            tmpfs_base = os.environ.get("OPENCLAW_TMPFS", "/tmp/openclaw")
            os.makedirs(tmpfs_base, exist_ok=True)
            tmpfs_path = Path(tempfile.mkdtemp(dir=tmpfs_base, prefix=f"agent_{agent_name}_"))

            # Decrypt and extract existing state, or create fresh agent
            if encrypted_state_dict:
                state_bytes = self._decrypt_state(encrypted_state_dict, encryption_mode)
                self._unpack_tarball(state_bytes, tmpfs_path)
                print(f"[Enclave] Extracted existing state ({len(state_bytes)} bytes)", flush=True)
                self._log_tarball_contents(tmpfs_path)
            else:
                # Decrypt soul content if provided (encrypted by client to enclave key)
                soul_content = None
                if encrypted_soul_dict:
                    encrypted_soul = EncryptedPayload.from_dict(encrypted_soul_dict)
                    soul_bytes = decrypt_with_private_key(
                        self.keypair.private_key,
                        encrypted_soul,
                        "client-to-enclave-transport",
                    )
                    soul_content = soul_bytes.decode("utf-8")
                    print(f"[Enclave] Decrypted soul content ({len(soul_content)} chars)", flush=True)

                default_model = "us.anthropic.claude-opus-4-5-20251101-v1:0"
                self._create_fresh_agent(tmpfs_path, agent_name, default_model, soul_content)
                print("[Enclave] Created fresh agent directory", flush=True)
                self._log_tarball_contents(tmpfs_path)

            # Decrypt user message
            encrypted_message = EncryptedPayload.from_dict(encrypted_message_dict)
            message_bytes = decrypt_with_private_key(
                self.keypair.private_key,
                encrypted_message,
                "client-to-enclave-transport",
            )
            user_content = message_bytes.decode("utf-8")
            print(f"[Enclave] Decrypted message: {user_content[:50]}...", flush=True)

            # Run OpenClaw agent via Node.js bridge
            # The bridge handles: model selection, session management, history,
            # tool execution, and memory — all via runEmbeddedPiAgent()
            chunk_count = 0
            input_tokens = 0
            output_tokens = 0
            agent_error = None
            event_types_seen = []  # Track all event types for diagnostics
            last_block_text = ""  # Fallback: capture block text if no partials
            done_meta_raw = {}  # Capture raw done metadata for diagnostics
            done_result_text = ""  # Capture result text from done event
            agent_events_data = []  # Capture agent_event payloads
            bridge_stderr = ""  # Capture bridge stderr output

            print("[Enclave] Starting OpenClaw agent bridge...", flush=True)

            try:
                for event in run_agent_streaming(
                    state_dir=str(tmpfs_path),
                    agent_name=agent_name,
                    message=user_content,
                    provider="amazon-bedrock",
                    env=self._get_aws_env(),
                ):
                    event_type = event.get("type")
                    event_types_seen.append(event_type)

                    if event_type == "partial":
                        # Token-by-token streaming — encrypt and forward to client
                        chunk_text = event.get("text", "")
                        if chunk_text:
                            chunk_count += 1
                            encrypted_chunk = encrypt_to_public_key(
                                client_public_key,
                                chunk_text.encode("utf-8"),
                                "enclave-to-client-transport",
                            )
                            self._send_event(conn, {"encrypted_content": encrypted_chunk.to_dict()})

                    elif event_type == "block":
                        # Accumulated text block — capture as fallback if no partials sent
                        block_text = event.get("text", "")
                        if block_text:
                            last_block_text = block_text
                            print(f"[Enclave] Block event: {len(block_text)} chars", flush=True)

                    elif event_type == "tool_result":
                        # Tool execution results — encrypt and forward
                        tool_text = event.get("text", "")
                        if tool_text:
                            encrypted_chunk = encrypt_to_public_key(
                                client_public_key,
                                tool_text.encode("utf-8"),
                                "enclave-to-client-transport",
                            )
                            self._send_event(
                                conn, {"encrypted_content": encrypted_chunk.to_dict(), "event_type": "tool_result"}
                            )

                    elif event_type == "error":
                        # Agent-level error (exit 0, error via NDJSON)
                        agent_error = event.get("message", "Unknown agent error")
                        print(f"[Enclave] Agent error: {agent_error}", flush=True)

                    elif event_type == "done":
                        # Completion with metadata
                        meta = event.get("meta", {})
                        duration_ms = meta.get("durationMs", 0)
                        stop_reason = meta.get("stopReason", "unknown")
                        # Extract token usage from agentMeta.usage
                        agent_meta = meta.get("agentMeta") or {}
                        usage = agent_meta.get("usage") or {}
                        input_tokens = usage.get("input", 0) or 0
                        output_tokens = usage.get("output", 0) or 0
                        meta_error = meta.get("error")
                        if meta_error:
                            if isinstance(meta_error, dict):
                                agent_error = meta_error.get("message", str(meta_error))
                            else:
                                agent_error = str(meta_error)
                        # Capture raw meta and result text for diagnostics
                        done_meta_raw = {
                            "durationMs": duration_ms,
                            "stopReason": stop_reason,
                            "error": str(meta_error) if meta_error else None,
                            "inputTokens": input_tokens,
                            "outputTokens": output_tokens,
                            "agentMetaKeys": list((agent_meta or {}).keys()),
                        }
                        done_result_text = event.get("resultText", "")
                        result_keys = event.get("resultKeys", [])
                        print(
                            f"[Enclave] Agent done: {duration_ms}ms, stop={stop_reason}, tokens={input_tokens}/{output_tokens}, resultText_len={len(done_result_text)}, resultKeys={result_keys}",
                            flush=True,
                        )

                    elif event_type == "agent_event":
                        # Capture agent lifecycle events for diagnostics
                        agent_events_data.append(
                            {"stream": event.get("stream"), "data_keys": list((event.get("data") or {}).keys())}
                        )

                    elif event_type == "bridge_stderr":
                        # Capture bridge stderr output
                        bridge_stderr = event.get("text", "")
                        print(f"[Enclave] Bridge stderr: {bridge_stderr[:5000]}", flush=True)

                    # Skip: assistant_start, partial_empty, block_empty, reasoning

            except (RuntimeError, FileNotFoundError) as e:
                print(f"[Enclave] Bridge error: {e}", flush=True)
                # Send diagnostic info even on error path
                if bridge_stderr:
                    self._send_event(
                        conn,
                        {
                            "diagnostic": {
                                "bridge_stderr": bridge_stderr[:5000],
                                "error": str(e),
                            }
                        },
                    )
                self._send_event(conn, {"error": str(e), "is_final": True})
                return

            if agent_error:
                self._send_event(conn, {"error": agent_error, "is_final": True})
                return

            # Send diagnostic info through vsock so parent can log it
            from collections import Counter

            evt_counts = dict(Counter(event_types_seen))
            self._send_event(
                conn,
                {
                    "diagnostic": {
                        "chunk_count": chunk_count,
                        "event_types": evt_counts,
                        "has_block_fallback": chunk_count == 0 and bool(last_block_text),
                        "block_text_len": len(last_block_text),
                        "done_meta": done_meta_raw,
                        "done_result_text_len": len(done_result_text),
                        "done_result_text_preview": done_result_text[:200] if done_result_text else "",
                        "agent_events": agent_events_data,
                        "bridge_stderr": bridge_stderr[:4000] if bridge_stderr else "",
                    }
                },
            )

            # Fallback chain: try block text, then done result text
            fallback_text = last_block_text or done_result_text

            # Fallback: if no partial events were streamed but we have text from
            # block events or the done result, send it as a single encrypted chunk
            if chunk_count == 0 and fallback_text:
                source = "block" if last_block_text else "done_result"
                print(
                    f"[Enclave] No partials streamed, using {source} fallback ({len(fallback_text)} chars)", flush=True
                )
                encrypted_chunk = encrypt_to_public_key(
                    client_public_key,
                    fallback_text.encode("utf-8"),
                    "enclave-to-client-transport",
                )
                self._send_event(conn, {"encrypted_content": encrypted_chunk.to_dict()})
                chunk_count = 1

            print(f"[Enclave] Stream complete: {chunk_count} chunks", flush=True)

            # Pack updated state
            tarball_bytes = self._pack_directory(tmpfs_path)
            print(f"[Enclave] Packed state: {len(tarball_bytes)} bytes", flush=True)

            # Encrypt state for storage
            state_result = self._encrypt_state(tarball_bytes, user_public_key, encryption_mode)

            # Send final event with updated state
            self._send_event(
                conn,
                {
                    "is_final": True,
                    "encrypted_state": state_result["encrypted_state"],
                    "encrypted_dek": state_result["encrypted_dek"],
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                },
            )

            print("[Enclave] AGENT_CHAT_STREAM complete", flush=True)

        except KeyError as e:
            print(f"[Enclave] AGENT_CHAT_STREAM missing field: {e}", flush=True)
            self._send_event(conn, {"error": f"Missing field: {e}", "is_final": True})

        except Exception as e:
            print(f"[Enclave] AGENT_CHAT_STREAM error: {e}", flush=True)
            import traceback

            traceback.print_exc()
            self._send_event(conn, {"error": str(e), "is_final": True})

        finally:
            # Always cleanup tmpfs
            if tmpfs_path and tmpfs_path.exists():
                shutil.rmtree(tmpfs_path, ignore_errors=True)
                print(f"[Enclave] Cleaned up tmpfs: {tmpfs_path}", flush=True)

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
            history.append(
                ConverseTurn(
                    role="assistant" if is_assistant else "user",
                    content=plaintext.decode("utf-8"),
                )
            )
        return history

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
            stream_start = time.time()

            for event in self.bedrock.converse_stream(model_id, messages, system, inference_config):
                event_time = time.time()
                if event["type"] == "content":
                    chunk_text = event["text"]
                    full_response += chunk_text
                    chunk_count += 1

                    print(
                        f"[Enclave] Chunk #{chunk_count} received at {event_time:.3f} (+{event_time - stream_start:.3f}s)",
                        flush=True,
                    )

                    # Encrypt chunk for transport to client
                    encrypted_chunk = encrypt_to_public_key(
                        client_public_key,
                        chunk_text.encode("utf-8"),
                        "enclave-to-client-transport",
                    )
                    self._send_event(conn, {"encrypted_content": encrypted_chunk.to_dict()})
                    send_time = time.time()
                    print(
                        f"[Enclave] Chunk #{chunk_count} sent at {send_time:.3f} (encrypt+send took {send_time - event_time:.3f}s)",
                        flush=True,
                    )

                elif event["type"] == "reasoning":
                    # Reasoning/thinking tokens from models like Kimi K2 Thinking, DeepSeek R1
                    reasoning_text = event["text"]
                    encrypted_thinking = encrypt_to_public_key(
                        client_public_key,
                        reasoning_text.encode("utf-8"),
                        "enclave-to-client-transport",
                    )
                    self._send_event(conn, {"encrypted_thinking": encrypted_thinking.to_dict()})

                elif event["type"] == "metadata":
                    input_tokens = event["usage"].get("inputTokens", 0)
                    output_tokens = event["usage"].get("outputTokens", 0)
                    print(f"[Enclave] Metadata received at {event_time:.3f}", flush=True)

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
            self._send_event(
                conn,
                {
                    "is_final": True,
                    "stored_user_message": stored_user.to_dict(),
                    "stored_assistant_message": stored_assistant.to_dict(),
                    "model_used": model_id,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                },
            )

            print("[Enclave] CHAT_STREAM complete", flush=True)

        except KeyError as e:
            print(f"[Enclave] CHAT_STREAM missing field: {e}", flush=True)
            self._send_event(conn, {"error": f"Missing field: {e}", "is_final": True})

        except Exception as e:
            print(f"[Enclave] CHAT_STREAM error: {e}", flush=True)
            import traceback

            traceback.print_exc()
            self._send_event(conn, {"error": str(e), "is_final": True})

    def handle_request(self, request: dict, conn: socket.socket) -> dict:
        """Route request to appropriate handler."""
        command = request.get("command", "").upper()

        # Streaming commands handle their own response
        if command == "CHAT_STREAM":
            self.handle_chat_stream(request, conn)
            return None  # Response already sent
        if command == "AGENT_CHAT_STREAM":
            self.handle_agent_chat_stream(request, conn)
            return None  # Response already sent

        # Non-streaming commands
        handlers = {
            "GET_PUBLIC_KEY": self.handle_get_public_key,
            "SET_CREDENTIALS": lambda: self.handle_set_credentials(request),
            "HEALTH": self.handle_health,
            "CHAT": lambda: self.handle_chat(request),
            "RUN_TESTS": self.handle_run_tests,
            "RUN_AGENT": lambda: self.handle_run_agent(request),
            "EXTRACT_AGENT_FILES": lambda: self.handle_extract_agent_files(request),
            "PACK_AGENT_FILES": lambda: self.handle_pack_agent_files(request),
        }

        handler = handlers.get(command)
        if handler:
            return handler()
        else:
            return {
                "status": "error",
                "error": f"Unknown command: {command}",
                "available_commands": list(handlers.keys()) + ["CHAT_STREAM", "AGENT_CHAT_STREAM"],
            }


def create_vsock_listener(port: int) -> socket.socket:
    """Create a vsock listener socket."""
    sock = socket.socket(AF_VSOCK, socket.SOCK_STREAM)
    sock.bind((socket.VMADDR_CID_ANY, port))
    sock.listen(5)
    return sock


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


def _start_vsock_tcp_bridge():
    """Start the TCP-to-vsock bridge subprocess for Node.js networking."""
    bridge_script = Path(__file__).parent / "vsock_tcp_bridge.py"
    if not bridge_script.exists():
        print(f"[Enclave] vsock_tcp_bridge.py not found at {bridge_script}, skipping", flush=True)
        return None

    print("[Enclave] Starting TCP-to-vsock bridge...", flush=True)
    proc = subprocess.Popen(
        [sys.executable, str(bridge_script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    # Read the "Ready" line to confirm it started
    import select as _sel

    if _sel.select([proc.stdout], [], [], 5.0)[0]:
        for _ in range(3):
            line = proc.stdout.readline().decode("utf-8", errors="replace").strip()
            if line:
                print(f"[Enclave] bridge: {line}", flush=True)
            if "Ready" in line:
                break
    # Let remaining output go to /dev/null (daemon threads handle logging)
    import threading

    def _drain(pipe):
        for line in pipe:
            txt = line.decode("utf-8", errors="replace").strip()
            if txt:
                print(f"[Enclave] bridge: {txt}", flush=True)

    threading.Thread(target=_drain, args=(proc.stdout,), daemon=True).start()
    print(f"[Enclave] TCP-to-vsock bridge started (PID {proc.pid})", flush=True)
    return proc


def main():
    region = os.environ.get("AWS_REGION", "us-east-1")

    print("=" * 60, flush=True)
    print("NITRO ENCLAVE BEDROCK SERVER (M4)", flush=True)
    print("=" * 60, flush=True)
    print(f"Python version: {sys.version}", flush=True)
    print(f"AWS region: {region}", flush=True)

    # Start TCP-to-vsock bridge for Node.js (OpenClaw) networking
    _start_vsock_tcp_bridge()

    server = BedrockServer(region=region)

    print(f"Listening on vsock port {VSOCK_PORT}...", flush=True)
    print("[Enclave] Waiting for SET_CREDENTIALS from parent...", flush=True)

    try:
        listener = create_vsock_listener(VSOCK_PORT)
        print("[Enclave] Server ready, waiting for connections...", flush=True)

        while True:
            conn, addr = listener.accept()
            handle_client(server, conn, addr)

    except Exception as e:
        print(f"[Enclave] Fatal error: {e}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
