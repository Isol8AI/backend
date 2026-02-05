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

            print("[Enclave] AWS credentials set", flush=True)
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

    def handle_chat(self, data: dict) -> dict:
        """
        Process an encrypted chat message through Bedrock.

        Required fields:
        - encrypted_message: EncryptedPayload (encrypted to enclave key)
        - user_public_key: Hex string of user's storage public key
        - model_id: Model identifier from frontend (e.g., "us.anthropic.claude-3-5-haiku-20241022-v1:0")

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
        Run an OpenClaw agent with an encrypted message.

        Required fields:
        - encrypted_message: EncryptedPayload (user's message, encrypted to enclave key)
        - user_public_key: Hex string of user's public key (for response encryption)
        - agent_name: Name of the agent to run
        - model: LLM model to use

        Optional fields:
        - encrypted_state: EncryptedPayload (existing agent state tarball)
          If not provided, creates a fresh agent.

        Returns:
        - encrypted_response: Agent's response (encrypted to user's key)
        - encrypted_state: Updated agent state tarball (encrypted to enclave's key)
        """
        tmpfs_path = None
        try:
            # Extract parameters
            user_public_key = hex_to_bytes(data["user_public_key"])
            agent_name = data["agent_name"]
            model = data["model"]
            encrypted_state_dict = data.get("encrypted_state")

            print(f"[Enclave] RUN_AGENT: agent={agent_name}, model={model}", flush=True)

            # Create tmpfs directory for this request
            tmpfs_base = os.environ.get("OPENCLAW_TMPFS", "/tmp/openclaw")
            tmpfs_path = Path(tempfile.mkdtemp(dir=tmpfs_base, prefix=f"agent_{agent_name}_"))
            print(f"[Enclave] Using tmpfs: {tmpfs_path}", flush=True)

            # Decrypt and extract existing state, or create fresh agent
            if encrypted_state_dict:
                encrypted_state = EncryptedPayload.from_dict(encrypted_state_dict)
                state_bytes = decrypt_with_private_key(
                    self.keypair.private_key,
                    encrypted_state,
                    "agent-state-storage",
                )
                self._unpack_tarball(state_bytes, tmpfs_path)
                print(f"[Enclave] Extracted existing state ({len(state_bytes)} bytes)", flush=True)
            else:
                self._create_fresh_agent(tmpfs_path, agent_name, model)
                print("[Enclave] Created fresh agent directory", flush=True)

            # Decrypt user message
            encrypted_message = EncryptedPayload.from_dict(data["encrypted_message"])
            message_bytes = decrypt_with_private_key(
                self.keypair.private_key,
                encrypted_message,
                "client-to-enclave-transport",
            )
            message = message_bytes.decode("utf-8")
            print(f"[Enclave] Decrypted message: {message[:50]}...", flush=True)

            # Run OpenClaw CLI
            result = self._run_openclaw(tmpfs_path, message, agent_name, model)

            if not result["success"]:
                return {
                    "status": "error",
                    "command": "RUN_AGENT",
                    "error": result["error"],
                }

            print(f"[Enclave] OpenClaw response: {result['response'][:50]}...", flush=True)

            # Pack updated state
            tarball_bytes = self._pack_directory(tmpfs_path)
            print(f"[Enclave] Packed state: {len(tarball_bytes)} bytes", flush=True)

            # Encrypt state for storage (to enclave's key for future decryption)
            encrypted_state_out = encrypt_to_public_key(
                self.keypair.public_key,
                tarball_bytes,
                "agent-state-storage",
            )

            # Encrypt response for transport (to user's key)
            encrypted_response = encrypt_to_public_key(
                user_public_key,
                result["response"].encode("utf-8"),
                "enclave-to-client-transport",
            )

            return {
                "status": "success",
                "command": "RUN_AGENT",
                "encrypted_response": encrypted_response.to_dict(),
                "encrypted_state": encrypted_state_out.to_dict(),
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
        model = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
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

    def handle_agent_chat_stream(self, data: dict, conn: socket.socket) -> None:
        """
        Process encrypted agent chat with streaming response.

        Streams newline-delimited JSON events:
        - {"encrypted_content": {...}}  - Encrypted chunk for client
        - {"is_final": true, "encrypted_state": {...}} - Final event with updated state
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
            agent_name = data["agent_name"]
            encrypted_soul_dict = data.get("encrypted_soul_content")

            print(f"[Enclave] AGENT_CHAT_STREAM: agent={agent_name}", flush=True)

            # Create tmpfs directory
            tmpfs_base = os.environ.get("OPENCLAW_TMPFS", "/tmp/openclaw")
            os.makedirs(tmpfs_base, exist_ok=True)
            tmpfs_path = Path(tempfile.mkdtemp(dir=tmpfs_base, prefix=f"agent_{agent_name}_"))

            # Decrypt and extract existing state, or create fresh agent
            if encrypted_state_dict:
                encrypted_state = EncryptedPayload.from_dict(encrypted_state_dict)
                state_bytes = decrypt_with_private_key(
                    self.keypair.private_key,
                    encrypted_state,
                    "agent-state-storage",
                )
                self._unpack_tarball(state_bytes, tmpfs_path)
                print(f"[Enclave] Extracted existing state ({len(state_bytes)} bytes)", flush=True)
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

                default_model = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
                self._create_fresh_agent(tmpfs_path, agent_name, default_model, soul_content)
                print("[Enclave] Created fresh agent directory", flush=True)

            # Decrypt user message
            encrypted_message = EncryptedPayload.from_dict(encrypted_message_dict)
            message_bytes = decrypt_with_private_key(
                self.keypair.private_key,
                encrypted_message,
                "client-to-enclave-transport",
            )
            user_content = message_bytes.decode("utf-8")
            print(f"[Enclave] Decrypted message: {user_content[:50]}...", flush=True)

            # Read agent state (SOUL.md, memory, session history)
            agent_state = self._read_agent_state(tmpfs_path, agent_name)
            model_id = agent_state["model"]
            system_prompt = agent_state["system_prompt"]
            history = agent_state["history"]
            session_file = agent_state["session_file"]

            print(f"[Enclave] Agent model: {model_id}", flush=True)
            print(f"[Enclave] History: {len(history)} messages", flush=True)

            # Build Bedrock messages
            messages = build_converse_messages(history, user_content)
            system = [{"text": system_prompt}]
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

            # Update session JSONL with user message and assistant response
            self._append_to_session(session_file, "user", user_content)
            self._append_to_session(session_file, "assistant", full_response)

            # TODO (Phase B): Memory flush — extract key information from conversation
            # and append to memory/YYYY-MM-DD.md

            # Pack updated state
            tarball_bytes = self._pack_directory(tmpfs_path)
            print(f"[Enclave] Packed state: {len(tarball_bytes)} bytes", flush=True)

            # Encrypt state for storage (to enclave's key for future decryption)
            encrypted_state_out = encrypt_to_public_key(
                self.keypair.public_key,
                tarball_bytes,
                "agent-state-storage",
            )

            # Send final event with updated state
            self._send_event(
                conn,
                {
                    "is_final": True,
                    "encrypted_state": encrypted_state_out.to_dict(),
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

    def _run_openclaw(self, agent_dir: Path, message: str, agent_name: str, model: str, timeout: int = 120) -> dict:
        """Run the OpenClaw CLI with a message."""
        env = os.environ.copy()
        env["OPENCLAW_STATE_DIR"] = str(agent_dir)
        env["OPENCLAW_HOME"] = str(agent_dir)
        env["HOME"] = str(agent_dir)

        cmd = [
            "openclaw",
            "agent",
            "--message",
            message,
            "--agent",
            agent_name,
            "--model",
            model,
            "--non-interactive",
        ]

        print(f"[Enclave] Running: {' '.join(cmd)}", flush=True)

        try:
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(agent_dir),
            )

            if result.returncode == 0:
                return {
                    "success": True,
                    "response": result.stdout.strip(),
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            else:
                return {
                    "success": False,
                    "error": result.stderr.strip() or f"Exit code: {result.returncode}",
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Command timed out after {timeout} seconds",
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

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


def main():
    region = os.environ.get("AWS_REGION", "us-east-1")

    print("=" * 60, flush=True)
    print("NITRO ENCLAVE BEDROCK SERVER (M4)", flush=True)
    print("=" * 60, flush=True)
    print(f"Python version: {sys.version}", flush=True)
    print(f"AWS region: {region}", flush=True)

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
