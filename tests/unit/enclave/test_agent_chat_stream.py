"""
Tests for BedrockServer agent chat stream methods.

Since bedrock_server.py lives in the enclave and imports enclave-only modules
(crypto_primitives, bedrock_client) that are not on the normal Python path,
we cannot import it directly. Instead, we test the pure filesystem logic
(_read_agent_state, _append_to_session) by:

1. Mocking the enclave-only imports so BedrockServer can be instantiated
2. Testing the filesystem parsing/writing logic directly

For handle_agent_chat_stream, we verify command routing in handle_request.
"""

import json
import sys
import types
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

import pytest


# ---------------------------------------------------------------------------
# Mock enclave-only modules so we can import bedrock_server
# ---------------------------------------------------------------------------

@dataclass
class _FakeConverseTurn:
    """Stand-in for bedrock_client.ConverseTurn used in _read_agent_state."""
    role: str
    content: str


def _build_enclave_mocks():
    """
    Create mock modules for crypto_primitives and bedrock_client so that
    ``import bedrock_server`` succeeds in the test environment.

    Returns the mocked bedrock_client module so tests can reference
    ConverseTurn.
    """
    # --- crypto_primitives mock ---
    crypto_mod = types.ModuleType("crypto_primitives")
    fake_keypair = MagicMock()
    fake_keypair.public_key = b"\x00" * 32
    fake_keypair.private_key = b"\x01" * 32
    crypto_mod.generate_x25519_keypair = MagicMock(return_value=fake_keypair)
    crypto_mod.encrypt_to_public_key = MagicMock()
    crypto_mod.decrypt_with_private_key = MagicMock()
    crypto_mod.EncryptedPayload = MagicMock()
    crypto_mod.KeyPair = MagicMock()
    crypto_mod.bytes_to_hex = lambda b: b.hex() if isinstance(b, bytes) else str(b)
    crypto_mod.hex_to_bytes = bytes.fromhex

    # --- bedrock_client mock ---
    bedrock_mod = types.ModuleType("bedrock_client")
    bedrock_mod.ConverseTurn = _FakeConverseTurn

    mock_bedrock_class = MagicMock()
    mock_bedrock_instance = MagicMock()
    mock_bedrock_instance.has_credentials.return_value = True
    mock_bedrock_class.return_value = mock_bedrock_instance
    bedrock_mod.BedrockClient = mock_bedrock_class
    bedrock_mod.BedrockResponse = MagicMock()
    bedrock_mod.build_converse_messages = MagicMock(return_value=[])

    # --- vsock_http_client mock (imported transitively) ---
    vsock_mod = types.ModuleType("vsock_http_client")
    vsock_mod.VsockHttpClient = MagicMock()

    return crypto_mod, bedrock_mod, vsock_mod


# Install mocks before importing bedrock_server
_crypto_mod, _bedrock_mod, _vsock_mod = _build_enclave_mocks()
sys.modules["crypto_primitives"] = _crypto_mod
sys.modules["bedrock_client"] = _bedrock_mod
sys.modules["vsock_http_client"] = _vsock_mod

# Patch socket.AF_VSOCK which does not exist on macOS/standard Linux
_real_socket = sys.modules.get("socket")

# Now import the module under test
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "enclave"))
from bedrock_server import BedrockServer  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_server() -> BedrockServer:
    """Instantiate a BedrockServer with mocked dependencies."""
    server = BedrockServer.__new__(BedrockServer)
    server.keypair = MagicMock()
    server.keypair.public_key = b"\x00" * 32
    server.keypair.private_key = b"\x01" * 32
    server.bedrock = MagicMock()
    server.bedrock.has_credentials.return_value = True
    server.region = "us-east-1"
    return server


def _create_agent_dir(
    tmp_path: Path,
    agent_name: str = "luna",
    *,
    model: str = "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    soul_content: str | None = None,
    memory_content: str | None = None,
    daily_memories: dict[str, str] | None = None,
    session_records: list[dict] | None = None,
    create_sessions_dir: bool = True,
    config_json: dict | None = None,
) -> Path:
    """
    Build an OpenClaw agent directory tree under ``tmp_path`` and return it.

    Parameters mirror the structure read by ``_read_agent_state``.
    """
    agent_dir = tmp_path

    # openclaw.json
    if config_json is not None:
        (agent_dir / "openclaw.json").write_text(json.dumps(config_json))
    else:
        cfg = {
            "version": "1.0",
            "agents": {agent_name: {"model": model}},
            "defaults": {"model": model, "agent": agent_name},
        }
        (agent_dir / "openclaw.json").write_text(json.dumps(cfg))

    # agents/<name>/
    agent_subdir = agent_dir / "agents" / agent_name
    agent_subdir.mkdir(parents=True, exist_ok=True)

    # SOUL.md
    if soul_content is not None:
        (agent_subdir / "SOUL.md").write_text(soul_content)

    # memory/
    memory_dir = agent_subdir / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)

    if memory_content is not None:
        (memory_dir / "MEMORY.md").write_text(memory_content)

    if daily_memories:
        for date_str, content in daily_memories.items():
            (memory_dir / f"{date_str}.md").write_text(content)

    # sessions/
    if create_sessions_dir:
        sessions_dir = agent_subdir / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        if session_records is not None:
            # Write all records to a single session file
            lines = [json.dumps(r) for r in session_records]
            (sessions_dir / "20260201_120000.jsonl").write_text("\n".join(lines) + "\n")

    return agent_dir


# ===========================================================================
# Tests for _read_agent_state
# ===========================================================================


class TestReadAgentStateModel:
    """Model resolution from openclaw.json."""

    def test_reads_model_from_agent_config(self, tmp_path):
        agent_dir = _create_agent_dir(tmp_path, model="my-custom-model")
        server = _make_server()

        result = server._read_agent_state(agent_dir, "luna")

        assert result["model"] == "my-custom-model"

    def test_falls_back_to_defaults_model(self, tmp_path):
        """When agent-specific model is missing, uses defaults.model."""
        cfg = {
            "version": "1.0",
            "agents": {"luna": {}},
            "defaults": {"model": "default-model", "agent": "luna"},
        }
        agent_dir = _create_agent_dir(tmp_path, config_json=cfg)
        server = _make_server()

        result = server._read_agent_state(agent_dir, "luna")

        assert result["model"] == "default-model"

    def test_falls_back_to_hardcoded_default_when_config_missing(self, tmp_path):
        """When openclaw.json does not exist, uses hardcoded default model."""
        agent_dir = _create_agent_dir(tmp_path)
        # Remove the config file
        (agent_dir / "openclaw.json").unlink()
        server = _make_server()

        result = server._read_agent_state(agent_dir, "luna")

        assert result["model"] == "anthropic.claude-sonnet-4-20250514"

    def test_falls_back_on_malformed_json(self, tmp_path):
        """Malformed openclaw.json does not crash; uses default model."""
        agent_dir = _create_agent_dir(tmp_path)
        (agent_dir / "openclaw.json").write_text("{invalid json!!!")
        server = _make_server()

        result = server._read_agent_state(agent_dir, "luna")

        assert result["model"] == "anthropic.claude-sonnet-4-20250514"

    def test_falls_back_when_agent_not_in_config(self, tmp_path):
        """When the requested agent name is not in the config's agents dict."""
        cfg = {
            "version": "1.0",
            "agents": {"other_agent": {"model": "other-model"}},
            "defaults": {"model": "fallback-model"},
        }
        agent_dir = _create_agent_dir(tmp_path, config_json=cfg)
        # Create the agent subdirectory for the requested agent
        (agent_dir / "agents" / "luna").mkdir(parents=True, exist_ok=True)
        (agent_dir / "agents" / "luna" / "sessions").mkdir(parents=True, exist_ok=True)
        server = _make_server()

        result = server._read_agent_state(agent_dir, "luna")

        assert result["model"] == "fallback-model"


class TestReadAgentStateSystemPrompt:
    """System prompt composition from SOUL.md, MEMORY.md, and daily memories."""

    def test_reads_soul_md(self, tmp_path):
        soul = "# Luna\nYou are a creative writing assistant."
        agent_dir = _create_agent_dir(tmp_path, soul_content=soul)
        server = _make_server()

        result = server._read_agent_state(agent_dir, "luna")

        assert "creative writing assistant" in result["system_prompt"]

    def test_reads_memory_md(self, tmp_path):
        memory = "User prefers Python over JavaScript."
        agent_dir = _create_agent_dir(
            tmp_path,
            soul_content="# Luna",
            memory_content=memory,
        )
        server = _make_server()

        result = server._read_agent_state(agent_dir, "luna")

        assert "Python over JavaScript" in result["system_prompt"]
        assert "## Memories" in result["system_prompt"]

    def test_reads_daily_memory_today(self, tmp_path):
        today_str = datetime.now().strftime("%Y-%m-%d")
        daily = {today_str: "Had a meeting about project X."}
        agent_dir = _create_agent_dir(
            tmp_path,
            soul_content="# Luna",
            daily_memories=daily,
        )
        server = _make_server()

        result = server._read_agent_state(agent_dir, "luna")

        assert "project X" in result["system_prompt"]
        assert "## Recent Notes" in result["system_prompt"]

    def test_reads_daily_memory_yesterday(self, tmp_path):
        yesterday = datetime.now() - timedelta(days=1)
        yesterday_str = yesterday.strftime("%Y-%m-%d")
        daily = {yesterday_str: "Discussed architecture decisions."}
        agent_dir = _create_agent_dir(
            tmp_path,
            soul_content="# Luna",
            daily_memories=daily,
        )
        server = _make_server()

        result = server._read_agent_state(agent_dir, "luna")

        assert "architecture decisions" in result["system_prompt"]

    def test_ignores_old_daily_memories(self, tmp_path):
        """Daily memory files older than yesterday are not included."""
        old_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
        daily = {old_date: "This should not appear."}
        agent_dir = _create_agent_dir(
            tmp_path,
            soul_content="# Luna",
            daily_memories=daily,
        )
        server = _make_server()

        result = server._read_agent_state(agent_dir, "luna")

        assert "This should not appear" not in result["system_prompt"]

    def test_combines_soul_memory_and_daily(self, tmp_path):
        today_str = datetime.now().strftime("%Y-%m-%d")
        agent_dir = _create_agent_dir(
            tmp_path,
            soul_content="# Luna\nBase personality.",
            memory_content="Likes cats.",
            daily_memories={today_str: "Working on tests."},
        )
        server = _make_server()

        result = server._read_agent_state(agent_dir, "luna")
        prompt = result["system_prompt"]

        assert "Base personality" in prompt
        assert "Likes cats" in prompt
        assert "Working on tests" in prompt

    def test_default_system_prompt_when_no_files(self, tmp_path):
        """When no SOUL.md / MEMORY.md exist, a default prompt is generated."""
        agent_dir = _create_agent_dir(tmp_path)
        # Remove SOUL.md if it exists (it was not created since soul_content was None)
        soul_file = agent_dir / "agents" / "luna" / "SOUL.md"
        if soul_file.exists():
            soul_file.unlink()
        server = _make_server()

        result = server._read_agent_state(agent_dir, "luna")

        assert "luna" in result["system_prompt"]
        assert "helpful AI assistant" in result["system_prompt"]

    def test_empty_soul_md_uses_default(self, tmp_path):
        """Empty SOUL.md (whitespace only) results in default prompt."""
        agent_dir = _create_agent_dir(tmp_path, soul_content="   \n  ")
        server = _make_server()

        result = server._read_agent_state(agent_dir, "luna")

        # The soul_content will be empty after strip(), so no soul_content in system_parts
        # If no system_parts, default prompt is used
        assert "luna" in result["system_prompt"]


class TestReadAgentStateHistory:
    """Session JSONL history parsing."""

    def test_parses_message_records(self, tmp_path):
        records = [
            {"type": "session", "timestamp": "20260201_120000", "agent": "luna"},
            {
                "type": "message",
                "timestamp": "2026-02-01T12:00:01",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hello"}],
                },
            },
            {
                "type": "message",
                "timestamp": "2026-02-01T12:00:02",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Hi there!"}],
                },
            },
        ]
        agent_dir = _create_agent_dir(tmp_path, session_records=records)
        server = _make_server()

        result = server._read_agent_state(agent_dir, "luna")

        assert len(result["history"]) == 2
        assert result["history"][0].role == "user"
        assert result["history"][0].content == "Hello"
        assert result["history"][1].role == "assistant"
        assert result["history"][1].content == "Hi there!"

    def test_skips_session_header_records(self, tmp_path):
        """Records with type != 'message' (e.g. 'session') are skipped."""
        records = [
            {"type": "session", "timestamp": "20260201_120000", "agent": "luna"},
            {
                "type": "message",
                "timestamp": "2026-02-01T12:00:01",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "Only this"}],
                },
            },
        ]
        agent_dir = _create_agent_dir(tmp_path, session_records=records)
        server = _make_server()

        result = server._read_agent_state(agent_dir, "luna")

        assert len(result["history"]) == 1
        assert result["history"][0].content == "Only this"

    def test_handles_string_content_blocks(self, tmp_path):
        """Content blocks can be plain strings instead of dicts."""
        records = [
            {
                "type": "message",
                "timestamp": "2026-02-01T12:00:01",
                "message": {
                    "role": "user",
                    "content": ["plain string content"],
                },
            },
        ]
        agent_dir = _create_agent_dir(tmp_path, session_records=records)
        server = _make_server()

        result = server._read_agent_state(agent_dir, "luna")

        assert len(result["history"]) == 1
        assert result["history"][0].content == "plain string content"

    def test_concatenates_multiple_content_blocks(self, tmp_path):
        """Multiple content blocks in a single message are concatenated."""
        records = [
            {
                "type": "message",
                "timestamp": "2026-02-01T12:00:01",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Part 1. "},
                        {"type": "text", "text": "Part 2."},
                    ],
                },
            },
        ]
        agent_dir = _create_agent_dir(tmp_path, session_records=records)
        server = _make_server()

        result = server._read_agent_state(agent_dir, "luna")

        assert len(result["history"]) == 1
        assert result["history"][0].content == "Part 1. Part 2."

    def test_empty_history_when_no_session_files(self, tmp_path):
        """Returns empty history and creates a new session file."""
        agent_dir = _create_agent_dir(tmp_path, create_sessions_dir=True)
        server = _make_server()

        result = server._read_agent_state(agent_dir, "luna")

        assert result["history"] == []

    def test_creates_new_session_file_when_none_exists(self, tmp_path):
        """When no .jsonl files exist, a new session file is created."""
        agent_dir = _create_agent_dir(tmp_path, create_sessions_dir=True)
        server = _make_server()

        result = server._read_agent_state(agent_dir, "luna")
        session_file = result["session_file"]

        assert session_file.exists()
        assert session_file.suffix == ".jsonl"

        # The new session file should contain a session header
        content = session_file.read_text().strip()
        header = json.loads(content)
        assert header["type"] == "session"
        assert header["agent"] == "luna"

    def test_creates_sessions_dir_when_missing(self, tmp_path):
        """When sessions directory does not exist, it is created."""
        agent_dir = _create_agent_dir(tmp_path, create_sessions_dir=False)
        # Also create the agents/<name> directory so the subdir path is valid
        agent_subdir = agent_dir / "agents" / "luna"
        agent_subdir.mkdir(parents=True, exist_ok=True)
        server = _make_server()

        result = server._read_agent_state(agent_dir, "luna")

        sessions_dir = agent_dir / "agents" / "luna" / "sessions"
        assert sessions_dir.exists()
        assert result["session_file"].parent == sessions_dir

    def test_returns_most_recent_session_file(self, tmp_path):
        """When multiple session files exist, the latest (by name) is used."""
        agent_dir = _create_agent_dir(tmp_path, create_sessions_dir=True)
        sessions_dir = agent_dir / "agents" / "luna" / "sessions"

        # Create older session
        older_record = {
            "type": "message",
            "timestamp": "2026-01-01T10:00:00",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": "Old message"}],
            },
        }
        (sessions_dir / "20260101_100000.jsonl").write_text(json.dumps(older_record) + "\n")

        # Create newer session
        newer_record = {
            "type": "message",
            "timestamp": "2026-02-01T12:00:00",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": "New message"}],
            },
        }
        (sessions_dir / "20260201_120000.jsonl").write_text(json.dumps(newer_record) + "\n")

        server = _make_server()

        result = server._read_agent_state(agent_dir, "luna")

        assert result["session_file"].name == "20260201_120000.jsonl"
        assert len(result["history"]) == 1
        assert result["history"][0].content == "New message"

    def test_skips_blank_lines(self, tmp_path):
        """Blank lines in session JSONL are silently skipped."""
        agent_dir = _create_agent_dir(tmp_path, create_sessions_dir=True)
        sessions_dir = agent_dir / "agents" / "luna" / "sessions"

        content = (
            '{"type": "session", "timestamp": "20260201_120000", "agent": "luna"}\n'
            "\n"
            '{"type": "message", "timestamp": "2026-02-01T12:00:01", "message": {"role": "user", "content": [{"type": "text", "text": "Hello"}]}}\n'
            "\n"
            "\n"
        )
        (sessions_dir / "20260201_120000.jsonl").write_text(content)

        server = _make_server()

        result = server._read_agent_state(agent_dir, "luna")

        assert len(result["history"]) == 1
        assert result["history"][0].content == "Hello"

    def test_skips_malformed_json_lines(self, tmp_path):
        """Malformed JSON lines in session are silently skipped."""
        agent_dir = _create_agent_dir(tmp_path, create_sessions_dir=True)
        sessions_dir = agent_dir / "agents" / "luna" / "sessions"

        content = (
            '{"type": "message", "timestamp": "t1", "message": {"role": "user", "content": [{"type": "text", "text": "Good"}]}}\n'
            "this is not valid json\n"
            '{"type": "message", "timestamp": "t2", "message": {"role": "assistant", "content": [{"type": "text", "text": "Also good"}]}}\n'
        )
        (sessions_dir / "20260201_120000.jsonl").write_text(content)

        server = _make_server()

        result = server._read_agent_state(agent_dir, "luna")

        assert len(result["history"]) == 2
        assert result["history"][0].content == "Good"
        assert result["history"][1].content == "Also good"

    def test_ignores_non_user_assistant_roles(self, tmp_path):
        """Messages with roles other than 'user'/'assistant' are skipped."""
        records = [
            {
                "type": "message",
                "timestamp": "t1",
                "message": {
                    "role": "system",
                    "content": [{"type": "text", "text": "System msg"}],
                },
            },
            {
                "type": "message",
                "timestamp": "t2",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "User msg"}],
                },
            },
        ]
        agent_dir = _create_agent_dir(tmp_path, session_records=records)
        server = _make_server()

        result = server._read_agent_state(agent_dir, "luna")

        assert len(result["history"]) == 1
        assert result["history"][0].role == "user"

    def test_ignores_empty_text_messages(self, tmp_path):
        """Messages with empty text content are skipped."""
        records = [
            {
                "type": "message",
                "timestamp": "t1",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": ""}],
                },
            },
            {
                "type": "message",
                "timestamp": "t2",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "Real message"}],
                },
            },
        ]
        agent_dir = _create_agent_dir(tmp_path, session_records=records)
        server = _make_server()

        result = server._read_agent_state(agent_dir, "luna")

        assert len(result["history"]) == 1
        assert result["history"][0].content == "Real message"


# ===========================================================================
# Tests for _append_to_session
# ===========================================================================


class TestAppendToSession:
    """Tests for session JSONL appending."""

    def test_appends_user_message(self, tmp_path):
        session_file = tmp_path / "session.jsonl"
        session_file.write_text("")  # Start with empty file
        server = _make_server()

        server._append_to_session(session_file, "user", "Hello there")

        lines = [line for line in session_file.read_text().strip().split("\n") if line]
        assert len(lines) == 1

        record = json.loads(lines[0])
        assert record["type"] == "message"
        assert record["message"]["role"] == "user"
        assert record["message"]["content"] == [{"type": "text", "text": "Hello there"}]
        assert "timestamp" in record

    def test_appends_assistant_message(self, tmp_path):
        session_file = tmp_path / "session.jsonl"
        session_file.write_text("")
        server = _make_server()

        server._append_to_session(session_file, "assistant", "I can help with that.")

        lines = [line for line in session_file.read_text().strip().split("\n") if line]
        record = json.loads(lines[0])
        assert record["message"]["role"] == "assistant"
        assert record["message"]["content"][0]["text"] == "I can help with that."

    def test_appends_multiple_messages_preserves_order(self, tmp_path):
        session_file = tmp_path / "session.jsonl"
        session_file.write_text("")
        server = _make_server()

        server._append_to_session(session_file, "user", "First")
        server._append_to_session(session_file, "assistant", "Second")
        server._append_to_session(session_file, "user", "Third")

        lines = [line for line in session_file.read_text().strip().split("\n") if line]
        assert len(lines) == 3

        assert json.loads(lines[0])["message"]["content"][0]["text"] == "First"
        assert json.loads(lines[1])["message"]["content"][0]["text"] == "Second"
        assert json.loads(lines[2])["message"]["content"][0]["text"] == "Third"

    def test_appends_to_existing_content(self, tmp_path):
        """Appending does not overwrite existing session content."""
        session_file = tmp_path / "session.jsonl"
        header = json.dumps({"type": "session", "timestamp": "20260201", "agent": "luna"})
        session_file.write_text(header + "\n")
        server = _make_server()

        server._append_to_session(session_file, "user", "New message")

        lines = [line for line in session_file.read_text().strip().split("\n") if line]
        assert len(lines) == 2
        assert json.loads(lines[0])["type"] == "session"
        assert json.loads(lines[1])["type"] == "message"

    def test_content_block_format_is_openclaw_compatible(self, tmp_path):
        """Verify the exact JSONL format matches OpenClaw expectations."""
        session_file = tmp_path / "session.jsonl"
        session_file.write_text("")
        server = _make_server()

        server._append_to_session(session_file, "user", "Test content")

        lines = [line for line in session_file.read_text().strip().split("\n") if line]
        record = json.loads(lines[0])

        # Verify structure
        assert "type" in record
        assert record["type"] == "message"
        assert "timestamp" in record
        assert "message" in record
        assert "role" in record["message"]
        assert "content" in record["message"]
        assert isinstance(record["message"]["content"], list)
        assert len(record["message"]["content"]) == 1
        assert record["message"]["content"][0]["type"] == "text"
        assert record["message"]["content"][0]["text"] == "Test content"

    def test_timestamp_is_iso_format(self, tmp_path):
        """Verify the timestamp is in ISO 8601 format."""
        session_file = tmp_path / "session.jsonl"
        session_file.write_text("")
        server = _make_server()

        server._append_to_session(session_file, "user", "msg")

        lines = [line for line in session_file.read_text().strip().split("\n") if line]
        record = json.loads(lines[0])
        ts = record["timestamp"]

        # Should be parseable as ISO format
        parsed = datetime.fromisoformat(ts)
        assert isinstance(parsed, datetime)

    def test_roundtrip_append_then_read(self, tmp_path):
        """Messages appended by _append_to_session are correctly parsed by _read_agent_state."""
        agent_dir = _create_agent_dir(tmp_path, soul_content="# Luna", create_sessions_dir=True)
        sessions_dir = agent_dir / "agents" / "luna" / "sessions"
        session_file = sessions_dir / "20260201_120000.jsonl"

        header = json.dumps({"type": "session", "timestamp": "20260201_120000", "agent": "luna"})
        session_file.write_text(header + "\n")

        server = _make_server()

        # Append messages
        server._append_to_session(session_file, "user", "What is 2+2?")
        server._append_to_session(session_file, "assistant", "2+2 equals 4.")

        # Read back via _read_agent_state
        result = server._read_agent_state(agent_dir, "luna")

        assert len(result["history"]) == 2
        assert result["history"][0].role == "user"
        assert result["history"][0].content == "What is 2+2?"
        assert result["history"][1].role == "assistant"
        assert result["history"][1].content == "2+2 equals 4."


# ===========================================================================
# Tests for handle_agent_chat_stream command routing
# ===========================================================================


class TestHandleAgentChatStreamRouting:
    """Verify AGENT_CHAT_STREAM is correctly routed in handle_request."""

    def test_agent_chat_stream_returns_none(self):
        """AGENT_CHAT_STREAM is a streaming command, so handle_request returns None."""
        server = _make_server()
        mock_conn = MagicMock()

        # Patch handle_agent_chat_stream to avoid actually running the handler
        with patch.object(server, "handle_agent_chat_stream") as mock_handler:
            result = server.handle_request({"command": "AGENT_CHAT_STREAM"}, mock_conn)

        assert result is None
        mock_handler.assert_called_once_with({"command": "AGENT_CHAT_STREAM"}, mock_conn)

    def test_agent_chat_stream_case_insensitive(self):
        """Command matching is case-insensitive (uppercased)."""
        server = _make_server()
        mock_conn = MagicMock()

        with patch.object(server, "handle_agent_chat_stream") as mock_handler:
            result = server.handle_request({"command": "agent_chat_stream"}, mock_conn)

        assert result is None
        mock_handler.assert_called_once()

    def test_agent_chat_stream_in_available_commands(self):
        """AGENT_CHAT_STREAM appears in the available_commands list for unknown commands."""
        server = _make_server()
        mock_conn = MagicMock()

        result = server.handle_request({"command": "BOGUS_COMMAND"}, mock_conn)

        assert result is not None
        assert result["status"] == "error"
        assert "AGENT_CHAT_STREAM" in result["available_commands"]

    def test_chat_stream_still_works(self):
        """CHAT_STREAM still works correctly alongside AGENT_CHAT_STREAM."""
        server = _make_server()
        mock_conn = MagicMock()

        with patch.object(server, "handle_chat_stream") as mock_handler:
            result = server.handle_request({"command": "CHAT_STREAM"}, mock_conn)

        assert result is None
        mock_handler.assert_called_once()

    def test_non_streaming_commands_return_dict(self):
        """Non-streaming commands like HEALTH return a dict response."""
        server = _make_server()
        mock_conn = MagicMock()

        result = server.handle_request({"command": "HEALTH"}, mock_conn)

        assert result is not None
        assert isinstance(result, dict)
        assert result["command"] == "HEALTH"

    def test_all_known_commands_in_available_list(self):
        """All known commands appear in available_commands error message."""
        server = _make_server()
        mock_conn = MagicMock()

        result = server.handle_request({"command": "UNKNOWN"}, mock_conn)

        expected_commands = [
            "GET_PUBLIC_KEY",
            "SET_CREDENTIALS",
            "HEALTH",
            "CHAT",
            "RUN_TESTS",
            "RUN_AGENT",
            "CHAT_STREAM",
            "AGENT_CHAT_STREAM",
        ]
        for cmd in expected_commands:
            assert cmd in result["available_commands"], f"{cmd} missing from available_commands"
