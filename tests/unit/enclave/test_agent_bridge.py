"""
Tests for agent_bridge.py — Python bridge to OpenClaw via Node.js subprocess.

agent_bridge.py lives in enclave/ and uses only standard library modules,
so no enclave-specific mocking is required. We mock subprocess.Popen to
simulate the Node.js bridge process.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add enclave directory to sys.path so we can import agent_bridge
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "enclave"))
from agent_bridge import run_agent_streaming, collect_response_text  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bridge_script(tmp_path):
    """Create a fake bridge script file so FileNotFoundError is not raised."""
    script = tmp_path / "run_agent.mjs"
    script.write_text("// fake bridge script")
    return str(script)


@pytest.fixture
def ndjson_events():
    """Standard sequence of NDJSON events from a successful agent run."""
    return [
        {"type": "assistant_start"},
        {"type": "partial", "text": "Hello"},
        {"type": "partial", "text": ", world!"},
        {"type": "block", "text": "Hello, world!"},
        {"type": "done", "meta": {"durationMs": 1234, "stopReason": "end_turn"}},
    ]


def _mock_popen(stdout_lines, returncode=0, stderr_text=""):
    """
    Create a mock subprocess.Popen that yields NDJSON lines from stdout.

    Args:
        stdout_lines: List of dicts to emit as NDJSON, or list of raw strings.
        returncode: Process exit code.
        stderr_text: Text returned by stderr.read().
    """
    mock_proc = MagicMock()

    # Build stdout as iterable of newline-terminated strings
    lines = []
    for item in stdout_lines:
        if isinstance(item, dict):
            lines.append(json.dumps(item) + "\n")
        else:
            lines.append(str(item) + "\n")

    mock_proc.stdout.__iter__ = MagicMock(return_value=iter(lines))
    mock_proc.stdin = MagicMock()
    mock_proc.stderr = MagicMock()
    mock_proc.stderr.read.return_value = stderr_text
    mock_proc.wait.return_value = returncode
    mock_proc.returncode = returncode

    return mock_proc


# ---------------------------------------------------------------------------
# Tests: run_agent_streaming — happy path
# ---------------------------------------------------------------------------


class TestRunAgentStreaming:
    """Tests for the run_agent_streaming generator."""

    @patch("agent_bridge.subprocess.Popen")
    def test_yields_all_ndjson_events(self, mock_popen_cls, bridge_script, ndjson_events):
        """All valid NDJSON events should be yielded."""
        mock_popen_cls.return_value = _mock_popen(ndjson_events, returncode=0)

        events = list(
            run_agent_streaming(
                state_dir="/tmp/agent_state",
                agent_name="luna",
                message="Hello!",
                bridge_script=bridge_script,
            )
        )

        assert len(events) == len(ndjson_events)
        assert events[0]["type"] == "assistant_start"
        assert events[1] == {"type": "partial", "text": "Hello"}
        assert events[2] == {"type": "partial", "text": ", world!"}
        assert events[3] == {"type": "block", "text": "Hello, world!"}
        assert events[4]["type"] == "done"
        assert events[4]["meta"]["durationMs"] == 1234

    @patch("agent_bridge.subprocess.Popen")
    def test_sends_correct_json_request(self, mock_popen_cls, bridge_script):
        """Verify the JSON request sent to stdin matches expected schema."""
        mock_proc = _mock_popen(
            [{"type": "done", "meta": {}}],
            returncode=0,
        )
        mock_popen_cls.return_value = mock_proc

        list(
            run_agent_streaming(
                state_dir="/tmp/test_state",
                agent_name="rex",
                message="How are you?",
                model="us.anthropic.claude-3-5-sonnet-v2",
                provider="amazon-bedrock",
                timeout_ms=60000,
                session_id="sess-123",
                bridge_script=bridge_script,
            )
        )

        # Check what was written to stdin
        written_data = mock_proc.stdin.write.call_args[0][0]
        request = json.loads(written_data)

        assert request["stateDir"] == "/tmp/test_state"
        assert request["agentName"] == "rex"
        assert request["message"] == "How are you?"
        assert request["model"] == "us.anthropic.claude-3-5-sonnet-v2"
        assert request["provider"] == "amazon-bedrock"
        assert request["timeoutMs"] == 60000
        assert request["sessionId"] == "sess-123"

    @patch("agent_bridge.subprocess.Popen")
    def test_minimal_request_omits_optional_fields(self, mock_popen_cls, bridge_script):
        """When optional params are None, they should not appear in the request."""
        mock_proc = _mock_popen(
            [{"type": "done", "meta": {}}],
            returncode=0,
        )
        mock_popen_cls.return_value = mock_proc

        list(
            run_agent_streaming(
                state_dir="/tmp/state",
                agent_name="luna",
                message="Hi",
                bridge_script=bridge_script,
            )
        )

        written_data = mock_proc.stdin.write.call_args[0][0]
        request = json.loads(written_data)

        # Only required fields should be present
        assert set(request.keys()) == {"stateDir", "agentName", "message"}
        assert "model" not in request
        assert "provider" not in request
        assert "timeoutMs" not in request
        assert "sessionId" not in request

    @patch("agent_bridge.subprocess.Popen")
    def test_stdin_is_flushed_and_closed(self, mock_popen_cls, bridge_script):
        """Stdin should be written, flushed, and closed."""
        mock_proc = _mock_popen(
            [{"type": "done", "meta": {}}],
            returncode=0,
        )
        mock_popen_cls.return_value = mock_proc

        list(
            run_agent_streaming(
                state_dir="/tmp/state",
                agent_name="luna",
                message="Test",
                bridge_script=bridge_script,
            )
        )

        mock_proc.stdin.write.assert_called_once()
        mock_proc.stdin.flush.assert_called_once()
        mock_proc.stdin.close.assert_called_once()

    @patch("agent_bridge.subprocess.Popen")
    def test_popen_called_with_correct_args(self, mock_popen_cls, bridge_script):
        """Popen should be invoked with node and the bridge script."""
        mock_popen_cls.return_value = _mock_popen(
            [{"type": "done", "meta": {}}],
            returncode=0,
        )

        list(
            run_agent_streaming(
                state_dir="/tmp/state",
                agent_name="luna",
                message="Test",
                bridge_script=bridge_script,
            )
        )

        call_args = mock_popen_cls.call_args
        cmd = call_args[0][0]
        assert cmd[0] == "node"
        assert cmd[1] == bridge_script

        # Verify subprocess options
        kwargs = call_args[1]
        assert kwargs["text"] is True

    @patch("agent_bridge.subprocess.Popen")
    def test_custom_node_path(self, mock_popen_cls, bridge_script):
        """Custom node_path should be used in the subprocess command."""
        mock_popen_cls.return_value = _mock_popen(
            [{"type": "done", "meta": {}}],
            returncode=0,
        )

        list(
            run_agent_streaming(
                state_dir="/tmp/state",
                agent_name="luna",
                message="Test",
                node_path="/usr/local/bin/node22",
                bridge_script=bridge_script,
            )
        )

        cmd = mock_popen_cls.call_args[0][0]
        assert cmd[0] == "/usr/local/bin/node22"

    @patch("agent_bridge.subprocess.Popen")
    def test_custom_env_merged_with_os_environ(self, mock_popen_cls, bridge_script):
        """Custom env vars should be merged with os.environ."""
        mock_popen_cls.return_value = _mock_popen(
            [{"type": "done", "meta": {}}],
            returncode=0,
        )

        custom_env = {"OPENCLAW_PATH": "/custom/path", "AWS_REGION": "eu-west-1"}
        list(
            run_agent_streaming(
                state_dir="/tmp/state",
                agent_name="luna",
                message="Test",
                env=custom_env,
                bridge_script=bridge_script,
            )
        )

        call_kwargs = mock_popen_cls.call_args[1]
        proc_env = call_kwargs["env"]
        assert proc_env["OPENCLAW_PATH"] == "/custom/path"
        assert proc_env["AWS_REGION"] == "eu-west-1"
        # Should also contain inherited env vars (like PATH)
        assert "PATH" in proc_env


# ---------------------------------------------------------------------------
# Tests: run_agent_streaming — error handling
# ---------------------------------------------------------------------------


class TestRunAgentStreamingErrors:
    """Tests for error handling in run_agent_streaming."""

    def test_missing_bridge_script_raises_file_not_found(self):
        """FileNotFoundError when bridge script doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Agent bridge script not found"):
            list(
                run_agent_streaming(
                    state_dir="/tmp/state",
                    agent_name="luna",
                    message="Test",
                    bridge_script="/nonexistent/run_agent.mjs",
                )
            )

    @patch("agent_bridge.subprocess.Popen")
    def test_nonzero_exit_raises_runtime_error(self, mock_popen_cls, bridge_script):
        """RuntimeError when the bridge process exits with non-zero code."""
        mock_popen_cls.return_value = _mock_popen(
            [{"type": "partial", "text": "partial output"}],
            returncode=1,
            stderr_text="Error: Cannot find module '@mariozechner/pi-ai'",
        )

        with pytest.raises(RuntimeError, match="Agent bridge failed.*exit 1"):
            list(
                run_agent_streaming(
                    state_dir="/tmp/state",
                    agent_name="luna",
                    message="Test",
                    bridge_script=bridge_script,
                )
            )

    @patch("agent_bridge.subprocess.Popen")
    def test_stderr_included_in_runtime_error(self, mock_popen_cls, bridge_script):
        """Stderr content should be included in the RuntimeError message."""
        error_msg = "OpenClaw not found at /opt/openclaw/dist/agents/pi-embedded-runner.js"
        mock_popen_cls.return_value = _mock_popen(
            [],
            returncode=1,
            stderr_text=error_msg,
        )

        with pytest.raises(RuntimeError, match="OpenClaw not found"):
            list(
                run_agent_streaming(
                    state_dir="/tmp/state",
                    agent_name="luna",
                    message="Test",
                    bridge_script=bridge_script,
                )
            )

    @patch("agent_bridge.subprocess.Popen")
    def test_malformed_ndjson_lines_skipped(self, mock_popen_cls, bridge_script):
        """Malformed NDJSON lines should be skipped, not crash."""
        mock_proc = _mock_popen([], returncode=0)
        # Override stdout to include malformed lines
        stdout_lines = [
            json.dumps({"type": "partial", "text": "good"}) + "\n",
            "this is not valid json\n",
            json.dumps({"type": "done", "meta": {}}) + "\n",
        ]
        mock_proc.stdout.__iter__ = MagicMock(return_value=iter(stdout_lines))
        mock_popen_cls.return_value = mock_proc

        events = list(
            run_agent_streaming(
                state_dir="/tmp/state",
                agent_name="luna",
                message="Test",
                bridge_script=bridge_script,
            )
        )

        # Only valid events should be yielded
        assert len(events) == 2
        assert events[0]["type"] == "partial"
        assert events[1]["type"] == "done"

    @patch("agent_bridge.subprocess.Popen")
    def test_empty_lines_skipped(self, mock_popen_cls, bridge_script):
        """Empty/whitespace lines should be skipped."""
        mock_proc = _mock_popen([], returncode=0)
        stdout_lines = [
            "\n",
            "  \n",
            json.dumps({"type": "done", "meta": {}}) + "\n",
            "\n",
        ]
        mock_proc.stdout.__iter__ = MagicMock(return_value=iter(stdout_lines))
        mock_popen_cls.return_value = mock_proc

        events = list(
            run_agent_streaming(
                state_dir="/tmp/state",
                agent_name="luna",
                message="Test",
                bridge_script=bridge_script,
            )
        )

        assert len(events) == 1
        assert events[0]["type"] == "done"

    @patch("agent_bridge.subprocess.Popen")
    def test_broken_pipe_on_stdin_write_handled(self, mock_popen_cls, bridge_script):
        """BrokenPipeError when writing to stdin should be handled gracefully."""
        mock_proc = _mock_popen(
            [{"type": "error", "message": "process died early"}],
            returncode=0,
        )
        mock_proc.stdin.write.side_effect = BrokenPipeError("Broken pipe")
        mock_popen_cls.return_value = mock_proc

        # Should not raise — the error event from stdout should still be yielded
        events = list(
            run_agent_streaming(
                state_dir="/tmp/state",
                agent_name="luna",
                message="Test",
                bridge_script=bridge_script,
            )
        )

        assert len(events) == 1
        assert events[0]["type"] == "error"

    @patch("agent_bridge.subprocess.Popen")
    def test_no_events_from_empty_stdout(self, mock_popen_cls, bridge_script):
        """No events should be yielded when stdout is empty."""
        mock_popen_cls.return_value = _mock_popen([], returncode=0)

        events = list(
            run_agent_streaming(
                state_dir="/tmp/state",
                agent_name="luna",
                message="Test",
                bridge_script=bridge_script,
            )
        )

        assert events == []


# ---------------------------------------------------------------------------
# Tests: run_agent_streaming — all event types
# ---------------------------------------------------------------------------


class TestRunAgentStreamingEventTypes:
    """Verify all NDJSON event types are yielded correctly."""

    @patch("agent_bridge.subprocess.Popen")
    def test_partial_event(self, mock_popen_cls, bridge_script):
        """Token-by-token partial events."""
        events_in = [{"type": "partial", "text": "chunk"}]
        mock_popen_cls.return_value = _mock_popen(events_in, returncode=0)

        events = list(
            run_agent_streaming(
                state_dir="/tmp/s",
                agent_name="a",
                message="m",
                bridge_script=bridge_script,
            )
        )

        assert events[0] == {"type": "partial", "text": "chunk"}

    @patch("agent_bridge.subprocess.Popen")
    def test_block_event(self, mock_popen_cls, bridge_script):
        """Accumulated text block events."""
        events_in = [{"type": "block", "text": "full block"}]
        mock_popen_cls.return_value = _mock_popen(events_in, returncode=0)

        events = list(
            run_agent_streaming(
                state_dir="/tmp/s",
                agent_name="a",
                message="m",
                bridge_script=bridge_script,
            )
        )

        assert events[0] == {"type": "block", "text": "full block"}

    @patch("agent_bridge.subprocess.Popen")
    def test_tool_result_event(self, mock_popen_cls, bridge_script):
        """Tool execution result events."""
        events_in = [{"type": "tool_result", "text": "ls output"}]
        mock_popen_cls.return_value = _mock_popen(events_in, returncode=0)

        events = list(
            run_agent_streaming(
                state_dir="/tmp/s",
                agent_name="a",
                message="m",
                bridge_script=bridge_script,
            )
        )

        assert events[0] == {"type": "tool_result", "text": "ls output"}

    @patch("agent_bridge.subprocess.Popen")
    def test_reasoning_event(self, mock_popen_cls, bridge_script):
        """Extended thinking events."""
        events_in = [{"type": "reasoning", "text": "thinking..."}]
        mock_popen_cls.return_value = _mock_popen(events_in, returncode=0)

        events = list(
            run_agent_streaming(
                state_dir="/tmp/s",
                agent_name="a",
                message="m",
                bridge_script=bridge_script,
            )
        )

        assert events[0] == {"type": "reasoning", "text": "thinking..."}

    @patch("agent_bridge.subprocess.Popen")
    def test_assistant_start_event(self, mock_popen_cls, bridge_script):
        """Assistant start marker event."""
        events_in = [{"type": "assistant_start"}]
        mock_popen_cls.return_value = _mock_popen(events_in, returncode=0)

        events = list(
            run_agent_streaming(
                state_dir="/tmp/s",
                agent_name="a",
                message="m",
                bridge_script=bridge_script,
            )
        )

        assert events[0] == {"type": "assistant_start"}

    @patch("agent_bridge.subprocess.Popen")
    def test_agent_event(self, mock_popen_cls, bridge_script):
        """Low-level agent lifecycle events."""
        events_in = [{"type": "agent_event", "stream": "lifecycle", "data": {"status": "running"}}]
        mock_popen_cls.return_value = _mock_popen(events_in, returncode=0)

        events = list(
            run_agent_streaming(
                state_dir="/tmp/s",
                agent_name="a",
                message="m",
                bridge_script=bridge_script,
            )
        )

        assert events[0]["type"] == "agent_event"
        assert events[0]["stream"] == "lifecycle"
        assert events[0]["data"]["status"] == "running"

    @patch("agent_bridge.subprocess.Popen")
    def test_error_event(self, mock_popen_cls, bridge_script):
        """Agent-level error events (exit 0, error via NDJSON)."""
        events_in = [{"type": "error", "message": "context_overflow"}]
        mock_popen_cls.return_value = _mock_popen(events_in, returncode=0)

        events = list(
            run_agent_streaming(
                state_dir="/tmp/s",
                agent_name="a",
                message="m",
                bridge_script=bridge_script,
            )
        )

        assert events[0] == {"type": "error", "message": "context_overflow"}

    @patch("agent_bridge.subprocess.Popen")
    def test_media_event(self, mock_popen_cls, bridge_script):
        """Media URL events."""
        events_in = [{"type": "media", "urls": ["https://example.com/img.png"]}]
        mock_popen_cls.return_value = _mock_popen(events_in, returncode=0)

        events = list(
            run_agent_streaming(
                state_dir="/tmp/s",
                agent_name="a",
                message="m",
                bridge_script=bridge_script,
            )
        )

        assert events[0]["type"] == "media"
        assert events[0]["urls"] == ["https://example.com/img.png"]

    @patch("agent_bridge.subprocess.Popen")
    def test_done_event_with_full_meta(self, mock_popen_cls, bridge_script):
        """Done event with complete metadata."""
        meta = {
            "durationMs": 5432,
            "agentMeta": {"turnsUsed": 3},
            "error": None,
            "stopReason": "end_turn",
        }
        events_in = [{"type": "done", "meta": meta}]
        mock_popen_cls.return_value = _mock_popen(events_in, returncode=0)

        events = list(
            run_agent_streaming(
                state_dir="/tmp/s",
                agent_name="a",
                message="m",
                bridge_script=bridge_script,
            )
        )

        assert events[0]["type"] == "done"
        assert events[0]["meta"]["durationMs"] == 5432
        assert events[0]["meta"]["stopReason"] == "end_turn"


# ---------------------------------------------------------------------------
# Tests: collect_response_text
# ---------------------------------------------------------------------------


class TestCollectResponseText:
    """Tests for the collect_response_text convenience function."""

    def test_concatenates_partial_events(self):
        """Should concatenate text from all partial events."""
        events = iter(
            [
                {"type": "assistant_start"},
                {"type": "partial", "text": "Hello"},
                {"type": "partial", "text": ", "},
                {"type": "partial", "text": "world!"},
                {"type": "block", "text": "Hello, world!"},
                {"type": "done", "meta": {}},
            ]
        )

        result = collect_response_text(events)
        assert result == "Hello, world!"

    def test_empty_stream(self):
        """Should return empty string for empty event stream."""
        result = collect_response_text(iter([]))
        assert result == ""

    def test_no_partial_events(self):
        """Should return empty string when no partial events exist."""
        events = iter(
            [
                {"type": "assistant_start"},
                {"type": "block", "text": "full block"},
                {"type": "done", "meta": {}},
            ]
        )

        result = collect_response_text(events)
        assert result == ""

    def test_partial_events_with_none_text(self):
        """Partial events with None text should be skipped."""
        events = iter(
            [
                {"type": "partial", "text": "Hello"},
                {"type": "partial", "text": None},
                {"type": "partial", "text": "!"},
            ]
        )

        result = collect_response_text(events)
        assert result == "Hello!"

    def test_partial_events_with_empty_text(self):
        """Partial events with empty string text should be skipped."""
        events = iter(
            [
                {"type": "partial", "text": "Hello"},
                {"type": "partial", "text": ""},
                {"type": "partial", "text": "!"},
            ]
        )

        result = collect_response_text(events)
        assert result == "Hello!"

    def test_ignores_non_partial_events(self):
        """Only partial events contribute to the response text."""
        events = iter(
            [
                {"type": "reasoning", "text": "thinking..."},
                {"type": "partial", "text": "response"},
                {"type": "tool_result", "text": "ls output"},
                {"type": "block", "text": "response"},
                {"type": "error", "message": "warning"},
            ]
        )

        result = collect_response_text(events)
        assert result == "response"


# ---------------------------------------------------------------------------
# Tests: streaming behavior (generator protocol)
# ---------------------------------------------------------------------------


class TestStreamingBehavior:
    """Tests verifying generator/streaming protocol compliance."""

    @patch("agent_bridge.subprocess.Popen")
    def test_events_yielded_incrementally(self, mock_popen_cls, bridge_script):
        """Events should be yielded one at a time, not buffered."""
        events_in = [{"type": "partial", "text": f"chunk-{i}"} for i in range(100)]
        mock_popen_cls.return_value = _mock_popen(events_in, returncode=0)

        gen = run_agent_streaming(
            state_dir="/tmp/s",
            agent_name="a",
            message="m",
            bridge_script=bridge_script,
        )

        # Consume first event
        first = next(gen)
        assert first["text"] == "chunk-0"

        # Consume second event
        second = next(gen)
        assert second["text"] == "chunk-1"

    @patch("agent_bridge.subprocess.Popen")
    def test_process_wait_called_after_stdout_consumed(self, mock_popen_cls, bridge_script):
        """Process should be waited on after stdout is fully consumed."""
        mock_proc = _mock_popen(
            [{"type": "done", "meta": {}}],
            returncode=0,
        )
        mock_popen_cls.return_value = mock_proc

        list(
            run_agent_streaming(
                state_dir="/tmp/s",
                agent_name="a",
                message="m",
                bridge_script=bridge_script,
            )
        )

        mock_proc.wait.assert_called_once()

    @patch("agent_bridge.subprocess.Popen")
    def test_stderr_read_after_process_completes(self, mock_popen_cls, bridge_script):
        """Stderr should be read after the process completes for diagnostics."""
        mock_proc = _mock_popen(
            [{"type": "done", "meta": {}}],
            returncode=0,
            stderr_text="[Bridge] agent=luna model=default session=...\n[Bridge] Done\n",
        )
        mock_popen_cls.return_value = mock_proc

        list(
            run_agent_streaming(
                state_dir="/tmp/s",
                agent_name="a",
                message="m",
                bridge_script=bridge_script,
            )
        )

        mock_proc.stderr.read.assert_called_once()
