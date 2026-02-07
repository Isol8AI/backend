"""
Python bridge to OpenClaw agent runner via Node.js subprocess.

Spawns `node run_agent.mjs` as a child process, sends a JSON request
via stdin, and yields NDJSON events from stdout as Python dicts.

This module is imported by bedrock_server.py inside the Nitro Enclave.
It is also testable in isolation with mocked subprocess.

Event types yielded:
  - {"type": "partial", "text": "..."}        Token-by-token chunks
  - {"type": "block", "text": "..."}          Accumulated text blocks
  - {"type": "tool_result", "text": "..."}    Tool execution results
  - {"type": "reasoning", "text": "..."}      Extended thinking output
  - {"type": "assistant_start"}               Before first token
  - {"type": "agent_event", ...}              Low-level diagnostics
  - {"type": "error", "message": "..."}       Agent-level error
  - {"type": "done", "meta": {...}}           Completion with metadata
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, Generator, Optional

logger = logging.getLogger(__name__)

# Path to the bridge script, relative to this file
_BRIDGE_SCRIPT = Path(__file__).parent / "run_agent.mjs"


def run_agent_streaming(
    state_dir: str,
    agent_name: str,
    message: str,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    timeout_ms: Optional[int] = None,
    session_id: Optional[str] = None,
    node_path: str = "node",
    bridge_script: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
) -> Generator[Dict[str, Any], None, None]:
    """
    Run OpenClaw agent via Node.js subprocess with NDJSON streaming.

    Args:
        state_dir: Absolute path to extracted agent tarball (tmpfs).
        agent_name: Agent identifier (matches agents/{name}/ directory).
        message: User message text.
        model: LLM model ID (optional, defaults set in bridge).
        provider: LLM provider name (optional, defaults to amazon-bedrock).
        timeout_ms: Max execution time in milliseconds (optional).
        session_id: Session ID for conversation continuity (optional).
        node_path: Path to Node.js binary (default: "node").
        bridge_script: Override path to run_agent.mjs (for testing).
        env: Environment variables to pass to subprocess (merged with os.environ).

    Yields:
        dict: NDJSON events from the agent runner.

    Raises:
        RuntimeError: If the bridge process exits with non-zero code.
        FileNotFoundError: If the bridge script doesn't exist.
    """
    script = Path(bridge_script) if bridge_script else _BRIDGE_SCRIPT

    if not script.exists():
        raise FileNotFoundError(
            f"Agent bridge script not found: {script}. Ensure run_agent.mjs is in the enclave directory."
        )

    # Build JSON request
    request: Dict[str, Any] = {
        "stateDir": state_dir,
        "agentName": agent_name,
        "message": message,
    }
    if model is not None:
        request["model"] = model
    if provider is not None:
        request["provider"] = provider
    if timeout_ms is not None:
        request["timeoutMs"] = timeout_ms
    if session_id is not None:
        request["sessionId"] = session_id

    # Merge environment
    import os

    proc_env = os.environ.copy()
    if env:
        proc_env.update(env)

    logger.info(
        "Starting agent bridge: agent=%s model=%s",
        agent_name,
        model or "default",
    )

    proc = subprocess.Popen(
        [node_path, str(script)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=proc_env,
    )

    try:
        # Send request via stdin and close to signal EOF
        try:
            proc.stdin.write(json.dumps(request))
            proc.stdin.flush()
            proc.stdin.close()
        except BrokenPipeError:
            # Process died before we finished writing — handled below
            pass

        # Stream NDJSON events from stdout
        event_count = 0
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                event_count += 1
                yield event
            except json.JSONDecodeError:
                logger.warning("Skipping malformed NDJSON line: %s", line[:100])
                continue

        # Wait for process to finish
        proc.wait()

        # Read stderr for diagnostics
        stderr_output = proc.stderr.read() if proc.stderr else ""

        if proc.returncode != 0:
            logger.error(
                "Agent bridge failed (exit %d): %s",
                proc.returncode,
                stderr_output[:500],
            )
            raise RuntimeError(f"Agent bridge failed (exit {proc.returncode}): {stderr_output[:500]}")

        logger.info(
            "Agent bridge completed: %d events, exit %d",
            event_count,
            proc.returncode,
        )
    except GeneratorExit:
        # Generator abandoned before full consumption — kill subprocess
        logger.warning("Agent bridge generator abandoned, killing subprocess")
        proc.kill()
        proc.wait()
        raise
    finally:
        # Ensure subprocess is always cleaned up
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        if proc.stdout:
            proc.stdout.close()
        if proc.stderr:
            proc.stderr.close()


def collect_response_text(events: Generator[Dict[str, Any], None, None]) -> str:
    """
    Consume a stream of bridge events and return the concatenated response text.

    This is a convenience function for non-streaming callers that just want
    the full response.

    Args:
        events: Generator from run_agent_streaming().

    Returns:
        Concatenated text from all "partial" events.
    """
    parts = []
    for event in events:
        if event.get("type") == "partial" and event.get("text"):
            parts.append(event["text"])
    return "".join(parts)
