"""
AgentRunner for tarball operations and OpenClaw CLI invocation.

This module handles:
1. Packing OpenClaw agent directories into tarballs
2. Unpacking tarballs to tmpfs
3. Running the OpenClaw CLI
4. Cleaning up tmpfs after use

The runner enables OpenClaw to run completely unchanged - we just
wrap it with encrypt/decrypt at the boundary.
"""

import io
import json
import logging
import os
import shutil
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# Default SOUL.md content for new agents
DEFAULT_SOUL_CONTENT = """# {agent_name}

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

# Default OpenClaw config
DEFAULT_OPENCLAW_CONFIG = {
    "version": "1.0",
    "agents": {},
    "defaults": {
        "model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
    },
}


@dataclass
class AgentConfig:
    """Configuration for creating a new agent."""

    agent_name: str
    soul_content: Optional[str] = None
    model: str = "anthropic.claude-3-5-sonnet-20241022-v2:0"

    def __post_init__(self):
        if self.soul_content is None:
            self.soul_content = DEFAULT_SOUL_CONTENT.format(agent_name=self.agent_name)


@dataclass
class AgentRunResult:
    """Result of running the OpenClaw CLI."""

    success: bool
    response: str = ""
    error: str = ""
    stdout: str = ""
    stderr: str = ""


class AgentRunner:
    """
    Runner for OpenClaw tarball operations and CLI invocation.

    This class handles the mechanics of:
    - Packing/unpacking agent directories as tarballs
    - Creating fresh agent directories for new users
    - Running the OpenClaw CLI with proper environment
    - Cleaning up tmpfs directories after use
    """

    def __init__(
        self,
        tmpfs_base: str = "/tmp/openclaw",
        openclaw_path: str = "openclaw",
        node_path: Optional[str] = None,
    ):
        """
        Initialize the runner.

        Args:
            tmpfs_base: Base path for user tmpfs directories
            openclaw_path: Path to the openclaw CLI binary
            node_path: Path to Node.js binary (auto-detected if not provided)
        """
        self.tmpfs_base = Path(tmpfs_base)
        self.openclaw_path = openclaw_path
        self.node_path = node_path or self._find_node()

    def _find_node(self) -> str:
        """Find Node.js binary path."""
        try:
            result = subprocess.run(
                ["which", "node"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return "node"

    def get_user_tmpfs_path(self, user_id: str) -> Path:
        """
        Get the tmpfs path for a specific user.

        Each user gets an isolated directory to prevent cross-user access.
        User IDs are sanitized to prevent path traversal attacks.
        """
        # Sanitize user_id to prevent path traversal
        safe_user_id = "".join(c for c in user_id if c.isalnum() or c in "_-")
        return self.tmpfs_base / safe_user_id

    def pack_directory(self, directory: Path) -> bytes:
        """
        Pack a directory into a gzip-compressed tarball.

        Args:
            directory: Path to the directory to pack

        Returns:
            Bytes of the gzip-compressed tarball
        """
        buffer = io.BytesIO()

        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            for item in directory.rglob("*"):
                if item.is_file():
                    arcname = item.relative_to(directory)
                    tar.add(item, arcname=str(arcname))

        buffer.seek(0)
        return buffer.read()

    def unpack_tarball(self, tarball_bytes: bytes, target_dir: Path) -> None:
        """
        Unpack a tarball to a target directory.

        Args:
            tarball_bytes: The tarball data
            target_dir: Directory to extract to (created if doesn't exist)

        Raises:
            ValueError: If tarball contains unsafe paths (path traversal)
        """
        target_dir.mkdir(parents=True, exist_ok=True)

        buffer = io.BytesIO(tarball_bytes)
        with tarfile.open(fileobj=buffer, mode="r:gz") as tar:
            # Security: Check for path traversal
            for member in tar.getmembers():
                if member.name.startswith("/") or ".." in member.name:
                    raise ValueError(f"Unsafe path in tarball: {member.name}")
            tar.extractall(target_dir)

    def create_fresh_agent(self, agent_dir: Path, config: AgentConfig) -> None:
        """
        Create a fresh agent directory structure for a new user.

        Args:
            agent_dir: Path to create the agent directory
            config: Agent configuration
        """
        agent_dir.mkdir(parents=True, exist_ok=True)

        # Create openclaw.json config
        openclaw_config = DEFAULT_OPENCLAW_CONFIG.copy()
        openclaw_config["agents"] = {config.agent_name: {"model": config.model}}
        openclaw_config["defaults"] = dict(openclaw_config["defaults"])
        openclaw_config["defaults"]["agent"] = config.agent_name

        config_file = agent_dir / "openclaw.json"
        config_file.write_text(json.dumps(openclaw_config, indent=2))

        # Create agent directory structure
        agent_subdir = agent_dir / "agents" / config.agent_name
        agent_subdir.mkdir(parents=True, exist_ok=True)

        # Create SOUL.md
        soul_file = agent_subdir / "SOUL.md"
        soul_file.write_text(config.soul_content)

        # Create memory directory
        memory_dir = agent_subdir / "memory"
        memory_dir.mkdir(exist_ok=True)
        (memory_dir / "MEMORY.md").write_text("# Memories\n\nNo memories yet.\n")

        # Create sessions directory
        sessions_dir = agent_subdir / "sessions"
        sessions_dir.mkdir(exist_ok=True)

        logger.info(f"Created fresh agent directory: {agent_dir}")

    def cleanup_directory(self, directory: Path) -> None:
        """
        Securely clean up a directory.

        In production on tmpfs, this just removes the directory.
        The memory is automatically cleared when the enclave restarts.

        Args:
            directory: Path to clean up
        """
        if directory.exists():
            shutil.rmtree(directory)
            logger.debug(f"Cleaned up directory: {directory}")

    def run_agent(
        self,
        agent_dir: Path,
        message: str,
        agent_name: str,
        model: str,
        timeout_seconds: int = 120,
    ) -> AgentRunResult:
        """
        Run the OpenClaw CLI with a message.

        Args:
            agent_dir: Path to the agent's OpenClaw directory
            message: User message to send
            agent_name: Agent name to use
            model: LLM model to use
            timeout_seconds: Timeout for the CLI call

        Returns:
            AgentRunResult with success status and response
        """
        # Build environment
        # OPENCLAW_STATE_DIR is the canonical env var for OpenClaw's state directory
        # This tells OpenClaw to use agent_dir as ~/.openclaw/
        env = os.environ.copy()
        env["OPENCLAW_STATE_DIR"] = str(agent_dir)
        env["OPENCLAW_HOME"] = str(agent_dir)  # Legacy/fallback
        env["HOME"] = str(agent_dir)  # Some tools use HOME

        # Build command
        cmd = [
            self.openclaw_path,
            "agent",
            "--message",
            message,
            "--agent",
            agent_name,
            "--model",
            model,
            "--non-interactive",
        ]

        logger.info(f"Running OpenClaw: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                cwd=str(agent_dir),
            )

            if result.returncode == 0:
                return AgentRunResult(
                    success=True,
                    response=result.stdout.strip(),
                    stdout=result.stdout,
                    stderr=result.stderr,
                )
            else:
                return AgentRunResult(
                    success=False,
                    error=result.stderr.strip() or f"Exit code: {result.returncode}",
                    stdout=result.stdout,
                    stderr=result.stderr,
                )

        except subprocess.TimeoutExpired:
            return AgentRunResult(
                success=False,
                error=f"Command timeout after {timeout_seconds} seconds",
            )
        except Exception as e:
            return AgentRunResult(
                success=False,
                error=str(e),
            )
