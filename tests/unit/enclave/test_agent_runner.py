"""Tests for AgentRunner (tarball packing/unpacking and CLI invocation)."""

import io
import tarfile
from unittest.mock import patch, MagicMock

import pytest

from core.enclave.agent_runner import AgentRunner, AgentConfig, AgentRunResult


class TestAgentRunner:
    """Test AgentRunner tarball operations."""

    @pytest.fixture
    def runner(self):
        """Create a runner instance for testing."""
        return AgentRunner(
            tmpfs_base="/tmp/openclaw_test",
            openclaw_path="/usr/local/bin/openclaw",
        )

    @pytest.fixture
    def sample_agent_dir(self, tmp_path):
        """Create a sample agent directory structure."""
        agent_dir = tmp_path / ".openclaw"
        agent_dir.mkdir()

        # Create config
        config_file = agent_dir / "openclaw.json"
        config_file.write_text('{"version": "1.0"}')

        # Create agent directory
        agents_dir = agent_dir / "agents" / "luna"
        agents_dir.mkdir(parents=True)

        # Create SOUL.md
        soul_file = agents_dir / "SOUL.md"
        soul_file.write_text("# Luna\nA friendly AI companion.")

        # Create memory directory
        memory_dir = agents_dir / "memory"
        memory_dir.mkdir()
        (memory_dir / "MEMORY.md").write_text("# Memories\n")

        # Create sessions directory
        sessions_dir = agents_dir / "sessions"
        sessions_dir.mkdir()
        (sessions_dir / "default.jsonl").write_text('{"role": "user", "content": "Hello"}\n')

        return agent_dir

    def test_pack_agent_directory(self, runner, sample_agent_dir):
        """Test packing an agent directory into a tarball."""
        tarball_bytes = runner.pack_directory(sample_agent_dir)

        assert isinstance(tarball_bytes, bytes)
        assert len(tarball_bytes) > 0

        # Verify it's a valid gzip tarball
        with tarfile.open(fileobj=io.BytesIO(tarball_bytes), mode="r:gz") as tar:
            names = tar.getnames()
            assert "openclaw.json" in names
            assert "agents/luna/SOUL.md" in names
            assert "agents/luna/memory/MEMORY.md" in names
            assert "agents/luna/sessions/default.jsonl" in names

    def test_unpack_tarball_to_directory(self, runner, sample_agent_dir, tmp_path):
        """Test unpacking a tarball to a directory."""
        # First pack
        tarball_bytes = runner.pack_directory(sample_agent_dir)

        # Then unpack to new location
        extract_dir = tmp_path / "extracted"
        runner.unpack_tarball(tarball_bytes, extract_dir)

        # Verify structure
        assert (extract_dir / "openclaw.json").exists()
        assert (extract_dir / "agents" / "luna" / "SOUL.md").exists()
        assert (extract_dir / "agents" / "luna" / "memory" / "MEMORY.md").exists()

        # Verify content
        soul_content = (extract_dir / "agents" / "luna" / "SOUL.md").read_text()
        assert "Luna" in soul_content

    def test_create_fresh_agent_directory(self, runner, tmp_path):
        """Test creating a fresh agent directory for new users."""
        agent_dir = tmp_path / "new_agent"
        config = AgentConfig(
            agent_name="rex",
            soul_content="# Rex\nA loyal AI companion.",
            model="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        )

        runner.create_fresh_agent(agent_dir, config)

        # Verify structure
        assert (agent_dir / "openclaw.json").exists()
        assert (agent_dir / "agents" / "rex" / "SOUL.md").exists()

        # Verify SOUL content
        soul = (agent_dir / "agents" / "rex" / "SOUL.md").read_text()
        assert "Rex" in soul
        assert "loyal AI companion" in soul

    def test_cleanup_directory(self, runner, tmp_path):
        """Test secure cleanup of tmpfs directory."""
        test_dir = tmp_path / "to_cleanup"
        test_dir.mkdir()
        (test_dir / "secret.txt").write_text("sensitive data")
        (test_dir / "subdir").mkdir()
        (test_dir / "subdir" / "nested.txt").write_text("more data")

        runner.cleanup_directory(test_dir)

        assert not test_dir.exists()

    def test_get_user_tmpfs_path(self, runner):
        """Test generating user-specific tmpfs paths."""
        path = runner.get_user_tmpfs_path("user_123")
        assert "user_123" in str(path)
        assert str(path).startswith("/tmp/openclaw_test")

    def test_get_user_tmpfs_path_sanitizes_input(self, runner):
        """Test that user IDs are sanitized to prevent path traversal."""
        # Attempt path traversal
        path = runner.get_user_tmpfs_path("../../../etc/passwd")
        # Should remove path traversal characters
        assert ".." not in str(path)
        assert "/" not in path.name  # The final component shouldn't have slashes
        # Path should stay within tmpfs_base
        assert str(path).startswith("/tmp/openclaw_test/")

    @patch("subprocess.run")
    def test_run_openclaw_cli(self, mock_run, runner, tmp_path):
        """Test running the OpenClaw CLI."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Hello! I'm Luna.",
            stderr="",
        )

        result = runner.run_agent(
            agent_dir=tmp_path,
            message="Hello!",
            agent_name="luna",
            model="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        )

        assert result.success is True
        assert "Luna" in result.response
        mock_run.assert_called_once()

        # Verify environment variables were set
        call_args = mock_run.call_args
        env = call_args.kwargs.get("env", {})
        assert "OPENCLAW_HOME" in env

    @patch("subprocess.run")
    def test_run_openclaw_cli_error(self, mock_run, runner, tmp_path):
        """Test handling CLI errors."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="Error: Model not found",
        )

        result = runner.run_agent(
            agent_dir=tmp_path,
            message="Hello!",
            agent_name="luna",
            model="invalid-model",
        )

        assert result.success is False
        assert "Error" in result.error

    @patch("subprocess.run")
    def test_run_openclaw_cli_timeout(self, mock_run, runner, tmp_path):
        """Test handling CLI timeout."""
        import subprocess

        mock_run.side_effect = subprocess.TimeoutExpired(cmd="openclaw", timeout=120)

        result = runner.run_agent(
            agent_dir=tmp_path,
            message="Hello!",
            agent_name="luna",
            model="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        )

        assert result.success is False
        assert "timeout" in result.error.lower()


class TestAgentConfig:
    """Test AgentConfig dataclass."""

    def test_default_values(self):
        """Test AgentConfig default values."""
        config = AgentConfig(agent_name="luna")

        assert config.agent_name == "luna"
        assert config.soul_content is not None
        assert "AI companion" in config.soul_content
        assert config.model is not None

    def test_custom_values(self):
        """Test AgentConfig with custom values."""
        config = AgentConfig(
            agent_name="rex",
            soul_content="# Custom Soul",
            model="custom-model",
        )

        assert config.agent_name == "rex"
        assert config.soul_content == "# Custom Soul"
        assert config.model == "custom-model"


class TestAgentRunResult:
    """Test AgentRunResult dataclass."""

    def test_success_result(self):
        """Test successful run result."""
        result = AgentRunResult(
            success=True,
            response="Hello!",
        )
        assert result.success is True
        assert result.response == "Hello!"
        assert result.error == ""

    def test_error_result(self):
        """Test error run result."""
        result = AgentRunResult(
            success=False,
            error="Something went wrong",
        )
        assert result.success is False
        assert result.error == "Something went wrong"
