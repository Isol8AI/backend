"""
AgentState model for storing encrypted OpenClaw agent state.

Each user's agent state is stored as an encrypted tarball containing
the complete ~/.openclaw directory structure (config, SOUL.md, memory,
sessions). The server cannot read this data - only the enclave can
decrypt it.
"""

from datetime import datetime, timezone
import uuid

from sqlalchemy import (
    Column,
    String,
    LargeBinary,
    Integer,
    DateTime,
    UniqueConstraint,
    Index,
)
from sqlalchemy.dialects.postgresql import UUID

from models.base import Base


class AgentState(Base):
    """
    Encrypted agent state storage.

    Stores the complete OpenClaw agent directory as an encrypted tarball.
    The tarball is encrypted to the enclave's public key so only the
    enclave can decrypt and process it.

    Attributes:
        id: Unique identifier (UUID) - primary key
        user_id: Clerk user ID who owns this agent
        agent_name: User-chosen name for the agent (e.g., "luna", "rex")
        encrypted_tarball: Encrypted tarball of ~/.openclaw directory
        tarball_size_bytes: Size for monitoring/quotas
        created_at: When the agent was first created
        updated_at: When the agent state was last updated
    """

    __tablename__ = "agent_states"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    user_id = Column(String, nullable=False, index=True)
    agent_name = Column(String, nullable=False)

    # Encrypted tarball containing the agent's ~/.openclaw directory
    # Nullable: tarball is None until first message (enclave creates fresh state)
    encrypted_tarball = Column(LargeBinary, nullable=True)

    # Metadata (not encrypted - needed for queries and quotas)
    tarball_size_bytes = Column(Integer, nullable=True)

    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint("user_id", "agent_name", name="uq_agent_states_user_agent"),
        Index("idx_agent_states_user", "user_id"),
    )

    def __repr__(self) -> str:
        return f"<AgentState(id={self.id}, user_id={self.user_id}, agent_name={self.agent_name})>"
