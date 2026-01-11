"""Factory for creating Message test instances."""
import uuid
from datetime import datetime
import factory
from models.message import Message, MessageRole


class MessageFactory(factory.Factory):
    """Factory for creating Message model instances for testing."""

    class Meta:
        model = Message

    id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    session_id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    role = MessageRole.USER.value
    content = factory.Sequence(lambda n: f"Test message content {n}")
    model_used = None
    timestamp = factory.LazyFunction(datetime.utcnow)


class UserMessageFactory(MessageFactory):
    """Factory specifically for user messages."""

    role = MessageRole.USER.value
    model_used = None


class AssistantMessageFactory(MessageFactory):
    """Factory specifically for assistant messages."""

    role = MessageRole.ASSISTANT.value
    model_used = "Qwen/Qwen2.5-72B-Instruct"
    content = factory.Sequence(lambda n: f"Assistant response {n}")
