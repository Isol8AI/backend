"""Factory for creating User test instances."""
import factory
from models.user import User


class UserFactory(factory.Factory):
    """Factory for creating User model instances for testing."""

    class Meta:
        model = User

    # Generate Clerk-like user IDs
    id = factory.Sequence(lambda n: f"user_test_{n}")
