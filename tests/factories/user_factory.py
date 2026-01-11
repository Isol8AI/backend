"""Factory for creating User test instances."""
import factory

from models.user import User


class UserFactory(factory.Factory):
    """Factory for creating User model instances."""

    class Meta:
        model = User

    id = factory.Sequence(lambda n: f"user_test_{n}")
