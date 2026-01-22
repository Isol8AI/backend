"""Factory for creating Session test instances."""

import uuid

import factory

from models.session import Session


class SessionFactory(factory.Factory):
    """Factory for creating Session model instances."""

    class Meta:
        model = Session

    id = factory.LazyFunction(lambda: str(uuid.uuid4()))
    user_id = factory.Sequence(lambda n: f"user_test_{n}")
    name = factory.Sequence(lambda n: f"Test Session {n}")
