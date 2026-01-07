from sqlalchemy import String, Column
from .base import Base

class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True)  # Clerk ID (sub)
    encrypted_key = Column(String, nullable=False) # Base64 encoded encrypted key
