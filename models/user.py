from sqlalchemy import String, Column
from .base import Base

class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True)  # Clerk User ID
