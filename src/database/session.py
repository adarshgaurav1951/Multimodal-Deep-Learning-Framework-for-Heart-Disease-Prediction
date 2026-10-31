"""
Database Engine, Session Management, and Initialization.
"""
from __future__ import annotations

import os
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from src.config import settings
from src.database.models import Base, User
from src.security.auth import hash_password

from sqlalchemy.pool import StaticPool

# SQLite connects with check_same_thread=False
connect_args = {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}
engine_kwargs = {"connect_args": connect_args, "echo": False}
if ":memory:" in settings.DATABASE_URL:
    engine_kwargs["poolclass"] = StaticPool

engine = create_engine(
    settings.DATABASE_URL,
    **engine_kwargs
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI database session dependency."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables and seed initial administrator and researcher users."""
    Base.metadata.create_all(bind=engine)

    with SessionLocal() as db:
        admin_email = "admin@research.cardio.ai"
        existing_admin = db.query(User).filter(User.email == admin_email).first()
        if not existing_admin:
            admin = User(
                email=admin_email,
                password_hash=hash_password("AdminSecure2026!"),
                role="admin",
                is_active=True
            )
            db.add(admin)

        researcher_email = "researcher@cardio.ai"
        existing_res = db.query(User).filter(User.email == researcher_email).first()
        if not existing_res:
            res = User(
                email=researcher_email,
                password_hash=hash_password("Research2026!"),
                role="researcher",
                is_active=True
            )
            db.add(res)

        client_email = "api_client@cardio.ai"
        existing_client = db.query(User).filter(User.email == client_email).first()
        if not existing_client:
            cli = User(
                email=client_email,
                password_hash=hash_password("ApiClient2026!"),
                role="api_client",
                is_active=True
            )
            db.add(cli)

        db.commit()
