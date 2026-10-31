"""
Cryptographic utility functions.
"""
import hashlib
import uuid


def hash_record_id(raw_id: str, salt: str = "cardio_salt_2026") -> str:
    """Generate deterministic pseudonymous record hash."""
    return hashlib.sha256(f"{raw_id}:{salt}".encode("utf-8")).hexdigest()[:16]


def generate_request_id() -> str:
    """Generate unique UUID4 request identifier for audit tracking."""
    return str(uuid.uuid4())
