"""
SQLAlchemy Database Models for Research Management and Audit Logging.
Supports PostgreSQL and SQLite.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Boolean, DateTime, Integer, Text, ForeignKey, JSON
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def generate_uuid() -> str:
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False, default="researcher")  # admin, researcher, reviewer, api_client
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    experiments = relationship("Experiment", back_populates="owner")
    audit_logs = relationship("AuditLog", back_populates="actor")
    inference_events = relationship("InferenceEvent", back_populates="actor")


class DatasetModel(Base):
    __tablename__ = "datasets"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(128), nullable=False)
    version = Column(String(64), nullable=False)
    manifest_hash = Column(String(64), nullable=False)
    storage_uri = Column(String(512), nullable=True)
    data_dictionary_uri = Column(String(512), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    experiments = relationship("Experiment", back_populates="dataset")


class Experiment(Base):
    __tablename__ = "experiments"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    owner_id = Column(String(36), ForeignKey("users.id"), nullable=True)
    dataset_id = Column(String(36), ForeignKey("datasets.id"), nullable=True)
    name = Column(String(128), nullable=False)
    status = Column(String(50), nullable=False, default="completed")  # draft, running, completed, failed
    config = Column(JSON, nullable=False, default=dict)
    code_version = Column(String(64), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    owner = relationship("User", back_populates="experiments")
    dataset = relationship("DatasetModel", back_populates="experiments")
    metrics = relationship("MetricRecord", back_populates="experiment", cascade="all, delete-orphan")
    model_versions = relationship("ModelVersion", back_populates="experiment")


class MetricRecord(Base):
    __tablename__ = "metrics"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    experiment_id = Column(String(36), ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False)
    split_name = Column(String(50), nullable=False)  # train, val, test
    values = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    experiment = relationship("Experiment", back_populates="metrics")


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    experiment_id = Column(String(36), ForeignKey("experiments.id"), nullable=True)
    version = Column(String(64), unique=True, nullable=False, index=True)
    artifact_uri = Column(String(512), nullable=False)
    artifact_sha256 = Column(String(64), nullable=False)
    approval_status = Column(String(50), nullable=False, default="candidate")  # candidate, approved_for_research, rejected
    model_card = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    experiment = relationship("Experiment", back_populates="model_versions")
    inference_events = relationship("InferenceEvent", back_populates="model_version")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    actor_id = Column(String(36), ForeignKey("users.id"), nullable=True)
    actor_username = Column(String(255), nullable=True)
    action = Column(String(128), nullable=False)
    resource_type = Column(String(64), nullable=True)
    resource_id = Column(String(128), nullable=True)
    request_id = Column(String(64), nullable=True)
    metadata_json = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    actor = relationship("User", back_populates="audit_logs")


class InferenceEvent(Base):
    __tablename__ = "inference_events"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    model_version_id = Column(String(36), ForeignKey("model_versions.id"), nullable=True)
    actor_id = Column(String(36), ForeignKey("users.id"), nullable=True)
    request_id = Column(String(64), nullable=False, index=True)
    outcome = Column(String(64), nullable=False)  # success, error, model_not_loaded
    latency_ms = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    model_version = relationship("ModelVersion", back_populates="inference_events")
    actor = relationship("User", back_populates="inference_events")
