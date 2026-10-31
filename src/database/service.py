"""
Database Service / Repository layer.
Encapsulates CRUD operations for users, experiments, model versions, audit logs, and inference events.
"""
from __future__ import annotations

from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from src.database.models import User, Experiment, MetricRecord, ModelVersion, AuditLog, InferenceEvent, DatasetModel
from src.security.auth import hash_password


class DatabaseService:
    @staticmethod
    def get_user_by_email(db: Session, email: str) -> Optional[User]:
        return db.query(User).filter(User.email == email).first()

    @staticmethod
    def create_user(db: Session, email: str, password: str, role: str = "researcher") -> User:
        user = User(
            email=email,
            password_hash=hash_password(password),
            role=role,
            is_active=True
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user

    @staticmethod
    def log_audit(
        db: Session,
        action: str,
        actor_id: Optional[str] = None,
        actor_username: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        request_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AuditLog:
        log = AuditLog(
            action=action,
            actor_id=actor_id,
            actor_username=actor_username,
            resource_type=resource_type,
            resource_id=resource_id,
            request_id=request_id,
            metadata_json=metadata or {}
        )
        db.add(log)
        db.commit()
        return log

    @staticmethod
    def log_inference(
        db: Session,
        request_id: str,
        outcome: str,
        latency_ms: int,
        model_version_id: Optional[str] = None,
        actor_id: Optional[str] = None
    ) -> InferenceEvent:
        event = InferenceEvent(
            request_id=request_id,
            outcome=outcome,
            latency_ms=latency_ms,
            model_version_id=model_version_id,
            actor_id=actor_id
        )
        db.add(event)
        db.commit()
        return event

    @staticmethod
    def save_experiment(
        db: Session,
        name: str,
        config: Dict[str, Any],
        metrics_dict: Dict[str, Any],
        owner_id: Optional[str] = None,
        status: str = "completed"
    ) -> Experiment:
        exp = Experiment(
            name=name,
            config=config,
            owner_id=owner_id,
            status=status,
            code_version="v1.0.0"
        )
        db.add(exp)
        db.flush()

        for split_name, vals in metrics_dict.items():
            metric = MetricRecord(
                experiment_id=exp.id,
                split_name=split_name,
                values=vals
            )
            db.add(metric)

        db.commit()
        db.refresh(exp)
        return exp

    @staticmethod
    def register_model_version(
        db: Session,
        version: str,
        artifact_uri: str,
        artifact_sha256: str,
        model_card: Dict[str, Any],
        experiment_id: Optional[str] = None,
        approval_status: str = "approved_for_research"
    ) -> ModelVersion:
        existing = db.query(ModelVersion).filter(ModelVersion.version == version).first()
        if existing:
            existing.artifact_uri = artifact_uri
            existing.artifact_sha256 = artifact_sha256
            existing.model_card = model_card
            existing.approval_status = approval_status
            db.commit()
            return existing

        mv = ModelVersion(
            version=version,
            artifact_uri=artifact_uri,
            artifact_sha256=artifact_sha256,
            approval_status=approval_status,
            model_card=model_card,
            experiment_id=experiment_id
        )
        db.add(mv)
        db.commit()
        db.refresh(mv)
        return mv

    @staticmethod
    def list_experiments(db: Session, limit: int = 20) -> List[Experiment]:
        return db.query(Experiment).order_by(Experiment.created_at.desc()).limit(limit).all()

    @staticmethod
    def list_audit_logs(db: Session, limit: int = 50) -> List[AuditLog]:
        return db.query(AuditLog).order_by(AuditLog.created_at.desc()).limit(limit).all()

    @staticmethod
    def list_model_versions(db: Session) -> List[ModelVersion]:
        return db.query(ModelVersion).order_by(ModelVersion.created_at.desc()).all()
