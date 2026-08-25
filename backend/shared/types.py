"""DATORYX Shared Types and Base Classes."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Callable, Union
import uuid
import json


class TaskStatus(Enum):
    PENDING = auto()
    SCHEDULED = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()
    RETRYING = auto()
    TIMEOUT = auto()


class Priority(Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5


class MemoryType(Enum):
    SHORT_TERM = auto()
    LONG_TERM = auto()
    EPISODIC = auto()
    SEMANTIC = auto()
    VECTOR = auto()
    KNOWLEDGE_GRAPH = auto()


@dataclass
class Task:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    priority: Priority = Priority.MEDIUM
    payload: Dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: float = 30.0
    dependencies: List[str] = field(default_factory=list)
    assigned_agent: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "status": self.status.name,
            "priority": self.priority.name,
            "payload": self.payload,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "dependencies": self.dependencies,
            "assigned_agent": self.assigned_agent,
            "metadata": self.metadata
        }


@dataclass
class Event:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: str = ""
    source: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    priority: Priority = Priority.MEDIUM

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "source": self.source,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority.name
        }


@dataclass
class MemoryEntry:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: Any = None
    memory_type: MemoryType = MemoryType.SHORT_TERM
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    importance_score: float = 0.5
    associations: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": str(self.content)[:500],
            "memory_type": self.memory_type.name,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "importance_score": self.importance_score,
            "associations": self.associations
        }


@dataclass
class Goal:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    parent_id: Optional[str] = None
    sub_goals: List[str] = field(default_factory=list)
    priority: Priority = Priority.MEDIUM
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    criteria: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    deadline: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentContext:
    session_id: str = ""
    user_id: str = ""
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    current_goal: Optional[str] = None
    available_tools: List[str] = field(default_factory=list)
    working_memory: Dict[str, Any] = field(default_factory=dict)
    emotional_state: Dict[str, float] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)


class DatoryxException(Exception):
    """Base exception for DATORYX."""
    pass


class ConfigurationError(DatoryxException):
    pass


class ExecutionError(DatoryxException):
    pass


class MemoryError(DatoryxException):
    pass


class BrainError(DatoryxException):
    pass
