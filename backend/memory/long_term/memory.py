"""DATORYX Long Term Memory - Persistent storage with importance-based retention.

Backed by SQLite (or PostgreSQL, via DATORYX_DATABASE_URL) through
core.persistence. Previously this used a hand-rolled JSON file whose
_load() only restored the keyword index, not the actual memory content -
meaning every restart silently lost all stored memories while still
reporting a populated index. That bug is fixed here: memories are now
the source of truth in the database and are fully reconstructed on init.
"""
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from shared.types import MemoryEntry, MemoryType
from core.persistence import session_scope, init_db
from core.persistence.models import MemoryRecord


class LongTermMemory:
    """Persistent memory with consolidation from short-term."""

    def __init__(self, storage_path: str = None, user_id: str = "public"):
        # storage_path kept as an accepted (now unused) parameter for backward
        # compatibility with existing callers/config; real location is governed
        # by DATORYX_DATABASE_URL.
        self._storage_path = storage_path
        self.user_id = user_id or "public"
        init_db()
        self._memories: Dict[str, MemoryEntry] = {}
        self._index: Dict[str, List[str]] = {}  # keyword -> entry_ids
        self._load()

    def store(self, content: Any, importance: float = 0.5,
              metadata: Dict[str, Any] = None) -> str:
        """Store item in long-term memory."""
        entry = MemoryEntry(
            content=content,
            memory_type=MemoryType.LONG_TERM,
            importance_score=importance,
            metadata=metadata or {}
        )

        self._memories[entry.id] = entry
        self._index_entry(entry)
        self._persist_entry(entry)

        return entry.id

    def retrieve(self, entry_id: str) -> Optional[MemoryEntry]:
        """Retrieve item by ID."""
        entry = self._memories.get(entry_id)
        if entry:
            entry.access_count += 1
            entry.last_accessed = datetime.now(timezone.utc)
            self._persist_entry(entry)
        return entry

    def search(self, query: str, limit: int = 10) -> List[MemoryEntry]:
        """Search by keywords."""
        query_words = query.lower().split()
        scores: Dict[str, float] = {}

        for word in query_words:
            for entry_id in self._index.get(word, []):
                scores[entry_id] = scores.get(entry_id, 0) + 1

        results = []
        for entry_id, score in scores.items():
            entry = self._memories.get(entry_id)
            if entry:
                final_score = score * entry.importance_score
                results.append((final_score, entry))

        results.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in results[:limit]]

    def consolidate(self, stm_entries: List[MemoryEntry]) -> List[str]:
        """Consolidate short-term memories into long-term."""
        consolidated = []
        for entry in stm_entries:
            if entry.access_count >= 2 or entry.importance_score > 0.7:
                entry.memory_type = MemoryType.LONG_TERM
                self._memories[entry.id] = entry
                self._index_entry(entry)
                self._persist_entry(entry)
                consolidated.append(entry.id)

        return consolidated

    def forget(self, entry_id: str) -> bool:
        """Remove a memory."""
        entry = self._memories.pop(entry_id, None)
        if entry:
            self._remove_from_index(entry)
            with session_scope() as session:
                session.query(MemoryRecord).filter(
                    MemoryRecord.id == entry_id, MemoryRecord.user_id == self.user_id
                ).delete()
            return True
        return False

    def get_by_importance(self, min_score: float = 0.5, limit: int = 20) -> List[MemoryEntry]:
        """Get memories by importance threshold."""
        results = [
            entry for entry in self._memories.values()
            if entry.importance_score >= min_score
        ]
        results.sort(key=lambda e: e.importance_score, reverse=True)
        return results[:limit]

    def _index_entry(self, entry: MemoryEntry):
        """Index entry by keywords."""
        content_str = str(entry.content).lower()
        words = set(content_str.split())

        for word in words:
            if word not in self._index:
                self._index[word] = []
            if entry.id not in self._index[word]:
                self._index[word].append(entry.id)

    def _remove_from_index(self, entry: MemoryEntry):
        """Remove entry from index."""
        content_str = str(entry.content).lower()
        words = set(content_str.split())

        for word in words:
            if word in self._index and entry.id in self._index[word]:
                self._index[word].remove(entry.id)

    def _persist_entry(self, entry: MemoryEntry):
        """Upsert a single entry to the database."""
        try:
            with session_scope() as session:
                record = session.get(MemoryRecord, entry.id)
                if record is None:
                    record = MemoryRecord(id=entry.id, user_id=self.user_id)
                record.user_id = self.user_id
                record.memory_type = entry.memory_type.name
                record.content = _safe_serialize(entry.content)
                record.content_repr = str(entry.content)[:2000]
                record.entry_metadata = entry.metadata
                record.importance_score = entry.importance_score
                record.access_count = entry.access_count
                record.created_at = entry.created_at
                record.last_accessed = entry.last_accessed
                record.associations = entry.associations
                session.merge(record)
        except Exception as e:
            print(f"[LTM] Persist error: {e}")

    def _load(self):
        """Load all memories from the database into the in-memory cache."""
        try:
            with session_scope() as session:
                records = session.query(MemoryRecord).filter(
                    MemoryRecord.memory_type == MemoryType.LONG_TERM.name,
                    MemoryRecord.user_id == self.user_id,
                ).all()
                for record in records:
                    entry = MemoryEntry(
                        id=record.id,
                        content=_safe_deserialize(record.content),
                        memory_type=MemoryType.LONG_TERM,
                        metadata=record.entry_metadata or {},
                        created_at=record.created_at,
                        access_count=record.access_count or 0,
                        last_accessed=record.last_accessed,
                        importance_score=record.importance_score or 0.5,
                        associations=record.associations or []
                    )
                    self._memories[entry.id] = entry
                    self._index_entry(entry)
        except Exception as e:
            print(f"[LTM] Load error: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "total_memories": len(self._memories),
            "indexed_words": len(self._index),
            "avg_importance": sum(e.importance_score for e in self._memories.values()) / max(len(self._memories), 1)
        }


def _safe_serialize(content: Any) -> str:
    """Serialize content to a JSON string when possible, else str()."""
    import json
    try:
        return json.dumps(content)
    except (TypeError, ValueError):
        return str(content)


def _safe_deserialize(raw: str) -> Any:
    import json
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return raw
