"""DATORYX Short Term Memory - Working memory with limited capacity."""
from typing import Dict, Any, List, Optional
from collections import OrderedDict, deque
from datetime import datetime, timedelta, timezone
import time

from shared.types import MemoryEntry, MemoryType


class ShortTermMemory:
    """Limited-capacity working memory with recency and relevance decay."""

    def __init__(self, capacity: int = 100, decay_seconds: float = 300):
        self._capacity = capacity
        self._decay_seconds = decay_seconds
        self._memory: OrderedDict[str, MemoryEntry] = OrderedDict()
        self._access_history: Dict[str, List[float]] = {}

    def store(self, content: Any, metadata: Dict[str, Any] = None) -> str:
        """Store item in short-term memory."""
        entry = MemoryEntry(
            content=content,
            memory_type=MemoryType.SHORT_TERM,
            metadata=metadata or {}
        )

        # Evict oldest if at capacity
        if len(self._memory) >= self._capacity:
            self._evict_oldest()

        self._memory[entry.id] = entry
        self._memory.move_to_end(entry.id)
        self._access_history[entry.id] = [time.time()]

        return entry.id

    def retrieve(self, entry_id: str) -> Optional[MemoryEntry]:
        """Retrieve item by ID."""
        entry = self._memory.get(entry_id)
        if entry:
            entry.access_count += 1
            entry.last_accessed = datetime.now(timezone.utc)
            self._memory.move_to_end(entry_id)
            self._access_history[entry_id].append(time.time())
        return entry

    def search(self, query: str, limit: int = 5) -> List[MemoryEntry]:
        """Simple text search in STM."""
        results = []
        query_lower = query.lower()

        for entry in self._memory.values():
            content_str = str(entry.content).lower()
            metadata_str = str(entry.metadata).lower()

            if query_lower in content_str or query_lower in metadata_str:
                score = self._calculate_relevance(entry)
                results.append((score, entry))

        results.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in results[:limit]]

    def get_recent(self, limit: int = 10) -> List[MemoryEntry]:
        """Get most recently accessed items."""
        items = list(self._memory.values())
        items.sort(key=lambda e: e.last_accessed or e.created_at, reverse=True)
        return items[:limit]

    def clear(self):
        """Clear all short-term memory."""
        self._memory.clear()
        self._access_history.clear()

    def decay(self) -> int:
        """Remove items that haven't been accessed recently."""
        now = time.time()
        to_remove = []

        for entry_id, entry in self._memory.items():
            last_access = entry.last_accessed
            if last_access:
                elapsed = now - last_access.timestamp()
            else:
                elapsed = now - entry.created_at.timestamp()

            if elapsed > self._decay_seconds:
                to_remove.append(entry_id)

        for entry_id in to_remove:
            del self._memory[entry_id]
            del self._access_history[entry_id]

        return len(to_remove)

    def _evict_oldest(self):
        """Evict oldest item."""
        if self._memory:
            oldest_id, _ = self._memory.popitem(last=False)
            self._access_history.pop(oldest_id, None)

    def _calculate_relevance(self, entry: MemoryEntry) -> float:
        """Calculate relevance score."""
        recency = 1.0
        if entry.last_accessed:
            age = time.time() - entry.last_accessed.timestamp()
            recency = max(0, 1.0 - (age / self._decay_seconds))

        frequency = min(1.0, entry.access_count / 10.0)
        importance = entry.importance_score

        return (recency * 0.4) + (frequency * 0.3) + (importance * 0.3)

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "capacity": self._capacity,
            "current_size": len(self._memory),
            "utilization": len(self._memory) / self._capacity,
            "decay_threshold": self._decay_seconds
        }
