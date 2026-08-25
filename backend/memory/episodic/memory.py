"""DATORYX Episodic Memory - Event-based memory with temporal context.

Persisted to SQLite/PostgreSQL via core.persistence so experiences
survive process restarts (previously this was pure in-RAM and lost
everything on shutdown).

Consolidation (see consolidate()) is LLM-backed: periodically, batches of
raw episodes are distilled into durable semantic facts and knowledge-graph
entities/relationships - the way sleep consolidates episodic memory into
semantic knowledge - rather than just accumulating raw events forever.
"""
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta, timezone
import time

from shared.types import MemoryEntry, MemoryType
from shared.llm_helpers import llm_json, get_default_llm
from core.persistence import session_scope, init_db
from core.persistence.models import EpisodeRecord


class Episode:
    """Represents a single episode/experience."""

    def __init__(self, event: str, context: Dict[str, Any],
                 emotions: Dict[str, float] = None,
                 people: List[str] = None, location: str = "",
                 id: str = None, timestamp: datetime = None, replay_count: int = 0,
                 consolidated: bool = False):
        self.id = id or str(hash(f"{event}{time.time()}"))
        self.event = event
        self.context = context
        self.emotions = emotions or {}
        self.people = people or []
        self.location = location
        self.timestamp = timestamp or datetime.now(timezone.utc)
        self.replay_count = replay_count
        self.consolidated = consolidated
        self.associated_episodes: List[str] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "event": self.event,
            "context": self.context,
            "emotions": self.emotions,
            "people": self.people,
            "location": self.location,
            "timestamp": self.timestamp.isoformat(),
            "replay_count": self.replay_count,
            "consolidated": self.consolidated
        }


class EpisodicMemory:
    """Memory for experiences and events with temporal sequencing."""

    def __init__(self, capacity: int = 500, llm: Any = None, user_id: str = "public"):
        self.llm = llm or get_default_llm()
        self.user_id = user_id or "public"
        self._capacity = capacity
        self._episodes: Dict[str, Episode] = {}
        self._timeline: List[str] = []  # Ordered episode IDs
        self._emotion_index: Dict[str, List[str]] = {}  # emotion -> episode_ids
        init_db()
        self._load()

    def record_episode(self, event: str, context: Dict[str, Any] = None,
                        emotions: Dict[str, float] = None,
                        people: List[str] = None, location: str = "") -> str:
        """Record a new episode."""
        episode = Episode(event, context or {}, emotions, people, location)

        # Manage capacity (in-memory working set only; DB keeps full history)
        if len(self._episodes) >= self._capacity:
            oldest_id = self._timeline.pop(0)
            del self._episodes[oldest_id]

        self._episodes[episode.id] = episode
        self._timeline.append(episode.id)

        if emotions:
            for emotion, intensity in emotions.items():
                if intensity > 0.5:
                    if emotion not in self._emotion_index:
                        self._emotion_index[emotion] = []
                    self._emotion_index[emotion].append(episode.id)

        self._persist_episode(episode)
        return episode.id

    def recall(self, episode_id: str) -> Optional[Episode]:
        """Recall a specific episode."""
        episode = self._episodes.get(episode_id)
        if episode:
            episode.replay_count += 1
            self._persist_episode(episode)
        return episode

    def recall_by_time(self, start: datetime, end: datetime) -> List[Episode]:
        """Recall episodes within a time range."""
        return [
            self._episodes[eid] for eid in self._timeline
            if eid in self._episodes and start <= self._episodes[eid].timestamp <= end
        ]

    def recall_by_emotion(self, emotion: str, min_intensity: float = 0.5) -> List[Episode]:
        """Recall episodes by emotional context."""
        return [
            self._episodes[eid] for eid in self._emotion_index.get(emotion, [])
            if eid in self._episodes and self._episodes[eid].emotions.get(emotion, 0) >= min_intensity
        ]

    def recall_similar(self, episode_id: str, limit: int = 5) -> List[Episode]:
        """Find similar episodes based on context overlap."""
        source = self._episodes.get(episode_id)
        if not source:
            return []

        scores = []
        for eid, episode in self._episodes.items():
            if eid == episode_id:
                continue
            score = self._similarity_score(source, episode)
            scores.append((score, episode))

        scores.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scores[:limit]]

    def get_recent_episodes(self, limit: int = 10) -> List[Episode]:
        """Get most recent episodes."""
        recent_ids = self._timeline[-limit:]
        return [self._episodes[eid] for eid in recent_ids if eid in self._episodes]

    def get_life_narrative(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Generate a narrative of recent experiences."""
        episodes = self.get_recent_episodes(limit)
        return [
            {
                "time": ep.timestamp.isoformat(),
                "event": ep.event,
                "location": ep.location,
                "emotions": ep.emotions,
                "replay_count": ep.replay_count
            }
            for ep in episodes
        ]

    def consolidate(self, semantic_memory: Any = None, knowledge_graph: Any = None,
                     batch_size: int = 20) -> Dict[str, Any]:
        """Consolidate a batch of not-yet-consolidated episodes into durable
        knowledge, the way sleep consolidates episodic memory into semantic
        memory: the LLM reads the raw episodes, extracts recurring facts and
        entities, and those get written into semantic_memory (facts) and
        knowledge_graph (entities/relationships) if provided. Processed
        episodes are marked consolidated so repeated calls make progress
        through the backlog instead of reprocessing everything each time.
        """
        pending = [self._episodes[eid] for eid in self._timeline
                   if eid in self._episodes and not self._episodes[eid].consolidated][:batch_size]

        if not pending:
            return {"consolidated": 0, "summary": "", "facts_learned": 0, "graph_updates": 0}

        events_text = "\n".join(
            f"- [{ep.timestamp.isoformat()}] {ep.event} "
            f"(people: {ep.people or 'none'}, location: {ep.location or 'unknown'}, "
            f"context: {ep.context})"
            for ep in pending
        )

        system_prompt = (
            "You are the memory consolidation process inside DATORYX, distilling raw episodic "
            "events into durable semantic knowledge - the way sleep consolidates a day's "
            "experiences into lasting memory. Extract facts that generalize beyond this one moment, "
            "and a short summary. Don't restate every event - synthesize."
        )
        user_prompt = (
            f"Episodes to consolidate:\n{events_text}\n\n"
            'Return JSON: {"summary": "<2-3 sentence synthesis of this batch of episodes>", '
            '"facts": [{"subject": "...", "category": "...", "predicate": "...", "object": "...", '
            '"confidence": <0-1 float>}], '
            '"entities": [{"label": "...", "type": "person|organization|location|concept|other"}], '
            '"relationships": [{"source_label": "...", "relation": "...", "target_label": "..."}]}'
        )
        result = llm_json(self.llm, system_prompt, user_prompt)

        facts_learned = 0
        if semantic_memory is not None:
            for fact in result.get("facts", []):
                subject = fact.get("subject", "").strip()
                predicate = fact.get("predicate", "").strip()
                if not subject or not predicate:
                    continue
                if semantic_memory.query(subject) is None:
                    semantic_memory.define_concept(subject, fact.get("category", "entity"), {})
                if semantic_memory.learn_fact(subject, predicate, fact.get("object"),
                                               confidence=float(fact.get("confidence", 0.7))):
                    facts_learned += 1

        graph_updates = 0
        if knowledge_graph is not None:
            for entity in result.get("entities", []):
                label = entity.get("label", "").strip()
                if not label:
                    continue
                node_id = knowledge_graph._slugify(label) if hasattr(knowledge_graph, "_slugify") else label.lower().replace(" ", "_")
                knowledge_graph.add_node(node_id, label, node_type=entity.get("type", "entity"))
                graph_updates += 1
            for rel in result.get("relationships", []):
                src = rel.get("source_label", "").strip()
                tgt = rel.get("target_label", "").strip()
                if not src or not tgt:
                    continue
                src_id = knowledge_graph._slugify(src) if hasattr(knowledge_graph, "_slugify") else src.lower().replace(" ", "_")
                tgt_id = knowledge_graph._slugify(tgt) if hasattr(knowledge_graph, "_slugify") else tgt.lower().replace(" ", "_")
                if src_id not in knowledge_graph._nodes:
                    knowledge_graph.add_node(src_id, src)
                if tgt_id not in knowledge_graph._nodes:
                    knowledge_graph.add_node(tgt_id, tgt)
                knowledge_graph.add_edge(src_id, tgt_id, rel.get("relation", "related_to"))
                graph_updates += 1

        for ep in pending:
            ep.consolidated = True
            self._persist_episode(ep)

        return {
            "consolidated": len(pending),
            "summary": result.get("summary", ""),
            "facts_learned": facts_learned,
            "graph_updates": graph_updates,
            "remaining_backlog": len([e for e in self._episodes.values() if not e.consolidated])
        }

    def _similarity_score(self, ep1: Episode, ep2: Episode) -> float:
        """Calculate similarity between two episodes."""
        score = 0.0

        if ep1.context and ep2.context:
            shared_keys = set(ep1.context.keys()) & set(ep2.context.keys())
            score += len(shared_keys) * 0.2

        if ep1.people and ep2.people:
            shared_people = set(ep1.people) & set(ep2.people)
            score += len(shared_people) * 0.3

        if ep1.location and ep1.location == ep2.location:
            score += 0.2

        if ep1.emotions and ep2.emotions:
            shared_emotions = set(ep1.emotions.keys()) & set(ep2.emotions.keys())
            score += len(shared_emotions) * 0.1

        time_diff = abs((ep1.timestamp - ep2.timestamp).total_seconds())
        if time_diff < 3600:
            score += 0.2

        return min(1.0, score)

    def _persist_episode(self, episode: Episode):
        try:
            with session_scope() as session:
                record = session.get(EpisodeRecord, episode.id)
                if record is None:
                    record = EpisodeRecord(id=episode.id, user_id=self.user_id)
                record.user_id = self.user_id
                record.event = episode.event
                record.context = episode.context
                record.emotions = episode.emotions
                record.people = episode.people
                record.location = episode.location
                record.timestamp = episode.timestamp
                record.replay_count = episode.replay_count
                record.consolidated = episode.consolidated
                session.merge(record)
        except Exception as e:
            print(f"[Episodic] Persist error: {e}")

    def _load(self):
        """Restore episodes from the database, most recent `capacity` first."""
        try:
            with session_scope() as session:
                records = (
                    session.query(EpisodeRecord)
                    .filter(EpisodeRecord.user_id == self.user_id)
                    .order_by(EpisodeRecord.timestamp.asc())
                    .all()
                )
                records = records[-self._capacity:]
                for record in records:
                    episode = Episode(
                        event=record.event,
                        context=record.context or {},
                        emotions=record.emotions or {},
                        people=record.people or [],
                        location=record.location or "",
                        id=record.id,
                        timestamp=record.timestamp,
                        replay_count=record.replay_count or 0,
                        consolidated=bool(record.consolidated)
                    )
                    self._episodes[episode.id] = episode
                    self._timeline.append(episode.id)
                    for emotion, intensity in episode.emotions.items():
                        if intensity > 0.5:
                            self._emotion_index.setdefault(emotion, []).append(episode.id)
        except Exception as e:
            print(f"[Episodic] Load error: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get episodic memory statistics."""
        return {
            "total_episodes": len(self._episodes),
            "capacity": self._capacity,
            "timeline_length": len(self._timeline),
            "emotion_types": len(self._emotion_index),
            "avg_replay_count": sum(ep.replay_count for ep in self._episodes.values()) / max(len(self._episodes), 1),
            "unconsolidated": len([e for e in self._episodes.values() if not e.consolidated])
        }
