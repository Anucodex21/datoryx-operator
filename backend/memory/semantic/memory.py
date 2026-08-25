"""DATORYX Semantic Memory - Structured knowledge and facts.

Persisted to SQLite/PostgreSQL via core.persistence so learned concepts
and facts survive process restarts.
"""
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from datetime import datetime, timezone

from shared.types import MemoryEntry, MemoryType
from core.persistence import session_scope, init_db
from core.persistence.models import ConceptRecord


@dataclass
class Concept:
    """A semantic concept with properties and relationships."""
    name: str
    category: str
    properties: Dict[str, Any]
    definition: str = ""
    related_concepts: List[str] = None
    instances: List[str] = None

    def __post_init__(self):
        if self.related_concepts is None:
            self.related_concepts = []
        if self.instances is None:
            self.instances = []


class SemanticMemory:
    """Structured knowledge base with concepts and categories."""

    def __init__(self, user_id: str = "public"):
        self.user_id = user_id or "public"
        self._concepts: Dict[str, Concept] = {}
        self._categories: Dict[str, Set[str]] = {}  # category -> concept names
        self._fact_index: Dict[str, List[str]] = {}  # keyword -> concept names
        init_db()
        self._load()

    def define_concept(self, name: str, category: str,
                        properties: Dict[str, Any],
                        definition: str = "",
                        related: List[str] = None) -> Concept:
        """Define a new concept."""
        concept = Concept(
            name=name,
            category=category,
            properties=properties,
            definition=definition,
            related_concepts=related or []
        )

        self._concepts[name] = concept

        if category not in self._categories:
            self._categories[category] = set()
        self._categories[category].add(name)

        self._index_concept(concept)
        self._persist_concept(concept)

        return concept

    def learn_fact(self, subject: str, predicate: str, object_val: Any,
                   confidence: float = 1.0) -> bool:
        """Learn a new fact about a concept."""
        concept = self._concepts.get(subject)
        if not concept:
            return False

        concept.properties[predicate] = {
            "value": object_val,
            "confidence": confidence,
            "learned_at": str(datetime.now(timezone.utc))
        }

        self._index_concept(concept)
        self._persist_concept(concept)
        return True

    def query(self, concept_name: str) -> Optional[Concept]:
        """Query a concept by name."""
        return self._concepts.get(concept_name)

    def query_by_category(self, category: str) -> List[Concept]:
        """Get all concepts in a category."""
        names = self._categories.get(category, set())
        return [self._concepts[n] for n in names if n in self._concepts]

    def search(self, query: str) -> List[Concept]:
        """Search concepts by keyword."""
        query_lower = query.lower()
        results = []

        for name, concept in self._concepts.items():
            score = 0
            if query_lower in name.lower():
                score += 3
            if query_lower in concept.definition.lower():
                score += 2
            if query_lower in concept.category.lower():
                score += 1

            for prop_key, prop_val in concept.properties.items():
                val_str = str(prop_val).lower()
                if query_lower in val_str or query_lower in prop_key.lower():
                    score += 1

            if score > 0:
                results.append((score, concept))

        results.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in results]

    def infer(self, concept_name: str, property_name: str) -> Optional[Any]:
        """Try to infer a property using related concepts."""
        concept = self._concepts.get(concept_name)
        if not concept:
            return None

        if property_name in concept.properties:
            return concept.properties[property_name]

        for related_name in concept.related_concepts:
            related = self._concepts.get(related_name)
            if related and property_name in related.properties:
                return {
                    "inferred": True,
                    "source": related_name,
                    "value": related.properties[property_name]
                }

        return None

    def get_relationships(self, concept_name: str) -> Dict[str, List[str]]:
        """Get all relationships for a concept."""
        concept = self._concepts.get(concept_name)
        if not concept:
            return {}

        relationships = {
            "related": concept.related_concepts,
            "category": [concept.category],
            "instances": concept.instances
        }

        parents = []
        for name, other in self._concepts.items():
            if concept_name in other.related_concepts:
                parents.append(name)
        relationships["parents"] = parents

        return relationships

    def _index_concept(self, concept: Concept):
        """Index concept for search."""
        text = f"{concept.name} {concept.definition} {concept.category}"
        for prop in concept.properties:
            text += f" {prop}"

        words = set(text.lower().split())
        for word in words:
            if word not in self._fact_index:
                self._fact_index[word] = []
            if concept.name not in self._fact_index[word]:
                self._fact_index[word].append(concept.name)

    def _persist_concept(self, concept: Concept):
        try:
            with session_scope() as session:
                record = session.get(ConceptRecord, (concept.name, self.user_id))
                if record is None:
                    record = ConceptRecord(name=concept.name, user_id=self.user_id)
                record.category = concept.category
                record.properties = concept.properties
                record.definition = concept.definition
                record.related_concepts = concept.related_concepts
                record.instances = concept.instances
                session.merge(record)
        except Exception as e:
            print(f"[Semantic] Persist error: {e}")

    def _load(self):
        try:
            with session_scope() as session:
                records = session.query(ConceptRecord).filter(
                    ConceptRecord.user_id == self.user_id
                ).all()
                for record in records:
                    concept = Concept(
                        name=record.name,
                        category=record.category,
                        properties=record.properties or {},
                        definition=record.definition or "",
                        related_concepts=record.related_concepts or [],
                        instances=record.instances or []
                    )
                    self._concepts[concept.name] = concept
                    self._categories.setdefault(concept.category, set()).add(concept.name)
                    self._index_concept(concept)
        except Exception as e:
            print(f"[Semantic] Load error: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get semantic memory statistics."""
        total_facts = sum(len(c.properties) for c in self._concepts.values())
        return {
            "concepts": len(self._concepts),
            "categories": len(self._categories),
            "total_facts": total_facts,
            "avg_facts_per_concept": total_facts / max(len(self._concepts), 1)
        }
