"""DATORYX Knowledge Graph - Graph-based knowledge representation.

Persisted to SQLite/PostgreSQL via core.persistence so nodes and
relationships survive process restarts. Entity/relationship extraction from
raw text is LLM-backed (see ingest_text) - previously the only way to
populate the graph was to call add_node/add_edge by hand with pre-parsed IDs.
"""
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import re

from core.persistence import session_scope, init_db
from core.persistence.models import GraphNodeRecord, GraphEdgeRecord
from shared.llm_helpers import llm_json, get_default_llm


@dataclass
class Node:
    """Graph node representing an entity."""
    id: str
    label: str
    properties: Dict[str, Any] = field(default_factory=dict)
    node_type: str = "entity"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "type": self.node_type,
            "properties": self.properties
        }


@dataclass
class Edge:
    """Graph edge representing a relationship."""
    source: str
    target: str
    relation: str
    properties: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "weight": self.weight,
            "properties": self.properties
        }


class KnowledgeGraph:
    """Property graph for knowledge representation and reasoning."""

    def __init__(self, llm: Any = None, user_id: str = "public"):
        self.llm = llm or get_default_llm()
        self.user_id = user_id or "public"
        self._nodes: Dict[str, Node] = {}
        self._edges: Dict[str, List[Edge]] = defaultdict(list)  # node_id -> edges
        self._relations: Dict[str, List[Edge]] = defaultdict(list)  # relation -> edges
        self._adjacency: Dict[str, Set[str]] = defaultdict(set)
        init_db()
        self._load()

    def _slugify(self, label: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
        return slug or "entity"

    def ingest_text(self, text: str, source: str = "") -> Dict[str, Any]:
        """Extract entities and relationships from raw text using the LLM
        and add them to the graph. This is the real entry point for growing
        the graph from unstructured input (documents, conversation turns,
        episodic memory, etc.) instead of requiring hand-written node/edge
        calls with pre-decided IDs."""
        system_prompt = (
            "You are the knowledge extraction component of DATORYX's knowledge graph. Extract "
            "concrete named entities and the relationships between them from the given text. Only "
            "extract what the text actually supports - do not invent entities or relations that "
            "aren't there."
        )
        user_prompt = (
            f"Text:\n{text}\n\n"
            'Return JSON: {"entities": [{"label": "...", "type": "person|organization|location|'
            'concept|event|product|other", "properties": {...}}], '
            '"relationships": [{"source_label": "...", "relation": "<lowercase_snake_case verb '
            'phrase, e.g. works_at, located_in, causes>", "target_label": "...", '
            '"confidence": <0-1 float>}]}'
        )
        result = llm_json(self.llm, system_prompt, user_prompt)

        entities = result.get("entities", [])
        relationships = result.get("relationships", [])

        label_to_id: Dict[str, str] = {}
        nodes_created = []
        for entity in entities:
            label = entity.get("label", "").strip()
            if not label:
                continue
            node_id = self._slugify(label)
            # Avoid clobbering an existing distinct node with the same slug.
            suffix = 1
            base_id = node_id
            while node_id in self._nodes and self._nodes[node_id].label.lower() != label.lower():
                suffix += 1
                node_id = f"{base_id}_{suffix}"

            properties = entity.get("properties", {}) or {}
            if source:
                properties.setdefault("sources", [])
                if source not in properties["sources"]:
                    properties["sources"].append(source)

            self.add_node(node_id, label, node_type=entity.get("type", "entity"), properties=properties)
            label_to_id[label.lower()] = node_id
            nodes_created.append(node_id)

        edges_created = []
        for rel in relationships:
            src_label = rel.get("source_label", "").strip().lower()
            tgt_label = rel.get("target_label", "").strip().lower()
            src_id = label_to_id.get(src_label) or self._slugify(src_label)
            tgt_id = label_to_id.get(tgt_label) or self._slugify(tgt_label)

            # If a relationship references an entity that wasn't in the extracted
            # entity list (the model forgot to list it separately), create a
            # minimal node for it rather than dropping the relationship.
            if src_id not in self._nodes and src_label:
                self.add_node(src_id, rel.get("source_label", src_label))
                nodes_created.append(src_id)
            if tgt_id not in self._nodes and tgt_label:
                self.add_node(tgt_id, rel.get("target_label", tgt_label))
                nodes_created.append(tgt_id)

            if src_id in self._nodes and tgt_id in self._nodes:
                self.add_edge(src_id, tgt_id, rel.get("relation", "related_to"),
                              properties={"source_text": source} if source else {},
                              weight=float(rel.get("confidence", 1.0)))
                edges_created.append((src_id, rel.get("relation", "related_to"), tgt_id))

        return {
            "nodes_created": nodes_created,
            "edges_created": edges_created,
            "entities_found": len(entities),
            "relationships_found": len(relationships)
        }

    def add_node(self, node_id: str, label: str,
                 node_type: str = "entity",
                 properties: Dict[str, Any] = None) -> Node:
        """Add a node to the graph."""
        node = Node(
            id=node_id,
            label=label,
            node_type=node_type,
            properties=properties or {}
        )
        self._nodes[node_id] = node
        self._persist_node(node)
        return node

    def add_edge(self, source: str, target: str, relation: str,
                 properties: Dict[str, Any] = None, weight: float = 1.0) -> Edge:
        """Add a relationship between nodes."""
        edge = Edge(
            source=source,
            target=target,
            relation=relation,
            properties=properties or {},
            weight=weight
        )

        self._edges[source].append(edge)
        self._relations[relation].append(edge)
        self._adjacency[source].add(target)
        self._persist_edge(edge)

        return edge

    def get_node(self, node_id: str) -> Optional[Node]:
        """Get node by ID."""
        return self._nodes.get(node_id)

    def get_neighbors(self, node_id: str,
                       relation: str = None) -> List[Tuple[Node, Edge]]:
        """Get neighboring nodes with their connecting edges."""
        edges = self._edges.get(node_id, [])
        if relation:
            edges = [e for e in edges if e.relation == relation]

        results = []
        for edge in edges:
            target = self._nodes.get(edge.target)
            if target:
                results.append((target, edge))
        return results

    def find_path(self, start: str, end: str,
                  max_depth: int = 5) -> Optional[List[Edge]]:
        """Find path between two nodes using BFS."""
        if start not in self._nodes or end not in self._nodes:
            return None

        visited = {start}
        queue = deque([(start, [])])

        while queue:
            current, path = queue.popleft()

            if current == end and path:
                return path

            if len(path) >= max_depth:
                continue

            for edge in self._edges.get(current, []):
                if edge.target not in visited:
                    visited.add(edge.target)
                    queue.append((edge.target, path + [edge]))

        return None

    def query(self, pattern: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query graph with pattern matching."""
        results = []

        node_type = pattern.get("node_type")
        relation = pattern.get("relation")
        property_filter = pattern.get("properties", {})

        for node_id, node in self._nodes.items():
            if node_type and node.node_type != node_type:
                continue

            match = True
            for key, value in property_filter.items():
                if node.properties.get(key) != value:
                    match = False
                    break

            if not match:
                continue

            edges = self._edges.get(node_id, [])
            if relation:
                edges = [e for e in edges if e.relation == relation]

            for edge in edges:
                target = self._nodes.get(edge.target)
                if target:
                    results.append({
                        "source": node.to_dict(),
                        "edge": edge.to_dict(),
                        "target": target.to_dict()
                    })

        return results

    def infer_relation(self, node_a: str, node_b: str, describe: bool = False) -> Dict[str, Any]:
        """Infer possible relations between two nodes. Direct edges and
        common-neighbor chains are found via exact graph traversal (real
        graph theory, not guesswork). With describe=True, the LLM turns a
        multi-hop chain into a single readable relation label instead of
        the raw 'rel_a_then_rel_b' concatenation."""
        direct = [e.relation for e in self._edges.get(node_a, []) if e.target == node_b]

        neighbors_a = self._adjacency.get(node_a, set())
        neighbors_b = self._adjacency.get(node_b, set())
        common = neighbors_a & neighbors_b

        inferred_chains = []
        for neighbor in common:
            edges_a = [e for e in self._edges.get(node_a, []) if e.target == neighbor]
            edges_b = [e for e in self._edges.get(neighbor, []) if e.target == node_b]

            for ea in edges_a:
                for eb in edges_b:
                    inferred_chains.append(f"{ea.relation}_then_{eb.relation}_via_{neighbor}")

        result = {"direct": direct, "inferred_chains": inferred_chains}

        if describe and (direct or inferred_chains):
            node_a_obj, node_b_obj = self._nodes.get(node_a), self._nodes.get(node_b)
            system_prompt = (
                "You are the knowledge graph reasoning component of DATORYX. Summarize how two "
                "entities relate given the direct and multi-hop connections found between them."
            )
            user_prompt = (
                f"Entity A: {node_a_obj.label if node_a_obj else node_a}\n"
                f"Entity B: {node_b_obj.label if node_b_obj else node_b}\n"
                f"Direct relations found: {direct}\n"
                f"Multi-hop chains found: {inferred_chains}\n\n"
                'Return JSON: {"summary": "<one sentence describing the relationship>"}'
            )
            result["summary"] = llm_json(self.llm, system_prompt, user_prompt).get("summary", "")

        return result

    def get_subgraph(self, center_node: str, depth: int = 2) -> Dict[str, Any]:
        """Extract subgraph around a center node."""
        nodes = set()
        edges = []

        queue = deque([(center_node, 0)])
        nodes.add(center_node)

        while queue:
            current, current_depth = queue.popleft()

            if current_depth >= depth:
                continue

            for edge in self._edges.get(current, []):
                edges.append(edge.to_dict())
                if edge.target not in nodes:
                    nodes.add(edge.target)
                    queue.append((edge.target, current_depth + 1))

        return {
            "nodes": [self._nodes[n].to_dict() for n in nodes if n in self._nodes],
            "edges": edges
        }

    def _persist_node(self, node: Node):
        try:
            with session_scope() as session:
                record = session.get(GraphNodeRecord, (node.id, self.user_id))
                if record is None:
                    record = GraphNodeRecord(id=node.id, user_id=self.user_id)
                record.user_id = self.user_id
                record.label = node.label
                record.node_type = node.node_type
                record.properties = node.properties
                session.merge(record)
        except Exception as e:
            print(f"[KnowledgeGraph] Persist node error: {e}")

    def _persist_edge(self, edge: Edge):
        try:
            with session_scope() as session:
                record = GraphEdgeRecord(
                    user_id=self.user_id,
                    source=edge.source,
                    target=edge.target,
                    relation=edge.relation,
                    weight=edge.weight,
                    properties=edge.properties
                )
                session.add(record)
        except Exception as e:
            print(f"[KnowledgeGraph] Persist edge error: {e}")

    def _load(self):
        try:
            with session_scope() as session:
                for record in session.query(GraphNodeRecord).filter(
                    GraphNodeRecord.user_id == self.user_id
                ).all():
                    node = Node(
                        id=record.id,
                        label=record.label,
                        node_type=record.node_type or "entity",
                        properties=record.properties or {}
                    )
                    self._nodes[node.id] = node

                for record in session.query(GraphEdgeRecord).filter(
                    GraphEdgeRecord.user_id == self.user_id
                ).all():
                    edge = Edge(
                        source=record.source,
                        target=record.target,
                        relation=record.relation,
                        properties=record.properties or {},
                        weight=record.weight if record.weight is not None else 1.0
                    )
                    self._edges[edge.source].append(edge)
                    self._relations[edge.relation].append(edge)
                    self._adjacency[edge.source].add(edge.target)
        except Exception as e:
            print(f"[KnowledgeGraph] Load error: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        return {
            "nodes": len(self._nodes),
            "edges": sum(len(e) for e in self._edges.values()),
            "relations": len(self._relations),
            "avg_degree": sum(len(e) for e in self._edges.values()) / max(len(self._nodes), 1)
        }

    def to_cypher(self) -> List[str]:
        """Generate Cypher-like queries for export."""
        queries = []

        for node in self._nodes.values():
            props = ", ".join([f"{k}: '{v}'" for k, v in node.properties.items()])
            queries.append(f"CREATE (:{node.node_type} {{id: '{node.id}', label: '{node.label}', {props}}})")

        for edge_list in self._edges.values():
            for edge in edge_list:
                queries.append(
                    f"MATCH (a {{id: '{edge.source}'}}), (b {{id: '{edge.target}'}}) "
                    f"CREATE (a)-[:{edge.relation}]->(b)"
                )

        return queries
