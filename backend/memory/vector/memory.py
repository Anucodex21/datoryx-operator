"""DATORYX Vector Memory - Embedding-based similarity search.

Backed by ChromaDB with on-disk persistence (previously this was a pure
in-RAM numpy dict, so every embedding was lost on restart). Storage
location is controlled by DATORYX_CHROMA_PATH (default ./data/chroma).
The public API (store/search/delete/get_stats/...) is unchanged so
callers elsewhere in the codebase don't need to change.
"""
import os
import json
import numpy as np
import chromadb
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

from shared.types import MemoryEntry, MemoryType

CHROMA_PATH = os.environ.get("DATORYX_CHROMA_PATH", os.path.join("data", "chroma"))
COLLECTION_NAME = "datoryx_vector_memory"


def _safe_collection_suffix(user_id: str) -> str:
    """Chroma collection names are restricted to [a-zA-Z0-9._-]; sanitize a
    user_id so it can be appended safely."""
    import re
    return re.sub(r"[^a-zA-Z0-9._-]", "_", user_id or "public")[:48]


class VectorMemory:
    """ChromaDB-backed vector store with cosine similarity search."""

    def __init__(self, dimension: int = 128, max_entries: int = 10000, user_id: str = "public"):
        self._dimension = dimension
        self._max_entries = max_entries
        self.user_id = user_id or "public"
        os.makedirs(CHROMA_PATH, exist_ok=True)
        self._client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection_name = f"{COLLECTION_NAME}_{_safe_collection_suffix(self.user_id)}"
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        # Lightweight in-memory cache mirroring what's in Chroma, so the rest
        # of the codebase (which expects MemoryEntry objects, not raw Chroma
        # rows) keeps working unchanged.
        self._entries: Dict[str, MemoryEntry] = {}
        self._index: Dict[str, List[str]] = defaultdict(list)  # tag -> entry_ids
        self._load()

    def store(self, content: Any, embedding: List[float],
              metadata: Dict[str, Any] = None, tags: List[str] = None) -> str:
        """Store item with vector embedding."""
        entry = MemoryEntry(
            content=content,
            memory_type=MemoryType.VECTOR,
            embedding=embedding,
            metadata=metadata or {}
        )

        if len(embedding) != self._dimension:
            raise ValueError(f"Expected dimension {self._dimension}, got {len(embedding)}")

        if len(self._entries) >= self._max_entries:
            self._evict_oldest()

        tags = tags or []
        self._collection.upsert(
            ids=[entry.id],
            embeddings=[list(map(float, embedding))],
            documents=[str(content)[:5000]],
            metadatas=[{
                "metadata_json": json.dumps(metadata or {}),
                "tags_csv": ",".join(tags),
                "created_at": entry.created_at.isoformat(),
                "importance_score": entry.importance_score,
            }]
        )

        self._entries[entry.id] = entry
        for tag in tags:
            self._index[tag].append(entry.id)

        return entry.id

    def search(self, query_embedding: List[float], top_k: int = 5,
               tags: List[str] = None) -> List[Tuple[MemoryEntry, float]]:
        """Search by vector similarity using ChromaDB's cosine index."""
        if self._collection.count() == 0:
            return []

        fetch_n = min(self._collection.count(), max(top_k * 4, top_k) if tags else top_k)
        result = self._collection.query(
            query_embeddings=[list(map(float, query_embedding))],
            n_results=fetch_n,
            include=["distances", "metadatas", "documents"]
        )

        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]

        candidate_tag_ids = None
        if tags:
            candidate_tag_ids = set()
            for tag in tags:
                candidate_tag_ids.update(self._index.get(tag, []))

        results = []
        for entry_id, distance in zip(ids, distances):
            if candidate_tag_ids is not None and entry_id not in candidate_tag_ids:
                continue
            entry = self._entries.get(entry_id)
            if entry is None:
                continue
            entry.access_count += 1
            # cosine distance -> similarity
            similarity = 1.0 - float(distance)
            results.append((entry, similarity))
            if len(results) >= top_k:
                break

        return results

    def search_by_content(self, content: str, embed_func: callable,
                           top_k: int = 5) -> List[Tuple[MemoryEntry, float]]:
        """Search by generating embedding from content."""
        embedding = embed_func(content)
        return self.search(embedding, top_k)

    def get_cluster_centers(self, n_clusters: int = 5) -> List[np.ndarray]:
        """Simple k-means clustering to find cluster centers."""
        if self._collection.count() < n_clusters:
            return []

        raw = self._collection.get(include=["embeddings"])
        vectors = np.array(raw["embeddings"])
        if len(vectors) < n_clusters:
            return []

        centers = [vectors[np.random.randint(len(vectors))]]
        for _ in range(1, n_clusters):
            dists = np.array([
                min(np.linalg.norm(v - c) for c in centers)
                for v in vectors
            ])
            total = dists.sum()
            probs = dists / total if total > 0 else np.ones(len(dists)) / len(dists)
            idx = np.random.choice(len(vectors), p=probs)
            centers.append(vectors[idx])

        for _ in range(10):
            assignments = [
                min(range(len(centers)), key=lambda i: np.linalg.norm(v - centers[i]))
                for v in vectors
            ]

            new_centers = []
            for i in range(n_clusters):
                cluster_vectors = vectors[np.array(assignments) == i]
                if len(cluster_vectors) > 0:
                    new_centers.append(cluster_vectors.mean(axis=0))
                else:
                    new_centers.append(centers[i])
            centers = new_centers

        return centers

    def delete(self, entry_id: str) -> bool:
        """Delete entry by ID."""
        if entry_id in self._entries:
            self._collection.delete(ids=[entry_id])
            del self._entries[entry_id]

            for tag, ids in self._index.items():
                if entry_id in ids:
                    ids.remove(entry_id)
            return True
        return False

    def _evict_oldest(self):
        """Evict least accessed entry."""
        if not self._entries:
            return

        oldest = min(self._entries.values(), key=lambda e: e.access_count)
        self.delete(oldest.id)

    def _load(self):
        """Rebuild the in-memory MemoryEntry cache + tag index from Chroma."""
        try:
            if self._collection.count() == 0:
                return
            raw = self._collection.get(include=["metadatas", "documents", "embeddings"])
            for i, entry_id in enumerate(raw["ids"]):
                meta = raw["metadatas"][i] or {}
                stored_metadata = {}
                try:
                    stored_metadata = json.loads(meta.get("metadata_json", "{}"))
                except (TypeError, ValueError):
                    pass
                entry = MemoryEntry(
                    id=entry_id,
                    content=raw["documents"][i],
                    memory_type=MemoryType.VECTOR,
                    embedding=list(raw["embeddings"][i]) if raw.get("embeddings") is not None else None,
                    metadata=stored_metadata,
                    importance_score=meta.get("importance_score", 0.5)
                )
                self._entries[entry_id] = entry
                tags_csv = meta.get("tags_csv", "")
                for tag in [t for t in tags_csv.split(",") if t]:
                    self._index[tag].append(entry_id)
        except Exception as e:
            print(f"[VectorMemory] Load error: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get vector memory statistics."""
        return {
            "entries": len(self._entries),
            "dimension": self._dimension,
            "max_entries": self._max_entries,
            "utilization": len(self._entries) / self._max_entries,
            "tags": len(self._index)
        }
