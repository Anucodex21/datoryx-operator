"""DATORYX Session Memory.

The heavier DATORYX memory subsystems (memory/vector needs chromadb,
memory/episodic needs the core.persistence DB layer) aren't wired into
this product build yet - rather than half-wire a dependency chain that
isn't actually running, this is a small, real, SQLite-backed store that
does one job: remember the last N tool calls per (user, agent) so a
multi-turn chat actually has context, instead of every call being
stateless. Swap this for memory/vector + memory/episodic later without
changing the call sites in api/main.py.
"""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

_DB_PATH = Path(__file__).parent / "session_memory.db"


@contextmanager
def _conn():
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                agent_key TEXT NOT NULL,
                tool TEXT NOT NULL,
                query TEXT,
                success INTEGER NOT NULL,
                summary TEXT,
                created_at REAL NOT NULL
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_user_agent ON interactions(user_id, agent_key, created_at)")


def record_interaction(user_id: str, agent_key: str, tool: str, query: str,
                        success: bool, summary: str) -> None:
    init_db()
    with _conn() as c:
        c.execute(
            "INSERT INTO interactions (user_id, agent_key, tool, query, success, summary, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user_id, agent_key, tool, query, int(success), summary[:2000], time.time()),
        )


def recent_context(user_id: str, agent_key: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Last `limit` interactions this user had with this agent, oldest
    first, formatted for dropping straight into an LLM prompt as context.
    """
    init_db()
    with _conn() as c:
        rows = c.execute(
            "SELECT tool, query, success, summary, created_at FROM interactions "
            "WHERE user_id = ? AND agent_key = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, agent_key, limit),
        ).fetchall()
    return [
        {"tool": r["tool"], "query": r["query"], "success": bool(r["success"]), "summary": r["summary"]}
        for r in reversed(rows)
    ]


def format_context_for_prompt(user_id: str, agent_key: str, limit: int = 5) -> str:
    history = recent_context(user_id, agent_key, limit)
    if not history:
        return ""
    lines = ["Recent conversation history with this agent (most recent last):"]
    for h in history:
        status = "ok" if h["success"] else "failed"
        lines.append(f"- [{status}] {h['query']!r} -> {h['summary'][:200]}")
    return "\n".join(lines)
