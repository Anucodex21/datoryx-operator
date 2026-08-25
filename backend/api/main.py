"""DATORYX Operator - Product API.

This is the browser-facing product layer on top of the MCP server: same
AgentRegistry, same 20 agents / 120 capabilities, but reachable over plain
HTTP with multi-user JWT auth, so it works without an MCP client - just a
browser hitting the bundled dashboard, or any HTTP client.

Run:
    cd backend
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agents.registry import AgentRegistry
from api.auth import (
    CurrentUser,
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    get_current_user,
    login_user,
    register_user,
)

app = FastAPI(
    title="DATORYX Operator",
    description="Multi-agent AI operator: 20 LLM-backed agents, 120 capabilities, one API.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("DATORYX_CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

from api.extended_features import router as extended_router  # noqa: E402
app.include_router(extended_router)

_registry: Optional[AgentRegistry] = None


def get_registry() -> AgentRegistry:
    global _registry
    if _registry is None:
        _registry = AgentRegistry()
    return _registry


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

@app.post("/auth/register", response_model=TokenResponse)
def register(req: RegisterRequest):
    return register_user(req)


@app.post("/auth/login", response_model=TokenResponse)
def login(req: LoginRequest):
    return login_user(req)


@app.get("/auth/me", response_model=CurrentUser)
def me(user: CurrentUser = Depends(get_current_user)):
    return user


# ---------------------------------------------------------------------------
# Agents / capabilities
# ---------------------------------------------------------------------------

class CapabilityOut(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]


class AgentOut(BaseModel):
    key: str
    name: str
    description: str
    capabilities: List[CapabilityOut]


@app.get("/agents", response_model=List[AgentOut])
def list_agents(user: CurrentUser = Depends(get_current_user)):
    registry = get_registry()
    out = []
    for key, agent in registry.all().items():
        out.append(AgentOut(
            key=key,
            name=agent.name,
            description=agent.description,
            capabilities=[
                CapabilityOut(name=c.name, description=c.description, parameters=c.parameters)
                for c in agent.get_capabilities()
            ],
        ))
    return out


class ExecuteRequest(BaseModel):
    tool: str = "default"
    params: Dict[str, Any] = {}


class ExecuteResponse(BaseModel):
    success: bool
    data: Any = None
    message: str = ""
    execution_time: float = 0.0
    agent: str


@app.post("/agents/{agent_key}/execute", response_model=ExecuteResponse)
def execute_agent(agent_key: str, req: ExecuteRequest, user: CurrentUser = Depends(get_current_user)):
    registry = get_registry()
    try:
        agent = registry.get(agent_key)
    except KeyError:
        raise HTTPException(404, f"No agent named '{agent_key}'. See GET /agents.")

    result = agent.execute({"tool": req.tool, "params": req.params})
    return ExecuteResponse(
        success=result.success,
        data=result.data,
        message=result.message,
        execution_time=result.execution_time,
        agent=agent.name,
    )


class ChatRequest(BaseModel):
    agent: str
    query: str
    context: Optional[str] = None


@app.post("/chat", response_model=ExecuteResponse)
def chat(req: ChatRequest, user: CurrentUser = Depends(get_current_user)):
    """Free-form: ask any agent a natural-language question without
    needing to know its structured capability names. Remembers the last
    few turns this user had with this specific agent (memory/session_store)
    so multi-turn conversations aren't stateless.
    """
    registry = get_registry()
    try:
        agent = registry.get(req.agent)
    except KeyError:
        raise HTTPException(404, f"No agent named '{req.agent}'. See GET /agents.")

    from memory.session_store import format_context_for_prompt, record_interaction

    history_block = format_context_for_prompt(user.id, req.agent)
    context_note = req.context or ""
    if history_block:
        context_note = f"{history_block}\n\n{context_note}".strip()

    result = agent.execute({
        "tool": "default",
        "params": {"query": req.query, "context": {"note": context_note} if context_note else {}},
    })

    record_interaction(
        user_id=user.id, agent_key=req.agent, tool="chat",
        query=req.query, success=result.success,
        summary=str(result.data)[:500] if result.success else result.message,
    )

    return ExecuteResponse(
        success=result.success,
        data=result.data,
        message=result.message,
        execution_time=result.execution_time,
        agent=agent.name,
    )


@app.get("/health")
def health():
    return {"status": "ok", "time": time.time()}


# ---------------------------------------------------------------------------
# Static dashboard (single-page, dark instrument-panel UI)
# ---------------------------------------------------------------------------

_FRONTEND_DIR = _BACKEND_ROOT.parent / "frontend"
if _FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="static")

    @app.get("/")
    def dashboard():
        return FileResponse(str(_FRONTEND_DIR / "index.html"))
