"""Lightweight multi-user JWT auth for the DATORYX Operator API.

Deliberately simple (JSON file user store, no external DB dependency) so
the product runs with zero setup - swap `_load_users`/`_save_users` for a
real DB-backed store later without touching the route layer.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Optional

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, EmailStr, field_validator

SECRET_KEY = os.environ.get("DATORYX_JWT_SECRET", "dev-secret-change-me-in-production")
ALGORITHM = "HS256"
TOKEN_TTL_SECONDS = int(os.environ.get("DATORYX_JWT_TTL", 60 * 60 * 24 * 7))  # 7 days

_USERS_PATH = Path(__file__).parent / "users.json"
_bearer = HTTPBearer(auto_error=False)


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    name: str = ""

    @field_validator("password")
    @classmethod
    def _password_len(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("password must be at least 8 characters")
        return v


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = TOKEN_TTL_SECONDS


class CurrentUser(BaseModel):
    id: str
    email: str
    name: str = ""


def _load_users() -> dict:
    if not _USERS_PATH.exists():
        return {}
    return json.loads(_USERS_PATH.read_text())


def _save_users(users: dict) -> None:
    _USERS_PATH.write_text(json.dumps(users, indent=2))


def register_user(req: RegisterRequest) -> TokenResponse:
    users = _load_users()
    if req.email in users:
        raise HTTPException(status.HTTP_409_CONFLICT, "An account with this email already exists")
    user_id = str(uuid.uuid4())
    hashed = bcrypt.hashpw(req.password.encode(), bcrypt.gensalt()).decode()
    users[req.email] = {"id": user_id, "email": req.email, "name": req.name, "password_hash": hashed}
    _save_users(users)
    return _issue_token(user_id, req.email, req.name)


def login_user(req: LoginRequest) -> TokenResponse:
    users = _load_users()
    record = users.get(req.email)
    if not record or not bcrypt.checkpw(req.password.encode(), record["password_hash"].encode()):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid email or password")
    return _issue_token(record["id"], record["email"], record.get("name", ""))


def _issue_token(user_id: str, email: str, name: str) -> TokenResponse:
    now = int(time.time())
    payload = {"sub": user_id, "email": email, "name": name, "iat": now, "exp": now + TOKEN_TTL_SECONDS}
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    return TokenResponse(access_token=token, expires_in=TOKEN_TTL_SECONDS)


def get_current_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> CurrentUser:
    if creds is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing bearer token")
    try:
        payload = jwt.decode(creds.credentials, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token expired, please log in again")
    except jwt.InvalidTokenError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token")
    return CurrentUser(id=payload["sub"], email=payload["email"], name=payload.get("name", ""))
