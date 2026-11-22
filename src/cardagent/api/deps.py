"""Shared FastAPI dependencies: process state, the agent, admin authentication."""
from __future__ import annotations

import hmac

from fastapi import Header, HTTPException

from ..agent import CardAgent
from ..settings import get_settings

STATE: dict = {}


def agent() -> CardAgent:
    a = STATE.get("agent")
    if a is None:
        raise HTTPException(503, "not ready")
    return a


def require_admin(x_admin_token: str = Header(default="")):
    token = get_settings().api.admin_token
    if not token:
        raise HTTPException(403, "admin endpoints disabled (set CARDAGENT_API__ADMIN_TOKEN)")
    if not hmac.compare_digest(x_admin_token, token):
        raise HTTPException(401, "invalid admin token")
