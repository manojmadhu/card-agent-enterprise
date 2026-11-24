"""Shared FastAPI dependencies: process state, the agent, admin authentication, uploads, body parsing."""
from __future__ import annotations

import hmac
import json
import logging

import yaml
from fastapi import Header, HTTPException, Request, UploadFile
from pydantic import ValidationError

from ..agent import CardAgent
from ..settings import get_settings

STATE: dict = {}
audit_log = logging.getLogger("cardagent.audit")


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


def audit(action: str, **kw):
    audit_log.info("admin %s", action, extra={"extra_json": {"event": "admin", "action": action, **kw}})


async def read_image(upload: UploadFile) -> bytes:
    """Content-type and size checks shared by every endpoint that takes an image."""
    s = get_settings()
    if upload.content_type not in s.api.allowed_content_types:
        raise HTTPException(415, f"{upload.filename or 'image'}: unsupported content type {upload.content_type} "
                                 f"(allowed: {', '.join(s.api.allowed_content_types)})")
    data = await upload.read(s.api.max_image_bytes + 1)
    if len(data) > s.api.max_image_bytes:
        raise HTTPException(413, f"{upload.filename or 'image'}: larger than {s.api.max_image_bytes} bytes")
    return data


async def parse_body(request: Request, model):
    """JSON (default) or YAML (Content-Type: application/yaml) body validated against a pydantic model."""
    raw = await request.body()
    ctype = request.headers.get("content-type", "")
    is_yaml = "yaml" in ctype or "yml" in ctype
    try:
        data = yaml.safe_load(raw) if is_yaml else json.loads(raw or b"null")
    except (json.JSONDecodeError, yaml.YAMLError) as e:
        raise HTTPException(400, f"body is not valid {'YAML' if is_yaml else 'JSON'}: {e}")
    try:
        return model.model_validate(data)
    except ValidationError as e:
        raise HTTPException(422, json.loads(e.json(include_url=False)))


def image_media_type(head: bytes) -> str:
    if head[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    return "application/octet-stream"
