"""Registry: what is loaded, and import of git-managed card files (registry/cards + registry/refs)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import ValidationError

from ...agent import CardAgent
from ...store.sync import sync_registry
from ..deps import agent, audit, require_admin

router = APIRouter(prefix="/v1/registry", tags=["registry"])


@router.get("")
def registry(a: CardAgent = Depends(agent)):
    """Loaded families, variants and reference counts, plus the registry revision."""
    a.refresh_if_changed(force=True)
    return a.state.summary()


@router.post("/sync", dependencies=[Depends(require_admin)])
async def sync(prune: bool = True, reembed: bool = False, a: CardAgent = Depends(agent)):
    """Incremental import of registry/ into Qdrant (same as tools/sync_registry.py): only new, changed or
    removed files are processed. prune=false keeps cards whose files were removed; reembed=true
    re-embeds every reference (after changing the embedder)."""
    try:
        out = await run_in_threadpool(sync_registry, a.store, a.s.path(a.s.registry_dir), prune, reembed)
    except (ValueError, ValidationError) as e:
        raise HTTPException(422, f"registry invalid, nothing changed: {e}")
    a.refresh_if_changed(force=True)
    audit("registry_sync", **{k: v for k, v in out.items() if isinstance(v, (int, bool))})
    return out


@router.post("/reload", dependencies=[Depends(require_admin)])
async def reload(a: CardAgent = Depends(agent)):
    """Reload cards and runtime settings from Qdrant now (replicas also do this every refresh_seconds)."""
    try:
        return await run_in_threadpool(a.reload)
    except Exception as e:                                        # noqa: BLE001 - previous state is kept
        raise HTTPException(503, f"reload failed, previous version kept: {e}")
