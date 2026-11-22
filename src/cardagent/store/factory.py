from __future__ import annotations

from ..settings import PROJECT_ROOT, Settings


def build_store(settings: Settings, embedder):
    vs = settings.vector_store
    if vs.backend == "qdrant":
        from .qdrant_store import QdrantCardStore
        return QdrantCardStore(vs.qdrant, embedder, settings.retrieval.orb_features, PROJECT_ROOT)
    from .faiss_store import FaissCardStore
    return FaissCardStore(settings, embedder)
