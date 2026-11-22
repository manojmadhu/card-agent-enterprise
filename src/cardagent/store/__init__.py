"""Card store: the reference-image vector index plus card-family definitions.

Two backends behind one interface (see base.py), chosen by vector_store.backend:
  * faiss  - file-based index rebuilt from the on-disk registry (dev / single process)
  * qdrant - Qdrant collections; incremental add/remove, shared by all API replicas
Use store.factory.build_store() to construct one.
"""
from .base import AdminUnsupported, CardStore, RefHit, StoreSnapshot, StoreUnavailable  # noqa: F401
