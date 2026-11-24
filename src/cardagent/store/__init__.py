"""Qdrant storage for the platform (the only store).

  qdrant_store.QdrantCardStore   card families + reference vectors and keypoints   (<prefix>_families, _refs)
  runtime_settings               threshold overrides changed through the API        (meta point in _families)
  runs.RunStore                  every validation + the human review queue           (<prefix>_runs)
  evaluations.EvaluationStore    labelled evaluation sets + evaluation reports       (<prefix>_evaluations)
  sync.sync_registry             optional import of git-managed card files (registry/)
All of them share one Qdrant client; image files live on disk next to it (reference_store_dir,
runs.store_dir, evaluation.store_dir).
"""
from .base import RefHit, StoreSnapshot, StoreUnavailable  # noqa: F401
