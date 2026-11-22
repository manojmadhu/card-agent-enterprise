"""Test defaults: hermetic runs need no Qdrant server.

The pipeline/e2e tests run on the FAISS backend unless CARDAGENT_TEST_BACKEND=qdrant (then they use
the server from config/settings.yaml). Qdrant-specific tests (test_qdrant_store.py) always run
against an in-process Qdrant (location=":memory:")."""
import os

os.environ.setdefault("CARDAGENT_VECTOR_STORE__BACKEND", os.getenv("CARDAGENT_TEST_BACKEND", "faiss"))
