"""Typed runtime settings: config/settings.yaml, overridden by CARDAGENT_* environment variables."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

PROJECT_ROOT = Path(os.getenv("CARDAGENT_ROOT", Path(__file__).resolve().parents[2]))
CONFIG_FILE = Path(os.getenv("CARDAGENT_CONFIG", PROJECT_ROOT / "config" / "settings.yaml"))


class EmbedderCfg(BaseModel):
    kind: Literal["classic", "dinov2_onnx"] = "classic"
    onnx_path: str = "models/dinov2/dinov2_vits14.onnx"
    onnx_providers: list[str] = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    input_size: int = 224
    similarity_floor: float = 0.55


class OcrCfg(BaseModel):
    engine: Literal["tesseract", "paddle"] = "tesseract"
    tesseract_lang: str = "eng+ara"
    tesseract_cmd: str = ""          # path to the tesseract binary; empty = PATH, then the standard install dirs
    paddle_det_dir: str = "models/paddleocr/det"
    paddle_rec_dir_en: str = "models/paddleocr/rec_en"
    paddle_rec_dir_ar: str = "models/paddleocr/rec_ar"
    use_gpu: bool = False


class RetrievalCfg(BaseModel):
    shortlist_refs: int = 20
    max_verify_refs: int = 24
    refs_per_variant_verify: int = 2
    orb_features: int = 1000
    ratio_test: float = 0.75
    min_inliers: int = 25
    inliers_baseline: int = 35
    inliers_full_score: int = 120
    max_align_shift: float = 0.12    # skip alignment of a detected card if H moves a corner > this × card width
    max_candidate_refs: int = 500    # refs scored per request across the shortlisted families


class QdrantCfg(BaseModel):
    url: str = "http://127.0.0.1:6333"   # not "localhost": Windows tries IPv6 first (~2 s per call)
    api_key: str = ""                # set via CARDAGENT_VECTOR_STORE__QDRANT__API_KEY
    prefer_grpc: bool = False
    timeout_s: float = 10
    location: str = ""               # ":memory:" = in-process (tests); overrides url
    path: str = ""                   # embedded on-disk mode (single process, dev only); overrides url
    collection_prefix: str = "cardagent"
    reference_store_dir: str = "data/reference_store"   # copies of API-uploaded reference images
    keypoint_cache_size: int = 5000  # ORB keypoint sets cached per process (LRU)
    refresh_seconds: float = 5       # how often a replica checks for card changes made elsewhere
    sync_on_startup: bool = False    # sync the file registry into Qdrant when the service starts


class VectorStoreCfg(BaseModel):
    backend: Literal["faiss", "qdrant"] = "faiss"
    qdrant: QdrantCfg = QdrantCfg()


class ScoringCfg(BaseModel):
    family_weights: dict[str, float] = {"text": 0.40, "embed": 0.20, "geometry": 0.30, "colour": 0.10}
    variant_weights: dict[str, float] = {"colour": 0.40, "keyword": 0.25, "geometry": 0.15,
                                         "embed": 0.10, "prefix": 0.10}
    family_min: float = 0.50
    family_margin: float = 0.12
    variant_min: float = 0.45
    variant_margin: float = 0.08
    reject_below: float = 0.30
    geometry_required_without_text: float = 0.60
    field_min_confidence: float = 0.55


class QualityCfg(BaseModel):
    min_side: int = 400
    min_blur: float = 60.0
    max_glare: float = 0.08
    brightness_min: float = 0.15
    brightness_max: float = 0.92


class VlmCfg(BaseModel):
    enabled: bool = False
    kind: Literal["ollama", "openai"] = "ollama"
    url: str = "http://vlm:11434"
    model: str = "qwen2.5vl:7b"
    timeout_s: float = 30
    circuit_breaker_failures: int = 3
    circuit_breaker_cooldown_s: float = 60


class ApiCfg(BaseModel):
    admin_token: str = ""
    max_image_bytes: int = 8_000_000
    allowed_content_types: list[str] = ["image/jpeg", "image/png", "image/webp"]
    idempotency_cache_size: int = 2000


class LoggingCfg(BaseModel):
    level: str = "INFO"
    json_logs: bool = Field(True, alias="json")
    mask_pii: bool = True


class _YamlSource(PydanticBaseSettingsSource):
    def get_field_value(self, field, field_name):  # pragma: no cover - not used
        return None, field_name, False

    def __call__(self):
        return yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8")) if CONFIG_FILE.exists() else {}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CARDAGENT_", env_nested_delimiter="__", extra="ignore")

    registry_dir: str = "registry"
    index_dir: str = "data/index"
    strict_index_version: bool = False
    embedder: EmbedderCfg = EmbedderCfg()
    vector_store: VectorStoreCfg = VectorStoreCfg()
    ocr: OcrCfg = OcrCfg()
    retrieval: RetrievalCfg = RetrievalCfg()
    scoring: ScoringCfg = ScoringCfg()
    quality: QualityCfg = QualityCfg()
    vlm: VlmCfg = VlmCfg()
    api: ApiCfg = ApiCfg()
    logging: LoggingCfg = LoggingCfg()

    @classmethod
    def settings_customise_sources(cls, settings_cls, init_settings, env_settings, dotenv_settings,
                                   file_secret_settings):
        # precedence: explicit init > env vars > yaml file
        return init_settings, env_settings, _YamlSource(settings_cls)

    def path(self, p: str) -> Path:
        q = Path(p)
        return q if q.is_absolute() else PROJECT_ROOT / q


@lru_cache
def get_settings() -> Settings:
    return Settings()
