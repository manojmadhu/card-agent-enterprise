"""Typed runtime settings: config/settings.yaml, overridden by CARDAGENT_* environment variables."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator
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
    engine: Literal["paddle", "tesseract"] = "paddle"
    # paddle: PaddleOCR models on ONNX Runtime. Empty model paths = the PP-OCRv6 small models that
    # ship inside the rapidocr package. Paths are relative to the project folder.
    paddle_det_model: str = ""       # text detection .onnx
    paddle_rec_model: str = ""       # Latin-script recognition .onnx (English, French, ...)
    paddle_rec_keys: str = ""        # its character dictionary, only for models without one embedded
    paddle_rec_model_ar: str = "models/paddleocr/arabic_PP-OCRv5_rec_mobile.onnx"   # Arabic; "" = off
    paddle_rec_keys_ar: str = ""
    paddle_min_score: float = Field(0.5, ge=0, le=1)     # drop recognised lines below this confidence
    paddle_box_thresh: float = Field(0.5, ge=0, le=1)    # text detection box score
    paddle_unclip_ratio: float = Field(1.6, gt=0)        # how far detected boxes are grown
    use_gpu: bool = False            # CUDA through onnxruntime-gpu
    # tesseract
    tesseract_lang: str = "eng+ara"
    tesseract_cmd: str = ""          # path to the tesseract binary; empty = PATH, then the standard install dirs


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
    """The only store: card families, reference vectors, runs, settings and evaluations."""
    url: str = "http://127.0.0.1:6333"   # not "localhost": Windows tries IPv6 first (~2 s per call)
    api_key: str = ""                # set via CARDAGENT_QDRANT__API_KEY
    prefer_grpc: bool = False
    timeout_s: float = 10
    location: str = ""               # ":memory:" = in-process (tests); overrides url
    path: str = ""                   # embedded on-disk mode (single process, dev only); overrides url
    collection_prefix: str = "cardagent"
    reference_store_dir: str = "data/reference_store"   # copies of API-uploaded reference images
    keypoint_cache_size: int = 5000  # ORB keypoint sets cached per process (LRU)
    refresh_seconds: float = 5       # how often a replica checks for card changes made elsewhere
    sync_on_startup: bool = False    # sync the file registry into Qdrant when the service starts


class RunsCfg(BaseModel):
    """Run history (every validation) and the human review queue, stored in Qdrant."""
    enabled: bool = True
    store_images: bool = True        # keep the uploaded photo with each run (needed for review + enroll)
    store_dir: str = "data/run_store"
    retention_days: int = 30         # runs (and their photos) older than this are deleted; 0 = keep forever
    stats_max_runs: int = 50000      # cap on runs scanned for one /v1/stats call


class EvaluationCfg(BaseModel):
    """Labelled evaluation sets and evaluation reports."""
    store_dir: str = "data/evaluations"
    max_set_bytes: int = 500_000_000  # largest evaluation-set zip accepted
    min_accept_precision: float = 0.99
    min_status_agreement: float = 0.90


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
    # two photos in one request: each strongly matching (RANSAC geometry score >= this) a DIFFERENT
    # card family means they are not two sides of the same card -> REVIEW with SIDES_MISMATCH
    sides_mismatch_geometry: float = 0.30


SHARPNESS_DEFAULT = {"paddle": 15.0, "tesseract": 80.0}


class QualityCfg(BaseModel):
    min_side: int = 400
    # Sharpness = Laplacian variance of the contrast-stretched image (1st-99th percentile), so a dark or
    # low-contrast photo is not mistaken for a blurred one. None = by OCR engine: paddle 15 (it still
    # reads blur of sigma ~3.5), tesseract 80 (it starts misreading from sigma ~1.5).
    min_sharpness: float | None = Field(None, ge=0)
    min_blur: float = Field(0.0, ge=0)   # legacy raw Laplacian variance (drops with contrast); 0 = off
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
    cors_origins: list[str] = []     # browser origins allowed to call the API (e.g. the web console)

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _origins(cls, v):
        """Accept a JSON list (the usual env form) or a comma-separated string, and normalise each
        origin: a browser sends "http://localhost:5173", never with a path or trailing slash."""
        if isinstance(v, str):
            v = [x for x in (s.strip() for s in v.split(",")) if x]
        out = []
        for o in v or []:
            o = str(o).strip().rstrip("/")
            if o != "*" and not (o.startswith("http://") or o.startswith("https://")):
                raise ValueError(f"CORS origin '{o}' must look like http://host[:port] or https://host[:port]")
            if o != "*" and o.count("/") > 2:
                raise ValueError(f"CORS origin '{o}' must not contain a path: use scheme://host[:port] only")
            out.append(o)
        return out


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

    registry_dir: str = "registry"   # optional git-managed card files, imported with /v1/registry/sync
    embedder: EmbedderCfg = EmbedderCfg()
    qdrant: QdrantCfg = QdrantCfg()
    runs: RunsCfg = RunsCfg()
    evaluation: EvaluationCfg = EvaluationCfg()
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
