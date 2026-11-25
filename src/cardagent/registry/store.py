"""Loads card families + reference image paths. Version hash covers YAML AND reference images,
so any change to either produces a new registry version (stamped on every result).

Reference layout per variant (front images directly in the variant folder, as before sides existed):
  registry/refs/<family>/<variant>/*.png          front
  registry/refs/<family>/<variant>/front/*.png    front (optional explicit folder)
  registry/refs/<family>/<variant>/back/*.png     back  (the family must list the back side)
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

from .models import CardFamily

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass(frozen=True)
class RefImage:
    family_id: str
    variant_id: str
    path: Path
    side: str = "front"


@dataclass
class RegistrySnapshot:
    version: str
    families: dict[str, CardFamily]
    refs: list[RefImage]

    def summary(self) -> dict:
        per_variant: dict[tuple[str, str], int] = {}
        for r in self.refs:
            per_variant[(r.family_id, r.variant_id)] = per_variant.get((r.family_id, r.variant_id), 0) + 1
        return {
            "version": self.version,
            "families": len(self.families),
            "variants": sum(len(f.variants) for f in self.families.values()),
            "reference_images": len(self.refs),
            "items": [{"family": f.id, "country": f.country, "name": f.name,
                       "variants": [{"id": v.id, "name": v.name,
                                     "reference_images": per_variant.get((f.id, v.id), 0)}
                                    for v in f.variants]}
                      for f in sorted(self.families.values(), key=lambda x: x.id)],
        }


def load_registry(root: Path) -> RegistrySnapshot:
    """Raises on any invalid file: callers keep the previous snapshot in that case."""
    digest = hashlib.sha256()
    families: dict[str, CardFamily] = {}
    for path in sorted((root / "cards").rglob("*.yaml")):
        raw = path.read_bytes()
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(raw)
        fam = CardFamily.model_validate(yaml.safe_load(raw))
        if fam.id in families:
            raise ValueError(f"duplicate family id {fam.id} in {path}")
        families[fam.id] = fam

    refs: list[RefImage] = []
    for fam in families.values():
        for v in fam.variants:
            vdir = root / "refs" / fam.id / v.id
            if not vdir.exists():
                continue
            for side, folder in (("front", vdir), ("front", vdir / "front"), ("back", vdir / "back")):
                files = [p for p in sorted(folder.glob("*")) if p.is_file() and p.suffix.lower() in IMAGE_EXT] \
                    if folder.is_dir() else []
                if files and side not in fam.side_ids:
                    raise ValueError(f"{folder}: images of side '{side}', but {fam.id} has sides {fam.side_ids}")
                for p in files:
                    digest.update(p.relative_to(root).as_posix().encode())
                    digest.update(hashlib.sha256(p.read_bytes()).digest())
                    refs.append(RefImage(fam.id, v.id, p, side))
    return RegistrySnapshot(version=digest.hexdigest()[:16], families=families, refs=refs)
