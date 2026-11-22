"""OCR keyword inverted index (keyword -> family ids), built from the family definitions."""
from __future__ import annotations

import re
from difflib import SequenceMatcher, get_close_matches

from ..registry.models import CardFamily

_CONFUSABLE = str.maketrans({"O": "0", "Q": "Q", "I": "1", "L": "1", "S": "5", "B": "8"})


def skeleton(s: str) -> str:
    """OCR-confusable 'skeleton' (O->0, I/L->1, S->5, B->8) so 'DEMOQAO1' == 'DEMOQA01'."""
    return s.upper().translate(_CONFUSABLE)


def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


class KeywordIndex:
    def __init__(self, families: dict[str, CardFamily]):
        self.keyword_map: dict[str, set[str]] = {}
        self.negative_map: dict[str, set[str]] = {}
        for f in families.values():
            for k in f.keywords:
                self.keyword_map.setdefault(k, set()).add(f.id)
            for k in f.negative_keywords:
                self.negative_map.setdefault(k, set()).add(f.id)
        self._sk_map: dict[str, set[str]] = {}
        for kw, fams in self.keyword_map.items():
            self._sk_map.setdefault(skeleton(kw), set()).update(fams)

    def keyword_families(self, tokens: list[str], text: str) -> dict[str, float]:
        """family_id -> best keyword score (1.0 exact, ~0.8-0.95 fuzzy)."""
        hits: dict[str, float] = {}
        sk_text = skeleton(text)
        for kw, fams in self._sk_map.items():
            if re.search(rf"(?<!\w){re.escape(kw)}(?!\w)", sk_text):
                for f in fams:
                    hits[f] = 1.0
        vocab = list(self._sk_map)
        for tok in {skeleton(t) for t in tokens if len(t) >= 4}:
            if tok in self._sk_map:      # an exact keyword explains this word: no fuzzy credit to look-alikes
                continue
            for kw in get_close_matches(tok, vocab, n=3, cutoff=0.8):
                score = _ratio(tok, kw)
                for f in self._sk_map[kw]:
                    hits[f] = max(hits.get(f, 0.0), score)
        return hits

    def negative_hits(self, text: str) -> set[str]:
        out: set[str] = set()
        for kw, fams in self.negative_map.items():
            if re.search(rf"(?<!\w){re.escape(kw)}(?!\w)", text):
                out |= fams
        return out
