#!/usr/bin/env python3
"""
Few-shot retriever for NEF.

Loads Text2KGBench's pre-computed test→train similarity ranking and the
training split, and returns the top-K most similar (sentence, gold-triples)
exemplars for a given test sentence ID.

The retriever returns exemplars formatted to match NEF's extraction-prompt
JSON schema:
  [{"subject": "...", "predicate": "...", "object": "...",
    "object_type": "entity|literal|number|date", "confidence": 0.95}, ...]

This means the LLM sees in-context exemplars in the *same* format it is
asked to produce — no format-translation burden on the model.

Notes on entity surface forms:
- Gold triples use canonical underscored forms (e.g. "1_Decembrie_1918_University").
- NEF's extraction prompt instructs the LLM to copy entities *verbatim from the
  text* so that downstream Redis grounding can resolve them.
- We therefore replace underscores with spaces in exemplar subjects/objects so
  the in-context examples match the verbatim-from-text style the prompt asks
  for. Predicates are kept as the gold camelCase label since the predicate
  retrieval step embeds either form fine.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


_NUM_RE = re.compile(r"^-?\d{1,3}(?:,\d{3})*(?:\.\d+)?$|^-?\d+(?:\.\d+)?$")
_DATE_RE = re.compile(
    r"^\d{4}(-\d{2}(-\d{2})?)?$"
    r"|^\d{2}/\d{2}/\d{4}$"
    r"|^\d{4}/\d{2}/\d{2}$"
)


def _strip_outer_quotes(s: str) -> str:
    s = (s or "").strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        return s[1:-1].strip()
    return s


def _underscores_to_spaces(s: str) -> str:
    return (s or "").replace("_", " ").strip()


def _classify_object(raw_obj: str, cleaned_obj: str) -> str:
    """Best-effort object_type classification for exemplars."""
    raw = (raw_obj or "").strip()
    if raw.startswith('"') and raw.endswith('"'):
        return "literal"
    if _NUM_RE.match(cleaned_obj.replace(" ", "")):
        return "number"
    if _DATE_RE.match(cleaned_obj.replace(" ", "")):
        return "date"
    return "entity"


def _format_train_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a Text2KGBench train record into a NEF-prompt-ready exemplar.

    Returns a dict with keys "sent" and "triples_json"; "triples_json" is a list
    of dicts in the exact schema the NEF extraction prompt expects.
    """
    sent = (rec.get("sent") or "").strip()
    out_triples: List[Dict[str, Any]] = []
    for t in rec.get("triples", []) or []:
        sub_raw = (t.get("sub") or "").strip()
        rel_raw = (t.get("rel") or "").strip()
        obj_raw = (t.get("obj") or "").strip()
        if not (sub_raw and rel_raw and obj_raw):
            continue
        sub = _underscores_to_spaces(_strip_outer_quotes(sub_raw))
        obj = _underscores_to_spaces(_strip_outer_quotes(obj_raw))
        obj_type = _classify_object(obj_raw, obj)
        out_triples.append({
            "subject": sub,
            "predicate": rel_raw,
            "object": obj,
            "object_type": obj_type,
            "confidence": 0.95,
        })
    return {"sent": sent, "triples_json": out_triples}


class FewShotRetriever:
    """Lazy in-memory retriever over a single ontology's train split.

    Memory footprint is tiny (training splits are O(100s) of records per
    ontology), so we just keep everything in dicts.
    """

    def __init__(
        self,
        train_jsonl_path: Path,
        similars_json_path: Path,
        verbose: bool = True,
    ):
        train_jsonl_path = Path(train_jsonl_path)
        similars_json_path = Path(similars_json_path)
        if not train_jsonl_path.exists():
            raise FileNotFoundError(f"Train file not found: {train_jsonl_path}")
        if not similars_json_path.exists():
            raise FileNotFoundError(f"Similars file not found: {similars_json_path}")

        self._by_id: Dict[str, Dict[str, Any]] = {}
        with open(train_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                tid = rec.get("id")
                if tid:
                    self._by_id[tid] = rec

        with open(similars_json_path, "r", encoding="utf-8") as f:
            self._similars: Dict[str, List[str]] = json.load(f)

        if verbose:
            print(
                f"   📚 FewShotRetriever: {len(self._by_id)} train records, "
                f"{len(self._similars)} test→train similarity rows"
            )

    @classmethod
    def for_ontology(
        cls,
        onto_id: str,
        train_dir: Path,
        similars_dir: Path,
        verbose: bool = True,
    ) -> "FewShotRetriever":
        """Construct a retriever for ontology id like '1_university'."""
        train_path = Path(train_dir) / f"ont_{onto_id}_train.jsonl"
        similars_path = Path(similars_dir) / f"{onto_id}_test_train_similarity.json"
        return cls(train_path, similars_path, verbose=verbose)

    def retrieve(self, test_id: str, k: int = 3) -> List[Dict[str, Any]]:
        """Return up to K NEF-formatted exemplars most similar to test_id.

        Falls back gracefully:
        - If test_id is unknown, returns [].
        - If a similar train_id is missing from the train split, it is skipped.
        - Returns at most K exemplars (may be fewer if fewer are available).
        """
        if k <= 0:
            return []
        train_ids = self._similars.get(test_id) or []
        out: List[Dict[str, Any]] = []
        for tid in train_ids:
            rec = self._by_id.get(tid)
            if not rec:
                continue
            ex = _format_train_record(rec)
            if ex["sent"] and ex["triples_json"]:
                out.append(ex)
            if len(out) >= k:
                break
        return out
