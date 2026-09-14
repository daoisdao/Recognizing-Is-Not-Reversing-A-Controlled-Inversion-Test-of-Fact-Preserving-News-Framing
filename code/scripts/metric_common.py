# SPDX-FileCopyrightText: 2026 Yi Liu
# SPDX-License-Identifier: Apache-2.0

"""Deterministic, auditable pilot metrics for reconstruction outputs.

The benchmark's production Fact F1/IRR annotations may use a held-out evaluator
for genuinely ambiguous cases. The pilot implementation deliberately stays
local and deterministic: it reports lexical/number-compatible matches and marks
uncertain intervention matches instead of asking a model to judge itself.
"""

from __future__ import annotations

import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

from pipeline_common import ROOT, jsonl_read


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for", "from",
    "had", "has", "have", "in", "into", "is", "it", "its", "of", "on", "or",
    "that", "the", "their", "this", "to", "was", "were", "which", "with", "would",
}


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-").replace("‑", "-")
    return re.sub(r"\s+", " ", text).strip().lower()


def tokens(value: Any, *, keep_stopwords: bool = False) -> list[str]:
    words = re.findall(r"[\w]+", normalize_text(value), flags=re.UNICODE)
    if keep_stopwords:
        return words
    return [word for word in words if word not in STOPWORDS]


def token_set(value: Any) -> set[str]:
    return set(tokens(value))


def sentences(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    pieces = re.split(r"(?<=[.!?。！？])\s+|\n+", text)
    return [piece.strip() for piece in pieces if piece.strip()]


def number_date_strings(fact: dict[str, Any]) -> list[str]:
    return [normalize_text(item) for item in fact.get("numbers_dates", []) if str(item).strip()]


def contains_required_numbers(text: str, fact: dict[str, Any]) -> bool:
    normalized = normalize_text(text)
    return all(item in normalized for item in number_date_strings(fact))


def overlap_score(left: Any, right: Any) -> float:
    a, b = token_set(left), token_set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a)


def entity_overlap(text: str, fact: dict[str, Any]) -> int:
    words = token_set(text)
    return sum(bool(token_set(entity) & words) for entity in fact.get("entities", []))


def fact_match_score(text: str, fact: dict[str, Any]) -> tuple[bool, float, str]:
    """Return (supported, score, reason) for a source-fact/text pair.

    Numeric/date anchors are treated as hard constraints. The remaining test is
    token overlap against the source proposition plus a named-entity anchor.
    This is intentionally conservative and is labelled a lexical proxy in all
    generated outputs.
    """
    proposition = str(fact.get("proposition", ""))
    if not proposition or not contains_required_numbers(text, fact):
        return False, 0.0, "missing_number_or_date"
    score = overlap_score(proposition, text)
    entities = entity_overlap(text, fact)
    threshold = 0.34 if number_date_strings(fact) else 0.42
    supported = score >= threshold and (entities > 0 or score >= threshold + 0.16)
    reason = "token_and_entity_overlap" if supported else "insufficient_overlap"
    return supported, score, reason


def evaluate_fact_f1(reconstruction: str, facts: list[dict[str, Any]]) -> dict[str, Any]:
    supported_ids: list[str] = []
    fact_scores: dict[str, float] = {}
    fact_reasons: dict[str, str] = {}
    for fact in facts:
        supported, score, reason = fact_match_score(reconstruction, fact)
        fact_scores[str(fact.get("fact_id"))] = round(score, 6)
        fact_reasons[str(fact.get("fact_id"))] = reason
        if supported:
            supported_ids.append(str(fact.get("fact_id")))

    recon_sentences = sentences(reconstruction)
    supported_sentence_indices: list[int] = []
    unsupported_sentences: list[str] = []
    sentence_matches: dict[str, list[str]] = {}
    for index, sentence in enumerate(recon_sentences):
        matches: list[tuple[float, str]] = []
        for fact in facts:
            if not contains_required_numbers(sentence, fact):
                continue
            score = overlap_score(fact.get("proposition", ""), sentence)
            entities = entity_overlap(sentence, fact)
            threshold = 0.30 if number_date_strings(fact) else 0.38
            if score >= threshold and (entities > 0 or score >= threshold + 0.16):
                matches.append((score, str(fact.get("fact_id"))))
        matches.sort(reverse=True)
        ids = [fact_id for _, fact_id in matches]
        sentence_matches[str(index)] = ids[:3]
        if ids:
            supported_sentence_indices.append(index)
        else:
            unsupported_sentences.append(sentence)

    recall = len(supported_ids) / len(facts) if facts else 0.0
    precision = len(supported_sentence_indices) / len(recon_sentences) if recon_sentences else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "metric_method": "deterministic_lexical_number_proxy",
        "fact_precision": round(precision, 6),
        "fact_recall": round(recall, 6),
        "fact_f1": round(f1, 6),
        "source_fact_count": len(facts),
        "supported_fact_count": len(supported_ids),
        "supported_fact_ids": supported_ids,
        "missing_fact_ids": [str(f.get("fact_id")) for f in facts if str(f.get("fact_id")) not in supported_ids],
        "reconstruction_sentence_count": len(recon_sentences),
        "supported_sentence_count": len(supported_sentence_indices),
        "unsupported_sentences": unsupported_sentences[:20],
        "sentence_matches": sentence_matches,
        "fact_scores": fact_scores,
        "fact_match_reasons": fact_reasons,
        "ambiguous_cases": [],
    }


def phrase_present(article: str, phrase: str) -> tuple[bool, str, float]:
    target = normalize_text(phrase)
    if not target:
        return False, "empty", 0.0
    text = normalize_text(article)
    if target in text:
        return True, "exact_normalized", 1.0
    target_tokens = tokens(target, keep_stopwords=True)
    text_tokens = tokens(text, keep_stopwords=True)
    if not target_tokens or not text_tokens:
        return False, "none", 0.0
    best = 0.0
    lo = max(1, len(target_tokens) - 3)
    hi = min(len(text_tokens), len(target_tokens) + 3)
    for size in range(lo, hi + 1):
        for start in range(0, len(text_tokens) - size + 1):
            score = SequenceMatcher(None, target_tokens, text_tokens[start:start + size]).ratio()
            best = max(best, score)
    if best >= 0.88:
        return True, "fuzzy_token_window", round(best, 6)
    return False, "none", round(best, 6)


def evaluate_irr(reconstruction: str, edit_plan: list[dict[str, Any]]) -> dict[str, Any]:
    edits: list[dict[str, Any]] = []
    for edit in edit_plan or []:
        before, after = str(edit.get("before", "")), str(edit.get("after", ""))
        before_match, before_method, before_score = phrase_present(reconstruction, before)
        after_match, after_method, after_score = phrase_present(reconstruction, after)
        if before_match and not after_match:
            status = "reversed"
        elif after_match and not before_match:
            status = "retained"
        elif before_match and after_match:
            status = "both_present_ambiguous"
        else:
            status = "neither_present_ambiguous"
        edits.append({
            "edit_id": edit.get("edit_id"), "operator": edit.get("operator"),
            "affected_fact_ids": edit.get("affected_fact_ids", []),
            "status": status, "before_match": before_match, "after_match": after_match,
            "before_match_method": before_method, "after_match_method": after_method,
            "before_score": before_score, "after_score": after_score,
        })
    counts = {key: sum(row["status"] == key for row in edits) for key in (
        "reversed", "retained", "both_present_ambiguous", "neither_present_ambiguous")}
    injected = len(edits)
    decisive = counts["reversed"] + counts["retained"]
    return {
        "metric_method": "deterministic_normalized_phrase_and_fuzzy_window",
        "injected_edit_count": injected,
        "reversed_count": counts["reversed"], "retained_count": counts["retained"],
        "ambiguous_count": counts["both_present_ambiguous"] + counts["neither_present_ambiguous"],
        "irr": round(counts["reversed"] / injected, 6) if injected else None,
        "irr_decisive": round(counts["reversed"] / decisive, 6) if decisive else None,
        "decisive_edit_count": decisive, "edit_results": edits,
    }


def load_fact_map(mode: str) -> dict[str, list[dict[str, Any]]]:
    return {row["source_id"]: row.get("facts", []) for row in jsonl_read(ROOT / "data" / "facts" / f"{mode}.jsonl")}


def load_reconstruction_rows(family: str, mode: str, reasoning_mode: str = "direct") -> list[dict[str, Any]]:
    path = ROOT / "data/results" / "evaluated" / family / f"reconstruction_{mode}_{reasoning_mode}.jsonl"
    rows = [row for row in jsonl_read(path) if row.get("valid", True)]
    if not rows:
        raise FileNotFoundError(path)
    return rows


def write_jsonl_rows(path: Path, rows: Iterable[dict[str, Any]], *, resume: bool = False,
                     key: str = "item_id") -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not resume:
        raise SystemExit(f"Refusing to overwrite {path}; use --resume")
    existing = {str(row.get(key)) for row in jsonl_read(path)} if path.exists() else set()
    count = len(existing)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            if str(row.get(key)) in existing:
                continue
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            existing.add(str(row.get(key)))
            count += 1
    return count
