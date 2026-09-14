# SPDX-FileCopyrightText: 2026 Yi Liu
# SPDX-License-Identifier: Apache-2.0

"""Record a deterministic sentence-boundary repair for one provider overrun."""

from __future__ import annotations

import argparse
import json

from pipeline_common import ROOT, append_jsonl, jsonl_read, load_configs, word_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_id")
    args = parser.parse_args()
    cfg, _ = load_configs()
    max_words = int(cfg["sample"]["canonical_words_max"])
    min_words = int(cfg["sample"]["canonical_words_min"])
    source = next(row for row in jsonl_read(ROOT / "data" / "splits" / "full_filtered_sources.jsonl")
                  if row["source_id"] == args.source_id)
    parsed = sorted((ROOT / "data/results" / "parsed" / "glm52" / "canonicalize").glob(f"{args.source_id}_a*.json"))[-1]
    payload = json.loads(parsed.read_text(encoding="utf-8"))
    value = payload["output"]
    text = str(value["canonical_text"]).strip()
    parts = text.split(". ")
    while word_count(text) > max_words and len(parts) > 1:
        parts.pop()
        text = ". ".join(parts).rstrip(". ") + "."
    count = word_count(text)
    if not min_words <= count <= max_words:
        raise SystemExit(f"repair produced {count} words")
    metadata = dict(payload["metadata"])
    metadata["repair"] = "sentence_boundary_trim_after_provider_overrun"
    append_jsonl(ROOT / "data" / "canonical" / "full.jsonl", {
        **{k: source[k] for k in ("source_id", "title", "provenance_url", "revision_id",
                                  "publication_date", "license", "domain", "target_actor")},
        "canonical_text": text, "word_count": count, "generator_metadata": metadata,
    })
    print(f"Repaired {args.source_id}: {count} words from {parsed.name}")


if __name__ == "__main__":
    main()
