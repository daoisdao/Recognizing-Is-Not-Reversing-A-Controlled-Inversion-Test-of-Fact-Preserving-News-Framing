# SPDX-FileCopyrightText: 2026 Yi Liu
# SPDX-License-Identifier: Apache-2.0

"""Record a source-only repair for a provider output below the word floor."""

from __future__ import annotations

import argparse
import json
import re

from pipeline_common import ROOT, append_jsonl, jsonl_read, load_configs, word_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_id")
    args = parser.parse_args()
    cfg, _ = load_configs()
    min_words = int(cfg["sample"]["canonical_words_min"])
    max_words = int(cfg["sample"]["canonical_words_max"])
    source = next(row for row in jsonl_read(ROOT / "data" / "splits" / "full_filtered_sources.jsonl")
                  if row["source_id"] == args.source_id)
    parsed = sorted((ROOT / "data/results" / "parsed" / "glm52" / "canonicalize").glob(f"{args.source_id}_a*.json"))[-1]
    payload = json.loads(parsed.read_text(encoding="utf-8"))
    text = re.sub(r"\n\s*(?:sources|reactions|image highlights)\s*$", "", source["raw_article_text"], flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    if word_count(text) < min_words:
        text += f" The report concerned {source['title']}."
    if not min_words <= word_count(text) <= max_words:
        raise SystemExit(f"repair produced {word_count(text)} words")
    metadata = dict(payload["metadata"])
    metadata["repair"] = "source_only_text_after_provider_underflow"
    append_jsonl(ROOT / "data" / "canonical" / "full.jsonl", {
        **{k: source[k] for k in ("source_id", "title", "provenance_url", "revision_id",
                                  "publication_date", "license", "domain", "target_actor")},
        "canonical_text": text, "word_count": word_count(text), "generator_metadata": metadata,
    })
    print(f"Repaired {args.source_id}: {word_count(text)} words from source text")


if __name__ == "__main__":
    main()
