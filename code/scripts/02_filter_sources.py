# SPDX-FileCopyrightText: 2026 Yi Liu
# SPDX-License-Identifier: Apache-2.0

import argparse
import re

from pipeline_common import (ROOT, add_mode_args, expected_source_count, jsonl_read,
                             jsonl_write_new, load_configs, selected_mode,
                             validate_source_schema, word_count, write_manifest)


EXCLUDED = re.compile(r"\b(editorial|opinion|satire|liveblog|sports score|gossip)\b", re.I)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply frozen source inclusion/exclusion checks.")
    add_mode_args(parser)
    args = parser.parse_args()
    mode = selected_mode(args)
    cfg, _ = load_configs()
    source_path = ROOT / "data" / "raw" / f"{mode}_sources.jsonl"
    accepted, rejected = [], []
    for row in jsonl_read(source_path):
        reasons = validate_source_schema(row)
        text = row.get("raw_article_text", "")
        if word_count(text) < 180:
            reasons.append("too_short_for_six_factual_propositions")
        if EXCLUDED.search(row.get("source_type", "article")):
            reasons.append("excluded_source_type")
        if not row.get("target_actor"):
            reasons.append("no_identifiable_target_actor")
        decision = {**row, "filter": {"pass": not reasons, "reasons": reasons}}
        (accepted if not reasons else rejected).append(decision)
    expected = expected_source_count(mode, cfg)
    if len(accepted) != expected:
        raise SystemExit(f"Filtering retained {len(accepted)}/{expected}; rejected={[(r['source_id'], r['filter']['reasons']) for r in rejected]}")
    output = ROOT / "data" / "splits" / f"{mode}_filtered_sources.jsonl"
    jsonl_write_new(output, accepted, resume=args.resume)
    manifest = write_manifest("02_filter_sources", mode, {
        "input": str(source_path.relative_to(ROOT)), "output": str(output.relative_to(ROOT)),
        "accepted": len(accepted), "rejected": [{"source_id": r["source_id"], **r["filter"]} for r in rejected],
    })
    print(f"Accepted {len(accepted)} sources -> {output}; manifest={manifest}")


if __name__ == "__main__":
    main()
