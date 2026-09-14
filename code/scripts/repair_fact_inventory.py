# SPDX-FileCopyrightText: 2026 Yi Liu
# SPDX-License-Identifier: Apache-2.0

"""Record a deterministic fact-count repair for one GLM inventory."""

from __future__ import annotations

import argparse
import json

from pipeline_common import ROOT, append_jsonl, jsonl_read


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_id")
    args = parser.parse_args()
    source = next(row for row in jsonl_read(ROOT / "data" / "canonical" / "full.jsonl")
                  if row["source_id"] == args.source_id)
    parsed = sorted((ROOT / "data/results" / "parsed" / "glm52" / "facts").glob(f"{args.source_id}_a*.json"))[-1]
    payload = json.loads(parsed.read_text(encoding="utf-8"))
    facts = list((payload.get("output") or {}).get("facts") or [])
    if len(facts) < 6:
        raise SystemExit(f"cannot repair {args.source_id}: only {len(facts)} facts")
    facts = facts[:10]
    for index, fact in enumerate(facts, 1):
        fact["fact_id"] = f"F{index:02d}"
    metadata = dict(payload["metadata"])
    metadata["repair"] = "deterministic_first10_fact_count_bound"
    append_jsonl(ROOT / "data" / "facts" / "full.jsonl", {
        "source_id": args.source_id, "target_actor": source["target_actor"],
        "facts": facts, "human_audit_status": "pending_final_audit",
        "generator_metadata": metadata,
    })
    print(f"Repaired {args.source_id}: retained {len(facts)} facts from {parsed.name}")


if __name__ == "__main__":
    main()
