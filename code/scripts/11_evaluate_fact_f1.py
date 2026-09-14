# SPDX-FileCopyrightText: 2026 Yi Liu
# SPDX-License-Identifier: Apache-2.0

"""Compute deterministic Atomic Fact F1 for reconstruction outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from metric_common import evaluate_fact_f1, load_fact_map, load_reconstruction_rows, write_jsonl_rows
from pipeline_common import ROOT, config_hash, utc_now, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate reconstruction Fact F1.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pilot", action="store_true")
    group.add_argument("--full", action="store_true")
    parser.add_argument("--family", choices=("qwen", "deepseek", "kimi"), required=True)
    parser.add_argument("--reasoning-mode", choices=("direct", "thinking"), default="direct")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    mode = "pilot" if args.pilot else "full"
    facts = load_fact_map(mode)
    rows = load_reconstruction_rows(args.family, mode, args.reasoning_mode)
    suffix = "" if args.reasoning_mode == "direct" else f"_{args.reasoning_mode}"
    output = ROOT / "data/results" / "metrics" / args.family / f"fact_f1_{mode}{suffix}.jsonl"
    evaluated = []
    missing_sources: list[str] = []
    for row in rows:
        source_id = str(row.get("source_id"))
        source_facts = facts.get(source_id, [])
        if not source_facts:
            missing_sources.append(source_id)
            continue
        metric = evaluate_fact_f1(str(row.get("reconstructed_article", "")), source_facts)
        evaluated.append({
            "item_id": row.get("item_id") or row.get("sample_id"),
            "sample_id": row.get("sample_id"), "source_id": source_id,
            "operator": row.get("operator"), "strength": row.get("strength"),
            "direction": row.get("direction"), "family": args.family,
            "reconstruction_model_metadata": row.get("model_metadata", {}),
            "metric": metric, "valid": True,
        })
    count = write_jsonl_rows(output, evaluated, resume=args.resume)
    manifest = write_manifest(
        f"fact_f1_{args.family}_{args.reasoning_mode}", mode,
        {"family": args.family, "output": str(output.relative_to(ROOT)),
         "expected_reconstruction_rows": len(rows), "evaluated_rows": len(evaluated),
         "written_rows": count, "missing_fact_sources": sorted(set(missing_sources)),
         "metric_method": "deterministic_lexical_number_proxy",
         "config_hash": config_hash(), "timestamp": utc_now(),
         "pilot_excluded_from_final_statistics": mode == "pilot"},
    )
    if len(evaluated) != len(rows):
        raise SystemExit(f"Fact F1 incomplete: {len(evaluated)}/{len(rows)}; manifest={manifest}")
    print(f"Fact F1 complete: {len(evaluated)} rows; output={output}; manifest={manifest}")

if __name__ == "__main__":
    main()
