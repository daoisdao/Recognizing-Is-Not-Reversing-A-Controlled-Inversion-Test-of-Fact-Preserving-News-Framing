# SPDX-FileCopyrightText: 2026 Yi Liu
# SPDX-License-Identifier: Apache-2.0

"""Compute deterministic Intervention Reversal Rate from known edit plans."""

from __future__ import annotations

import argparse

from metric_common import evaluate_irr, load_reconstruction_rows, write_jsonl_rows
from pipeline_common import ROOT, config_hash, utc_now, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate reconstruction IRR.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pilot", action="store_true")
    group.add_argument("--full", action="store_true")
    parser.add_argument("--family", choices=("qwen", "deepseek", "kimi"), required=True)
    parser.add_argument("--reasoning-mode", choices=("direct", "thinking"), default="direct")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    mode = "pilot" if args.pilot else "full"
    rows = load_reconstruction_rows(args.family, mode, args.reasoning_mode)
    suffix = "" if args.reasoning_mode == "direct" else f"_{args.reasoning_mode}"
    output = ROOT / "data/results" / "metrics" / args.family / f"irr_{mode}{suffix}.jsonl"
    evaluated = []
    for row in rows:
        metric = evaluate_irr(str(row.get("reconstructed_article", "")), row.get("edit_plan", []))
        evaluated.append({
            "item_id": row.get("item_id") or row.get("sample_id"),
            "sample_id": row.get("sample_id"), "source_id": row.get("source_id"),
            "operator": row.get("operator"), "strength": row.get("strength"),
            "direction": row.get("direction"), "family": args.family,
            "reconstruction_model_metadata": row.get("model_metadata", {}),
            "metric": metric, "valid": True,
        })
    count = write_jsonl_rows(output, evaluated, resume=args.resume)
    manifest = write_manifest(
        f"irr_{args.family}_{args.reasoning_mode}", mode,
        {"family": args.family, "output": str(output.relative_to(ROOT)),
         "expected_reconstruction_rows": len(rows), "evaluated_rows": len(evaluated),
         "written_rows": count,
         "metric_method": "deterministic_normalized_phrase_and_fuzzy_window",
         "config_hash": config_hash(), "timestamp": utc_now(),
         "pilot_excluded_from_final_statistics": mode == "pilot"},
    )
    print(f"IRR complete: {len(evaluated)} rows; output={output}; manifest={manifest}")

if __name__ == "__main__":
    main()
