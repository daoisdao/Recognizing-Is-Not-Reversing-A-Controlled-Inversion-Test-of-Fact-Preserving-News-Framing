# SPDX-FileCopyrightText: 2026 Yi Liu
# SPDX-License-Identifier: Apache-2.0

"""Run blind/self-guided reconstruction using each model's D0 output."""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluator_common import (add_evaluator_args, load_contaminated, output_path,
                              run_parallel_jsonl, selected_mode)
from pipeline_common import jsonl_read, render, retry_evaluated, split_prompt


def main() -> None:
    parser = argparse.ArgumentParser(description="Blind/self-guided reconstruction.")
    add_evaluator_args(parser, use_candidates=True)
    parser.add_argument(
        "--allow-partial", action="store_true",
        help="Continue with rows whose valid D0 result exists; retain missing rows as provider-blocked/missing.",
    )
    args = parser.parse_args()
    mode = selected_mode(args)
    candidates = load_contaminated(mode, use_candidates=args.use_candidates)
    detection_path = output_path(args.family, "detection", mode)
    detection = {row["item_id"]: row for row in jsonl_read(detection_path) if row.get("valid", True)}
    missing = [row["sample_id"] for row in candidates if row["sample_id"] not in detection]
    if missing and not args.allow_partial:
        raise SystemExit(f"Missing D0 results for {len(missing)} contaminated samples; run detection first.")
    if missing:
        print(f"Partial reconstruction: retaining {len(missing)} missing-D0 samples as unavailable.")
    root = Path(__file__).resolve().parents[1]
    system, template = split_prompt(root / "prompts" / "R0_reconstruction.md")
    items = [{**row, "item_id": row["sample_id"], "detection_result": detection[row["sample_id"]]["model_output"]}
             for row in candidates if row["sample_id"] in detection]

    def worker(item: dict) -> dict:
        user = render(template, {"ARTICLE": item["transformed_article"],
                                 "MODEL_D0_OUTPUT": item["detection_result"]})
        value, metadata, attempts = retry_evaluated(
            family=args.family, stage="reconstruction", item_id=item["item_id"],
            system=system, user=user, expected_json=False,
            reasoning_mode="off", max_output_tokens=500,
        )
        if not isinstance(value, str) or not value.strip():
            raise ValueError("empty reconstruction")
        return {"item_id": item["item_id"], "sample_id": item["sample_id"],
                "source_id": item["source_id"], "operator": item["operator"],
                "strength": item["strength"], "direction": item["direction"],
                "target_actor": item["target_actor"],
                "transformed_article": item["transformed_article"],
                "edit_plan": item["edit_plan"], "detection_result": item["detection_result"],
                "reconstructed_article": value.strip(), "model_metadata": metadata,
                "attempts": attempts, "valid": True}

    completed, _ = run_parallel_jsonl(
        family=args.family, stage="reconstruction", mode=mode, items=items,
        worker=worker, workers=args.workers, resume=args.resume,
        expected_total=len(candidates),
    )
    if completed != len(candidates) and not args.allow_partial:
        raise SystemExit(f"Reconstruction incomplete: {completed}/{len(candidates)}")


if __name__ == "__main__":
    main()
