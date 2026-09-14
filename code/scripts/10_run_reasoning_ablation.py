# SPDX-FileCopyrightText: 2026 Yi Liu
# SPDX-License-Identifier: Apache-2.0

"""Run the frozen high-strength thinking D0 + R0 ablation."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from evaluator_common import (add_evaluator_args, load_contaminated, output_path,
                              run_parallel_jsonl, selected_mode)
from pipeline_common import jsonl_read, render, retry_evaluated, split_prompt, load_configs


def validate_detection(value: dict) -> None:
    if value.get("presentation_issue") not in {"none", "lexical", "agency", "salience", "other"}:
        raise ValueError("invalid presentation_issue")
    if value.get("direction") not in {"favorable", "unfavorable", "none"}:
        raise ValueError("invalid direction")
    confidence = value.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not math.isfinite(float(confidence)):
        raise ValueError("invalid confidence")
    if not isinstance(value.get("evidence"), list):
        raise ValueError("invalid evidence")
    if not isinstance(value.get("brief_explanation"), str) or len(value["brief_explanation"].split()) > 60:
        raise ValueError("invalid brief_explanation")


def main() -> None:
    parser = argparse.ArgumentParser(description="High-strength thinking D0 + R0 ablation.")
    add_evaluator_args(parser, use_candidates=True)
    parser.add_argument(
        "--allow-partial", action="store_true",
        help="Continue R0 with available thinking D0 rows and retain provider-blocked rows as missing.",
    )
    parser.add_argument(
        "--reconstruct-existing", action="store_true",
        help="Skip thinking D0 calls and reconstruct only from the existing thinking D0 file.",
    )
    args = parser.parse_args()
    mode = selected_mode(args)
    cfg, _ = load_configs()
    # Reasoning-enabled providers may consume the entire direct-condition
    # output budget in hidden reasoning tokens before emitting the final
    # task answer. Keep the frozen direct budget untouched, but reserve a
    # larger completion buffer for the pre-specified thinking ablation.
    thinking_detection_tokens = max(6000, int(cfg["inference"]["detection_max_output_tokens"]))
    thinking_reconstruction_tokens = max(6000, int(cfg["inference"]["reconstruction_max_output_tokens"]))
    contaminated = load_contaminated(mode, use_candidates=args.use_candidates)
    high = [row for row in contaminated if row["strength"] == "high"]
    expected_high = int(cfg["sample"]["final_sources"] if mode == "full" else cfg["sample"]["pilot_sources"]) * 3
    if len(high) != expected_high:
        raise SystemExit(f"Expected 3 high-strength rows per source, found {len(high)}")
    root = Path(__file__).resolve().parents[1]
    d_system, d_template = split_prompt(root / "prompts" / "D0_detection.md")
    d_items = [{
        "item_id": row["sample_id"], "sample_id": row["sample_id"], "source_id": row["source_id"],
        "article": row["transformed_article"],
        "truth_issue": {"lexical_affective": "lexical", "agency_prominence": "agency",
                         "salience_order": "salience"}[row["operator"]],
        "truth_direction": row["direction"], "truth_target_actor": row["target_actor"], "clean": False,
    } for row in high]

    def detection_worker(item: dict) -> dict:
        user = render(d_template, {"ARTICLE": item["article"]})
        value, metadata, attempts = retry_evaluated(
            family=args.family, stage="detection_thinking", item_id=item["item_id"],
            system=d_system, user=user, expected_json=True, reasoning_mode="thinking",
            max_output_tokens=thinking_detection_tokens,
        )
        validate_detection(value)
        return {**item, "model_output": value, "model_metadata": metadata,
                "attempts": attempts, "reasoning_mode": "thinking", "valid": True}

    if not args.reconstruct_existing:
        completed, _ = run_parallel_jsonl(
            family=args.family, stage="detection", mode=mode, items=d_items,
            worker=detection_worker, workers=args.workers, resume=args.resume,
            reasoning_mode="thinking", expected_total=len(d_items),
        )
        if completed != len(d_items) and not args.allow_partial:
            raise SystemExit(f"Thinking detection incomplete: {completed}/{len(d_items)}")
        if completed != len(d_items):
            print(f"Partial thinking ablation: {len(d_items) - completed} high-strength D0 rows unavailable.")

    detection_path = output_path(args.family, "detection", mode, "thinking")
    detection = {row["item_id"]: row for row in jsonl_read(detection_path) if row.get("valid", True)}
    r_system, r_template = split_prompt(root / "prompts" / "R0_reconstruction.md")
    r_items = [{**row, "item_id": row["sample_id"], "detection_result": detection[row["sample_id"]]["model_output"]}
               for row in high if row["sample_id"] in detection]

    def reconstruction_worker(item: dict) -> dict:
        user = render(r_template, {"ARTICLE": item["transformed_article"],
                                   "MODEL_D0_OUTPUT": item["detection_result"]})
        value, metadata, attempts = retry_evaluated(
            family=args.family, stage="reconstruction_thinking", item_id=item["item_id"],
            system=r_system, user=user, expected_json=False, reasoning_mode="thinking",
            max_output_tokens=thinking_reconstruction_tokens,
        )
        if not isinstance(value, str) or not value.strip():
            raise ValueError("empty reconstruction")
        return {"item_id": item["item_id"], "sample_id": item["sample_id"],
                "source_id": item["source_id"], "operator": item["operator"],
                "strength": item["strength"], "direction": item["direction"],
                "target_actor": item["target_actor"], "transformed_article": item["transformed_article"],
                "edit_plan": item["edit_plan"], "detection_result": item["detection_result"],
                "reconstructed_article": value.strip(), "model_metadata": metadata,
                "attempts": attempts, "reasoning_mode": "thinking", "valid": True}

    completed, _ = run_parallel_jsonl(
        family=args.family, stage="reconstruction", mode=mode, items=r_items,
        worker=reconstruction_worker, workers=args.workers, resume=args.resume,
        reasoning_mode="thinking", expected_total=len(high),
    )
    if completed != len(high) and not args.allow_partial:
        raise SystemExit(f"Thinking reconstruction incomplete: {completed}/{len(high)}")


if __name__ == "__main__":
    main()
