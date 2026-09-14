# SPDX-FileCopyrightText: 2026 Yi Liu
# SPDX-License-Identifier: Apache-2.0

"""Run blind framing detection for Qwen, DeepSeek, or Kimi."""

from __future__ import annotations

import argparse
import math

from evaluator_common import (add_evaluator_args, load_canonical, load_contaminated,
                              run_parallel_jsonl, selected_mode)
from pipeline_common import load_configs, render, retry_evaluated, split_prompt


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
    parser = argparse.ArgumentParser(description="Blind framing detection.")
    add_evaluator_args(parser, use_candidates=True)
    args = parser.parse_args()
    mode = selected_mode(args)
    cfg, _ = load_configs()
    max_output_tokens = int(cfg["inference"]["detection_max_output_tokens"])
    sources = {row["source_id"]: row for row in load_canonical(mode)}
    contaminated = load_contaminated(mode, use_candidates=args.use_candidates)
    system, template = split_prompt(__import__("pathlib").Path(__file__).resolve().parents[1] / "prompts" / "D0_detection.md")
    items = []
    for row in contaminated:
        items.append({
            "item_id": row["sample_id"], "sample_id": row["sample_id"], "source_id": row["source_id"],
            "article": row["transformed_article"], "truth_issue": {"lexical_affective": "lexical",
            "agency_prominence": "agency", "salience_order": "salience"}[row["operator"]],
            "truth_direction": row["direction"], "truth_target_actor": row["target_actor"],
            "clean": False,
        })
    for source in sources.values():
        items.append({
            "item_id": f"clean__{source['source_id']}", "sample_id": None,
            "source_id": source["source_id"], "article": source["canonical_text"],
            "truth_issue": "none", "truth_direction": "none",
            "truth_target_actor": None, "clean": True,
        })

    def worker(item: dict) -> dict:
        user = render(template, {"ARTICLE": item["article"]})
        value, metadata, attempts = retry_evaluated(
            family=args.family, stage="detection", item_id=item["item_id"],
            system=system, user=user, expected_json=True,
            reasoning_mode="off", max_output_tokens=max_output_tokens,
        )
        validate_detection(value)
        return {**item, "model_output": value, "model_metadata": metadata,
                "attempts": attempts, "valid": True}

    completed, _ = run_parallel_jsonl(
        family=args.family, stage="detection", mode=mode, items=items,
        worker=worker, workers=args.workers, resume=args.resume,
    )
    if completed != len(items):
        raise SystemExit(f"Detection incomplete: {completed}/{len(items)}")


if __name__ == "__main__":
    main()
