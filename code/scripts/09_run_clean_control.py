# SPDX-FileCopyrightText: 2026 Yi Liu
# SPDX-License-Identifier: Apache-2.0

"""Run the clean reconstruction control."""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluator_common import add_evaluator_args, load_canonical, run_parallel_jsonl, selected_mode
from pipeline_common import render, retry_evaluated, split_prompt


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean reconstruction control.")
    add_evaluator_args(parser)
    args = parser.parse_args()
    mode = selected_mode(args)
    sources = load_canonical(mode)
    root = Path(__file__).resolve().parents[1]
    system, template = split_prompt(root / "prompts" / "R-CLEAN_clean_reconstruction.md")
    items = [{**source, "item_id": f"clean__{source['source_id']}"} for source in sources]

    def worker(item: dict) -> dict:
        user = render(template, {"CLEAN_SOURCE": item["canonical_text"]})
        value, metadata, attempts = retry_evaluated(
            family=args.family, stage="clean_control", item_id=item["item_id"],
            system=system, user=user, expected_json=False,
            reasoning_mode="off", max_output_tokens=500,
        )
        if not isinstance(value, str) or not value.strip():
            raise ValueError("empty clean reconstruction")
        return {"item_id": item["item_id"], "source_id": item["source_id"],
                "source_article": item["canonical_text"],
                "reconstructed_article": value.strip(), "model_metadata": metadata,
                "attempts": attempts, "valid": True}

    completed, _ = run_parallel_jsonl(
        family=args.family, stage="clean_control", mode=mode, items=items,
        worker=worker, workers=args.workers, resume=args.resume,
    )
    if completed != len(items):
        raise SystemExit(f"Clean control incomplete: {completed}/{len(items)}")


if __name__ == "__main__":
    main()
