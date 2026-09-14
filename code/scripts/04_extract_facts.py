# SPDX-FileCopyrightText: 2026 Yi Liu
# SPDX-License-Identifier: Apache-2.0

import argparse

from pipeline_common import (FatalAPIError, ROOT, add_mode_args, append_jsonl, expected_source_count,
                             jsonl_read, load_configs, render, retry_generator, selected_mode, split_prompt,
                             validate_facts, write_manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract 6-10 atomic facts per canonical source.")
    add_mode_args(parser)
    args = parser.parse_args()
    mode = selected_mode(args)
    cfg, _ = load_configs()
    input_path = ROOT / "data" / "canonical" / f"{mode}.jsonl"
    output_path = ROOT / "data" / "facts" / f"{mode}.jsonl"
    if output_path.exists() and not args.resume:
        raise SystemExit(f"Refusing to overwrite {output_path}; use --resume")
    existing = {row["source_id"] for row in jsonl_read(output_path)} if args.resume else set()
    system, user_template = split_prompt(ROOT / "code/prompts" / "P0_atomic_fact_extraction.md")
    completed = 0
    for source in jsonl_read(input_path):
        source_id = source["source_id"]
        if source_id in existing:
            continue
        user = render(user_template, {"SOURCE_ARTICLE": source["canonical_text"]})
        last_error = None
        for attempt in range(1, 4):
            try:
                value, metadata = retry_generator(
                    max_attempts=1, stage="facts", item_id=source_id,
                    system=system, user=user, expected_json=True,
                )[:2]
                # GLM can return a valid inventory with more than the frozen
                # ten-fact cap. Keep the deterministic first ten and record
                # the repair; the validator still enforces the 610 bound.
                if len(value.get("facts") or []) > cfg["sample"]["atomic_facts_max"]:
                    value["facts"] = list(value["facts"])[:cfg["sample"]["atomic_facts_max"]]
                    metadata = {**metadata, "repair": "deterministic_first10_fact_count_bound"}
                facts = validate_facts(
                    source_id, value, cfg["sample"]["atomic_facts_min"], cfg["sample"]["atomic_facts_max"]
                )
                append_jsonl(output_path, {
                    "source_id": source_id, "target_actor": source["target_actor"],
                    "facts": facts, "human_audit_status": "pending_pilot_review",
                    "generator_metadata": metadata,
                })
                completed += 1
                break
            except FatalAPIError:
                raise
            except Exception as exc:
                last_error = exc
                append_jsonl(ROOT / "code/logs" / "invalid_samples.jsonl", {
                    "stage": "facts", "mode": mode, "source_id": source_id,
                    "attempt": attempt, "failure_reason": str(exc),
                })
        else:
            raise SystemExit(f"Fact extraction failed after 3 attempts for {source_id}: {last_error}")
    rows = jsonl_read(output_path)
    expected = expected_source_count(mode, cfg)
    if len(rows) != expected:
        raise SystemExit(f"Expected {expected} fact inventories, found {len(rows)}")
    manifest = write_manifest("04_extract_facts", mode, {
        "input": str(input_path.relative_to(ROOT)), "output": str(output_path.relative_to(ROOT)),
        "completed_this_run": completed, "total": len(rows),
        "human_audit_status": "pending_pilot_review",
    })
    print(f"Fact inventories ready: {len(rows)} -> {output_path}; manifest={manifest}")


if __name__ == "__main__":
    main()
