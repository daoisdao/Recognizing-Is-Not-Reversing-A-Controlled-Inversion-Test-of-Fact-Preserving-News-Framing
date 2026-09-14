# SPDX-FileCopyrightText: 2026 Yi Liu
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from pipeline_common import (FatalAPIError, ROOT, add_mode_args, append_jsonl, call_generator,
                             check_generator_output, condition_rows, generator_artifact_tag, jsonl_read,
                             jsonl_write_new, render, selected_mode, split_prompt,
                             write_manifest)


class GenerationGateError(ValueError):
    def __init__(self, failures, value):
        super().__init__(",".join(failures))
        self.failures = failures
        self.value = value


def generate(system, template, source, fact_row, condition, attempt, repair_context=""):
    user = render(template, {
        "TARGET_ACTOR": source["target_actor"],
        "FAVORABLE_OR_UNFAVORABLE": condition["direction"],
        "OPERATOR": condition["operator"],
        "LOW_MEDIUM_HIGH": condition["strength"],
        "ATOMIC_FACT_LIST": fact_row["facts"],
        "SOURCE_ARTICLE": source["canonical_text"],
    })
    if repair_context:
        user += "\n\nREPAIR THE PREVIOUS ATTEMPT STRICTLY:\n" + repair_context
    value, metadata = call_generator(
        stage="contamination_generation", item_id=condition["sample_id"],
        system=system, user=user, expected_json=True, attempt=attempt,
    )
    failures = check_generator_output(value, condition, source["canonical_text"])
    if failures:
        raise GenerationGateError(failures, value)
    return value, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the 3x3 contamination grid with the configured GLM5.2 generator.")
    add_mode_args(parser)
    parser.add_argument("--workers", type=int, default=1, choices=range(1, 9))
    parser.add_argument("--sample-ids", nargs="+", help="Generate only these explicit condition IDs for manual review.")
    args = parser.parse_args()
    mode = selected_mode(args)
    canonical_path = ROOT / "data" / "canonical" / f"{mode}.jsonl"
    facts_path = ROOT / "data" / "facts" / f"{mode}.jsonl"
    artifact_tag = generator_artifact_tag()
    review_ids = set(args.sample_ids or [])
    output_name = f"{mode}_review_{artifact_tag}.jsonl" if review_ids else f"{mode}_candidates_{artifact_tag}.jsonl"
    output_path = ROOT / "data" / "contaminated" / output_name
    conditions_path = ROOT / "data" / "splits" / f"{mode}_conditions.jsonl"
    if output_path.exists() and not args.resume:
        raise SystemExit(f"Refusing to overwrite {output_path}; use --resume")
    sources = jsonl_read(canonical_path)
    facts = {row["source_id"]: row for row in jsonl_read(facts_path)}
    conditions = condition_rows(sources)
    known_ids = {condition["sample_id"] for condition in conditions}
    unknown_ids = sorted(review_ids - known_ids)
    if unknown_ids:
        raise SystemExit(f"Unknown sample IDs: {unknown_ids}")
    jsonl_write_new(conditions_path, conditions, resume=args.resume)
    existing = {row["sample_id"] for row in jsonl_read(output_path)} if args.resume else set()
    source_map = {row["source_id"]: row for row in sources}
    system, template = split_prompt(ROOT / "code/prompts" / "P1_contamination_generator.md")
    pending = [condition for condition in conditions
               if (not review_ids or condition["sample_id"] in review_ids)
               and condition["sample_id"] not in existing]
    log_lock = threading.Lock()
    fatal_event = threading.Event()

    def process(condition):
        if fatal_event.is_set():
            raise FatalAPIError("batch cancelled after a non-retryable API failure")
        sample_id = condition["sample_id"]
        source = source_map[condition["source_id"]]
        last_error = None
        repair_context = ""
        for attempt in range(1, 4):
            try:
                value, metadata = generate(
                    system, template, source, facts[source["source_id"]], condition, attempt,
                    repair_context=repair_context,
                )
                return ({
                    **condition, "target_actor": source["target_actor"],
                    "transformed_article": value["transformed_article"].strip(),
                    "edit_plan": value["edit_plan"], "self_check": value["self_check"],
                    "generation_attempt": attempt, "generator_metadata": metadata,
                }, None)
            except GenerationGateError as exc:
                last_error = exc
                repair_context = (
                    "The prior output failed these deterministic checks: "
                    + ", ".join(exc.failures)
                    + ". Rewrite the transformed article so every edit is actually present, "
                      "then copy exact before/after substrings from the finished text. Do not "
                      "invent an edit-plan entry. PRIOR TRANSFORMED ARTICLE:\n"
                    + str(exc.value.get("transformed_article", ""))
                    + "\nPRIOR EDIT PLAN:\n"
                    + json.dumps(exc.value.get("edit_plan", []), ensure_ascii=False)
                )
                with log_lock:
                    append_jsonl(ROOT / "code/logs" / "invalid_samples.jsonl", {
                        **condition, "stage": "contamination_generation", "mode": mode,
                        "attempt": attempt, "failure_reason": str(exc),
                    })
            except FatalAPIError:
                fatal_event.set()
                raise
            except Exception as exc:
                last_error = exc
                with log_lock:
                    append_jsonl(ROOT / "code/logs" / "invalid_samples.jsonl", {
                        **condition, "stage": "contamination_generation", "mode": mode,
                        "attempt": attempt, "failure_reason": str(exc),
                    })
        return (None, {**condition, "attempts": 3, "failure_reason": str(last_error)})

    completed = 0
    invalid_final = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process, condition) for condition in pending]
        for future in as_completed(futures):
            row, invalid = future.result()
            if row is not None:
                append_jsonl(output_path, row)
                completed += 1
            if invalid is not None:
                invalid_final.append(invalid)
    rows = jsonl_read(output_path)
    expected = len(review_ids) if review_ids else len(sources) * 9
    if len(rows) != expected:
        raise SystemExit(f"Expected {expected} candidates, found {len(rows)}")
    direction_counts = {d: len({(r['source_id'], r['operator']) for r in rows if r['direction'] == d}) for d in ('favorable', 'unfavorable')}
    if not review_ids and direction_counts != {"favorable": 15 if mode == "pilot" else 90, "unfavorable": 15 if mode == "pilot" else 90}:
        raise SystemExit(f"Direction assignment is not balanced by source/operator pair: {direction_counts}")
    manifest = write_manifest("05_generate_contamination", mode, {
        "canonical_input": str(canonical_path.relative_to(ROOT)), "facts_input": str(facts_path.relative_to(ROOT)),
        "conditions": str(conditions_path.relative_to(ROOT)), "output": str(output_path.relative_to(ROOT)),
        "completed_this_run": completed, "total": len(rows), "direction_pair_counts": direction_counts,
        "workers": args.workers, "generator_artifact_tag": artifact_tag,
        "review_sample_ids": sorted(review_ids) if review_ids else None,
        "invalid_after_three_attempts": invalid_final,
    })
    if invalid_final:
        raise SystemExit(f"{len(invalid_final)} samples exhausted 3 attempts; manifest={manifest}")
    print(f"Generated {len(rows)} candidates -> {output_path}; manifest={manifest}")


if __name__ == "__main__":
    main()
