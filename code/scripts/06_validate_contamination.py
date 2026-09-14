# SPDX-FileCopyrightText: 2026 Yi Liu
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor

from pipeline_common import (FatalAPIError, ROOT, add_mode_args, append_jsonl, call_generator,
                             check_generator_output, generator_artifact_tag, jsonl_read, render, selected_mode,
                             split_prompt, validator_passed, write_manifest)


class ValidationGateError(ValueError):
    def __init__(self, failures, candidate, verdict):
        super().__init__(",".join(failures))
        self.failures = failures
        self.candidate = candidate
        self.verdict = verdict


def regenerate(p1_system, p1_template, source, facts, condition, attempt, repair_context=""):
    user = render(p1_template, {
        "TARGET_ACTOR": source["target_actor"],
        "FAVORABLE_OR_UNFAVORABLE": condition["direction"],
        "OPERATOR": condition["operator"], "LOW_MEDIUM_HIGH": condition["strength"],
        "ATOMIC_FACT_LIST": facts, "SOURCE_ARTICLE": source["canonical_text"],
    })
    if repair_context:
        user += "\n\nREPAIR THE PREVIOUS ATTEMPT STRICTLY:\n" + repair_context
    value, metadata = call_generator(
        stage="contamination_regeneration", item_id=condition["sample_id"],
        system=p1_system, user=user, expected_json=True, attempt=attempt,
    )
    failures = check_generator_output(value, condition, source["canonical_text"])
    if failures:
        raise ValueError(",".join(failures))
    return value, metadata


def validate(p2_system, p2_template, source, facts, condition, candidate, attempt):
    user = render(p2_template, {
        "SOURCE_ARTICLE": source["canonical_text"],
        "TRANSFORMED_ARTICLE": candidate["transformed_article"],
        "ATOMIC_FACTS": facts, "OPERATOR": condition["operator"],
        "DIRECTION": condition["direction"], "STRENGTH": condition["strength"],
        "EDIT_PLAN": candidate["edit_plan"],
    })
    return call_generator(
        stage="contamination_validation", item_id=condition["sample_id"],
        system=p2_system, user=user, expected_json=True, attempt=attempt,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Independently validate and retry contaminations.")
    add_mode_args(parser)
    parser.add_argument("--workers", type=int, default=1, choices=range(1, 9))
    args = parser.parse_args()
    mode = selected_mode(args)
    canonical = {r["source_id"]: r for r in jsonl_read(ROOT / "data" / "canonical" / f"{mode}.jsonl")}
    facts_map = {r["source_id"]: r["facts"] for r in jsonl_read(ROOT / "data" / "facts" / f"{mode}.jsonl")}
    artifact_tag = generator_artifact_tag()
    candidates_path = ROOT / "data" / "contaminated" / f"{mode}_candidates_{artifact_tag}.jsonl"
    output_path = ROOT / "data" / "contaminated" / f"{mode}_{artifact_tag}.jsonl"
    if output_path.exists() and not args.resume:
        raise SystemExit(f"Refusing to overwrite {output_path}; use --resume")
    existing = {row["sample_id"] for row in jsonl_read(output_path)} if args.resume else set()
    p1_system, p1_template = split_prompt(ROOT / "code/prompts" / "P1_contamination_generator.md")
    p2_system, p2_template = split_prompt(ROOT / "code/prompts" / "P2_contamination_validator.md")
    pending = [initial for initial in jsonl_read(candidates_path) if initial["sample_id"] not in existing]
    log_lock = threading.Lock()
    fatal_event = threading.Event()

    def process(initial):
        if fatal_event.is_set():
            raise FatalAPIError("batch cancelled after a non-retryable API failure")
        sample_id = initial["sample_id"]
        condition = {k: initial[k] for k in ("sample_id", "source_id", "operator", "strength", "direction")}
        source = canonical[condition["source_id"]]
        facts = facts_map[condition["source_id"]]
        candidate = initial
        last_failures = []
        repair_context = ""
        for attempt in range(1, 4):
            try:
                if attempt > 1:
                    value, gen_metadata = regenerate(
                        p1_system, p1_template, source, facts, condition, attempt,
                        repair_context=repair_context,
                    )
                    candidate = {
                        **condition, "target_actor": source["target_actor"],
                        "transformed_article": value["transformed_article"].strip(),
                        "edit_plan": value["edit_plan"], "self_check": value["self_check"],
                        "generation_attempt": attempt, "generator_metadata": gen_metadata,
                    }
                deterministic_failures = check_generator_output(candidate, condition, source["canonical_text"])
                if deterministic_failures:
                    raise ValueError(",".join(deterministic_failures))
                verdict, validator_metadata = validate(
                    p2_system, p2_template, source, facts, condition, candidate, attempt
                )
                passed, last_failures = validator_passed(verdict, [f["fact_id"] for f in facts])
                if not passed:
                    raise ValidationGateError(last_failures, candidate, verdict)
                return ({
                    **candidate, "validation": verdict, "validator_metadata": validator_metadata,
                    "validation_attempt": attempt, "valid": True,
                }, None)
            except FatalAPIError:
                fatal_event.set()
                raise
            except ValidationGateError as exc:
                last_failures = exc.failures
                repair_context = (
                    "The independent validator rejected the prior output for: "
                    + ", ".join(exc.failures)
                    + ". Rewrite the article and edit plan so every immutable fact, attribution, "
                      "actor role, quantity, date, and causal relation remains unchanged. Remove "
                      "any unsupported evaluation or institutional attribution. Ensure every listed "
                      "edit is actually present. PRIOR TRANSFORMED ARTICLE:\n"
                    + str(exc.candidate.get("transformed_article", ""))
                    + "\nPRIOR EDIT PLAN:\n"
                    + json.dumps(exc.candidate.get("edit_plan", []), ensure_ascii=False)
                    + "\nPRIOR VALIDATOR VERDICT:\n"
                    + json.dumps(exc.verdict, ensure_ascii=False)
                )
                with log_lock:
                    append_jsonl(ROOT / "code/logs" / "invalid_samples.jsonl", {
                        **condition, "stage": "contamination_validation", "mode": mode,
                        "attempt": attempt, "failure_reason": str(exc),
                    })
            except Exception as exc:
                last_failures = [str(exc)]
                repair_context = (
                    "The prior attempt failed: " + str(exc)
                    + ". Return strict JSON and correct the article/edit plan before retrying."
                )
                with log_lock:
                    append_jsonl(ROOT / "code/logs" / "invalid_samples.jsonl", {
                        **condition, "stage": "contamination_validation", "mode": mode,
                        "attempt": attempt, "failure_reason": str(exc),
                    })
        return (None, {**condition, "attempts": 3, "failure_reason": ";".join(last_failures)})

    invalid_final = []
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for row, invalid in executor.map(process, pending):
            if row is not None:
                append_jsonl(output_path, row)
                completed += 1
            if invalid is not None:
                invalid_final.append(invalid)
    rows = jsonl_read(output_path)
    expected = len(canonical) * 9
    manifest = write_manifest("06_validate_contamination", mode, {
        "input": str(candidates_path.relative_to(ROOT)), "output": str(output_path.relative_to(ROOT)),
        "completed_this_run": completed, "valid_total": len(rows), "expected": expected,
        "invalid_after_three_attempts": invalid_final, "workers": args.workers,
        "generator_artifact_tag": artifact_tag,
    })
    if len(rows) != expected:
        raise SystemExit(f"Only {len(rows)}/{expected} samples are valid; {len(invalid_final)} exhausted 3 attempts. manifest={manifest}")
    print(f"Validated {len(rows)} samples -> {output_path}; manifest={manifest}")


if __name__ == "__main__":
    main()
