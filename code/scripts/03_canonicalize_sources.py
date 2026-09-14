# SPDX-FileCopyrightText: 2026 Yi Liu
# SPDX-License-Identifier: Apache-2.0

import argparse

from pipeline_common import (FatalAPIError, ROOT, add_mode_args, append_jsonl, expected_source_count,
                             jsonl_read, load_configs, render, retry_generator, selected_mode, split_prompt, word_count,
                             write_manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Canonicalize licensed sources with the configured generator.")
    add_mode_args(parser)
    args = parser.parse_args()
    mode = selected_mode(args)
    cfg, _ = load_configs()
    input_path = ROOT / "data" / "splits" / f"{mode}_filtered_sources.jsonl"
    output_path = ROOT / "data" / "canonical" / f"{mode}.jsonl"
    if output_path.exists() and not args.resume:
        raise SystemExit(f"Refusing to overwrite {output_path}; use --resume")
    existing = {row["source_id"] for row in jsonl_read(output_path)} if args.resume else set()
    system, user_template = split_prompt(ROOT / "code/prompts" / "C0_canonicalize_source.md")
    completed = 0
    for source in jsonl_read(input_path):
        source_id = source["source_id"]
        if source_id in existing:
            continue
        user = render(user_template, {
            "TARGET_ACTOR": source["target_actor"],
            "SOURCE_ARTICLE": source["raw_article_text"],
        })
        last_error = None
        for attempt in range(1, 4):
            try:
                value, metadata = retry_generator(
                    max_attempts=1, stage="canonicalize", item_id=source_id,
                    system=system, user=user, expected_json=True,
                )[:2]
                canonical = str(value.get("canonical_text", "")).strip()
                checks = value.get("self_check") or {}
                count = word_count(canonical)
                if not cfg["sample"]["canonical_words_min"] <= count <= cfg["sample"]["canonical_words_max"]:
                    raise ValueError(f"canonical_word_count={count}")
                if any(checks.get(k) is not True for k in (
                    "source_only", "key_facts_preserved", "entities_numbers_dates_preserved", "neutral_wording"
                )):
                    raise ValueError("canonical self-check failed")
                append_jsonl(output_path, {
                    **{k: source[k] for k in ("source_id", "title", "provenance_url", "revision_id",
                                               "publication_date", "license", "domain", "target_actor")},
                    "canonical_text": canonical, "word_count": count,
                    "generator_metadata": metadata,
                })
                completed += 1
                break
            except FatalAPIError:
                raise
            except Exception as exc:
                last_error = exc
                append_jsonl(ROOT / "code/logs" / "invalid_samples.jsonl", {
                    "stage": "canonicalize", "mode": mode, "source_id": source_id,
                    "attempt": attempt, "failure_reason": str(exc),
                })
        else:
            raise SystemExit(f"Canonicalization failed after 3 attempts for {source_id}: {last_error}")
    rows = jsonl_read(output_path)
    expected = expected_source_count(mode, cfg)
    if len(rows) != expected:
        raise SystemExit(f"Expected {expected} canonical sources, found {len(rows)}")
    manifest = write_manifest("03_canonicalize_sources", mode, {
        "input": str(input_path.relative_to(ROOT)), "output": str(output_path.relative_to(ROOT)),
        "completed_this_run": completed, "total": len(rows),
    })
    print(f"Canonical sources ready: {len(rows)} -> {output_path}; manifest={manifest}")


if __name__ == "__main__":
    main()
