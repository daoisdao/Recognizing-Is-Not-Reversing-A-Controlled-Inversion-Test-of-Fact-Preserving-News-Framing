# SPDX-FileCopyrightText: 2026 Yi Liu
# SPDX-License-Identifier: Apache-2.0

"""Repair the six exhausted final contamination rows, then re-run frozen P2.

This is a narrowly scoped recovery step: it does not change prompts, edit
budgets, validator thresholds, or the frozen condition set. Each repaired
article is constructed from the canonical source with explicit, source-only
replacements and is accepted only after the unchanged P2 validator passes.
"""

from __future__ import annotations

import argparse
import importlib
import json

from pipeline_common import (
    ROOT,
    FatalAPIError,
    append_jsonl,
    check_generator_output,
    jsonl_read,
    render,
    split_prompt,
    validator_passed,
    write_manifest,
)


PENDING = {
    "S019__agency_prominence__medium": [
        (
            "Ministers from thirteen Asian countries pledged on Friday to double the wild tiger population on the continent to seven thousand by 2022.",
            "A pledge to double the wild tiger population on the continent to seven thousand by 2022 was made on Friday by ministers from thirteen Asian countries.",
        ),
        (
            "Thai Prime Minister Abhisit Vejjajiva stated in a video address that the wild tiger is in crisis and that this may be the last chance to save it.",
            "In a video address, it was stated by Thai Prime Minister Abhisit Vejjajiva that the wild tiger is in crisis and that this may be the last chance to save it.",
        ),
    ],
    "S019__agency_prominence__high": [
        (
            "Ministers from thirteen Asian countries pledged on Friday to double the wild tiger population on the continent to seven thousand by 2022.",
            "A pledge to double the wild tiger population on the continent to seven thousand by 2022 was made on Friday by ministers from thirteen Asian countries.",
        ),
        (
            "Thai Prime Minister Abhisit Vejjajiva stated in a video address that the wild tiger is in crisis and that this may be the last chance to save it.",
            "In a video address, it was stated by Thai Prime Minister Abhisit Vejjajiva that the wild tiger is in crisis and that this may be the last chance to save it.",
        ),
        (
            "Meeting host Suwit Khunkitti, Thailand's minister of natural resources and environment, noted that there were 100,000 tigers across range countries 100 years ago, compared to about 3,500 today.",
            "It was noted by meeting host Suwit Khunkitti, Thailand's minister of natural resources and environment, that there were 100,000 tigers across range countries 100 years ago, compared to about 3,500 today.",
        ),
    ],
    "S031__agency_prominence__high": [
        (
            "Corus, the world's fifth largest steel producer, announced on Friday that it may be forced to mothball its steelworks in Teesside, England, threatening the jobs of 1,920 employees.",
            "It was announced on Friday by Corus, the world's fifth largest steel producer, that its steelworks in Teesside, England, may be mothballed, threatening the jobs of 1,920 employees.",
        ),
        (
            "Corus may be forced to close its Teesside operations as a consortium has refused to honour a 10-year contract with Corus' Teesside Cast Products, which accounted for 78% of the plant's operations.",
            "The closure of Corus' Teesside operations may be forced by a consortium's refusal to honour a 10-year contract with Corus' Teesside Cast Products, which accounted for 78% of the plant's operations.",
        ),
        (
            "Corus said a 90-day consultation period is due to begin.",
            "A 90-day consultation period is due to begin, Corus said.",
        ),
    ],
    "S034__agency_prominence__high": [
        (
            "AutoTronics Taipei has been held since 2006, bringing together companies from electronic and automobile-related industries.",
            "Since 2006, AutoTronics Taipei has brought together companies from electronic and automobile-related industries.",
        ),
        (
            "Prior to its launch, the Car Electronics Pavilion was established in 2005 at TAITRONICS Autumn (Taipei International Electronic Autumn Show), attracting attention from both industries.",
            "At TAITRONICS Autumn (Taipei International Electronic Autumn Show) in 2005, the Car Electronics Pavilion was established before AutoTronics Taipei's launch, attracting attention from both industries.",
        ),
        (
            "Industry participants noted that success depends on collaboration across industries, as no company can monopolize any industry, even as the automobile industry is expected to become a trillion-dollar industry in Taiwan.",
            "Industry participants noted that collaboration across industries is essential to success, as no company can monopolize any industry, even as the automobile industry is expected to become a trillion-dollar industry in Taiwan.",
        ),
    ],
    "S042__agency_prominence__high": [
        (
            "Wikimedia Argentina and the Wikimedia Foundation have received proposals for endorsement, with decisions pending.",
            "The Wikimedia Foundation and Wikimedia Argentina have received proposals for endorsement, with decisions pending.",
        ),
        (
            "The conference will cover Wikimedia and the free culture movement in relation to communities in North America, including the United States and Canada, and Latin America.",
            "Wikimedia and the free culture movement will be covered by the conference in relation to communities in North America, including the United States and Canada, and Latin America.",
        ),
        (
            "Sponsors include Georgia Tech, the Public Journalism Network, and the Carolina Open Source Initiative.",
            "Georgia Tech, the Public Journalism Network, and the Carolina Open Source Initiative sponsor the conference.",
        ),
    ],
    "S069__lexical_affective__high": [
        (
            "He had no prior criminal record or history of mental illness.",
            "He had no known prior criminal record or history of mental illness.",
        ),
        (
            "Underwood drew suspicion at an information checkpoint set up at the apartment complex.",
            "Underwood only came under suspicion at an information checkpoint set up at the apartment complex.",
        ),
        (
            "Authorities said deep saw marks on the body indicated an attempt to decapitate her.",
            "Authorities said deep saw marks on the body suggested an attempt to decapitate her.",
        ),
        (
            "Police suspected postmortem molestation, as the body was found unclothed.",
            "Police suspected possible postmortem molestation, as the body was found unclothed.",
        ),
        (
            "Authorities believed other individuals, including a woman and a 5-year-old boy, had been targeted and considered.",
            "Authorities believed other individuals, including a woman and a 5-year-old boy, may have been targeted and considered.",
        ),
        (
            "Underwood was arrested on suspicion of first-degree murder and held without bail.",
            "Underwood was arrested on suspicion of first-degree murder and held without bail, pending further proceedings.",
        ),
    ],
}


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    candidates = {row["sample_id"]: row for row in jsonl_read(ROOT / "data/contaminated/full_candidates_glm52.jsonl")}
    canonical = {row["source_id"]: row for row in jsonl_read(ROOT / "data/canonical/full.jsonl")}
    facts = {row["source_id"]: row["facts"] for row in jsonl_read(ROOT / "data/facts/full.jsonl")}
    output_path = ROOT / "data/contaminated/full_glm52.jsonl"
    existing = {row["sample_id"] for row in jsonl_read(output_path)}
    p2_system, p2_template = split_prompt(ROOT / "code/prompts/P2_contamination_validator.md")
    validator = importlib.import_module("06_validate_contamination")
    repaired_ids: list[str] = []
    failures: list[dict[str, object]] = []

    for sample_id, replacements in PENDING.items():
        if sample_id in existing:
            continue
        initial = candidates[sample_id]
        source = canonical[initial["source_id"]]
        article = source["canonical_text"]
        for before, after in replacements:
            if article.count(before) != 1:
                raise ValueError(f"{sample_id}: expected one occurrence of repair text")
            article = article.replace(before, after)
        prior_edits = initial.get("edit_plan") or []
        edits = []
        for index, (before, after) in enumerate(replacements, 1):
            prior = prior_edits[index - 1] if index - 1 < len(prior_edits) else {}
            edits.append({
                "edit_id": f"E{index:02d}", "operator": initial["operator"],
                "before": before, "after": after,
                "affected_fact_ids": prior.get("affected_fact_ids", []),
                "purpose": "Deterministic source-only repair of the requested framing intervention.",
            })
        candidate = {
            **{key: initial[key] for key in ("sample_id", "source_id", "operator", "strength", "direction")},
            "target_actor": source["target_actor"], "transformed_article": article,
            "edit_plan": edits,
            "self_check": {"all_facts_preserved": True, "new_facts_added": False,
                            "facts_removed": False, "cross_operator_changes": False},
            "generation_attempt": 3,
            "generator_metadata": {
                **(initial.get("generator_metadata") or {}),
                "repair": "deterministic_source_only_repair_v1",
                "repair_reason": "three_p2_attempts_exhausted",
            },
        }
        condition = {key: candidate[key] for key in ("sample_id", "source_id", "operator", "strength", "direction")}
        deterministic_failures = check_generator_output(candidate, condition, source["canonical_text"])
        if deterministic_failures:
            failures.append({**condition, "failure_reason": ";".join(deterministic_failures)})
            continue
        last_failures: list[str] = []
        for attempt in range(1, 4):
            try:
                verdict, validator_metadata = validator.validate(
                    p2_system, p2_template, source, facts[condition["source_id"]],
                    condition, candidate, attempt,
                )
                passed, last_failures = validator_passed(
                    verdict, [fact["fact_id"] for fact in facts[condition["source_id"]]]
                )
                if passed:
                    append_jsonl(output_path, {
                        **candidate, "validation": verdict,
                        "validator_metadata": validator_metadata,
                        "validation_attempt": attempt, "valid": True,
                    })
                    repaired_ids.append(sample_id)
                    break
            except FatalAPIError:
                raise
            except Exception as exc:
                last_failures = [str(exc)]
        else:
            failures.append({**condition, "failure_reason": ";".join(last_failures)})

    rows = jsonl_read(output_path)
    manifest = write_manifest("repair_validate_final_contamination", "full", {
        "repaired_ids": repaired_ids, "failed_ids": [row["sample_id"] for row in failures],
        "valid_total": len(rows), "expected": 540, "failures": failures,
    })
    print(json.dumps({"repaired": repaired_ids, "failed": failures,
                      "valid_total": len(rows), "manifest": str(manifest.relative_to(ROOT))},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
