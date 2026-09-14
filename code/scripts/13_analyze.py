# SPDX-FileCopyrightText: 2026 Yi Liu
# SPDX-License-Identifier: Apache-2.0

"""Aggregate pilot/full detection and reconstruction metrics.

Pilot outputs are descriptive only. Source-clustered bootstrap intervals are
enabled for a full run and are never presented as final inferential statistics
for the 10-source pilot.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from metric_common import load_reconstruction_rows
from pipeline_common import ROOT, jsonl_read, write_manifest


FAMILIES = ("qwen", "deepseek", "kimi")
LABELS = ("none", "lexical", "agency", "salience")
OPERATORS = ("lexical_affective", "agency_prominence", "salience_order")
STRENGTHS = ("low", "medium", "high")


def f1_for_label(rows: list[dict[str, Any]], label: str) -> float:
    tp = sum(row.get("predicted_issue") == label and row.get("truth_issue") == label for row in rows)
    fp = sum(row.get("predicted_issue") == label and row.get("truth_issue") != label for row in rows)
    fn = sum(row.get("predicted_issue") != label and row.get("truth_issue") == label for row in rows)
    if 2 * tp + fp + fn == 0:
        return 0.0
    return 2 * tp / (2 * tp + fp + fn)


def detection_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    prepared = []
    for row in rows:
        output = row.get("model_output", {}) or {}
        prepared.append({
            "source_id": row.get("source_id"), "truth_issue": row.get("truth_issue", "none"),
            "predicted_issue": output.get("presentation_issue", "none"),
            "truth_direction": row.get("truth_direction", "none"),
            "predicted_direction": output.get("direction", "none"),
            "clean": bool(row.get("clean")),
        })
    per_class = {label: round(f1_for_label(prepared, label), 6) for label in LABELS}
    macro = sum(per_class.values()) / len(LABELS)
    contaminated = [row for row in prepared if not row["clean"]]
    clean = [row for row in prepared if row["clean"]]
    direction_accuracy = (
        sum(row["truth_direction"] == row["predicted_direction"] for row in contaminated) / len(contaminated)
        if contaminated else None
    )
    false_positive = (
        sum(row["predicted_issue"] != "none" for row in clean) / len(clean)
        if clean else None
    )
    return {
        "n": len(prepared), "contaminated_n": len(contaminated), "clean_n": len(clean),
        "macro_f1": round(macro, 6), "per_class_f1": per_class,
        "direction_accuracy_contaminated": round(direction_accuracy, 6) if direction_accuracy is not None else None,
        "clean_false_positive_rate": round(false_positive, 6) if false_positive is not None else None,
    }


def mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def grouped_mean(rows: list[dict[str, Any]], value: Callable[[dict[str, Any]], float | None],
                 predicate: Callable[[dict[str, Any]], bool] | None = None) -> dict[str, Any]:
    selected = [row for row in rows if predicate is None or predicate(row)]
    values = [float(value(row)) for row in selected if value(row) is not None]
    return {"n": len(values), "mean": mean(values)}


def cluster_bootstrap(rows: list[dict[str, Any]], value: Callable[[dict[str, Any]], float | None],
                      *, reps: int = 5000, seed: int = 20260907) -> dict[str, Any]:
    """Source-clustered percentile bootstrap for a scalar row-level metric."""
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if value(row) is not None:
            clusters[str(row.get("source_id"))].append(row)
    cluster_ids = sorted(clusters)
    if not cluster_ids:
        return {"n_sources": 0, "replicates": reps, "estimate": None, "ci95": [None, None]}
    point_values = [float(value(row)) for row in rows if value(row) is not None]
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(reps):
        sampled = [rng.choice(cluster_ids) for _ in cluster_ids]
        values = [float(value(row)) for cluster_id in sampled for row in clusters[cluster_id]]
        estimates.append(sum(values) / len(values))
    estimates.sort()
    lo = estimates[max(0, int(0.025 * reps) - 1)]
    hi = estimates[min(reps - 1, int(0.975 * reps))]
    return {"n_sources": len(cluster_ids), "replicates": reps,
            "estimate": round(sum(point_values) / len(point_values), 6),
            "ci95": [round(lo, 6), round(hi, 6)]}


def irr_summary(rows: list[dict[str, Any]], *, bootstrap: bool = False) -> dict[str, Any]:
    """Aggregate IRR as total reversed edits divided by total injected edits."""
    reversed_count = sum(int(row.get("irr_reversed_count", 0) or 0) for row in rows)
    injected_count = sum(int(row.get("irr_injected_edit_count", 0) or 0) for row in rows)
    result = {"n": len(rows), "reversed_count": reversed_count,
              "injected_edit_count": injected_count,
              "mean": round(reversed_count / injected_count, 6) if injected_count else None}
    if bootstrap:
        result["bootstrap_source_clustered"] = cluster_bootstrap_ratio(rows)
    else:
        result["bootstrap_source_clustered"] = None
    return result


def cluster_bootstrap_ratio(rows: list[dict[str, Any]], *, reps: int = 5000,
                            seed: int = 20260907) -> dict[str, Any]:
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if int(row.get("irr_injected_edit_count", 0) or 0):
            clusters[str(row.get("source_id"))].append(row)
    cluster_ids = sorted(clusters)
    reversed_count = sum(int(row.get("irr_reversed_count", 0) or 0) for row in rows)
    injected_count = sum(int(row.get("irr_injected_edit_count", 0) or 0) for row in rows)
    if not cluster_ids or not injected_count:
        return {"n_sources": 0, "replicates": reps, "estimate": None, "ci95": [None, None]}
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(reps):
        sampled = [rng.choice(cluster_ids) for _ in cluster_ids]
        rev = sum(int(row.get("irr_reversed_count", 0) or 0) for cid in sampled for row in clusters[cid])
        inj = sum(int(row.get("irr_injected_edit_count", 0) or 0) for cid in sampled for row in clusters[cid])
        estimates.append(rev / inj if inj else 0.0)
    estimates.sort()
    lo = estimates[max(0, int(0.025 * reps) - 1)]
    hi = estimates[min(reps - 1, int(0.975 * reps))]
    return {"n_sources": len(cluster_ids), "replicates": reps,
            "estimate": round(reversed_count / injected_count, 6),
            "ci95": [round(lo, 6), round(hi, 6)]}


def read_detection(family: str, mode: str) -> list[dict[str, Any]]:
    path = ROOT / "data/results" / "evaluated" / family / f"detection_{mode}_direct.jsonl"
    rows = [row for row in jsonl_read(path) if row.get("valid", True)]
    if not rows:
        raise FileNotFoundError(path)
    return rows


def read_metric(family: str, name: str, mode: str) -> list[dict[str, Any]]:
    path = ROOT / "data/results" / "metrics" / family / f"{name}_{mode}.jsonl"
    rows = [row for row in jsonl_read(path) if row.get("valid", True)]
    if not rows:
        raise FileNotFoundError(path)
    return rows


def expected_full_ids(mode: str) -> tuple[set[str], set[str]]:
    """Return frozen contamination and clean IDs for explicit missingness accounting."""
    contaminated_path = ROOT / "data" / "contaminated" / f"{mode}_glm52.jsonl"
    canonical_path = ROOT / "data" / "canonical" / f"{mode}.jsonl"
    contaminated = {str(row.get("sample_id")) for row in jsonl_read(contaminated_path)}
    clean = {f"clean__{row.get('source_id')}" for row in jsonl_read(canonical_path)}
    return contaminated, clean


def summarize_family(family: str, mode: str, *, bootstrap: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    detection_rows = read_detection(family, mode)
    detection = detection_stats(detection_rows)
    expected_contaminated, expected_clean = expected_full_ids(mode)
    expected_detection = expected_contaminated | expected_clean
    observed_detection = {str(row.get("item_id")) for row in detection_rows}
    missing_detection = sorted(expected_detection - observed_detection)
    detection_by_id = {row.get("item_id"): row for row in detection_rows}
    reconstruction_rows = load_reconstruction_rows(family, mode)
    observed_reconstruction = {str(row.get("item_id")) for row in reconstruction_rows}
    missing_reconstruction = sorted(expected_contaminated - observed_reconstruction)
    reconstruction_by_id = {row.get("item_id"): row for row in reconstruction_rows}
    fact_rows = read_metric(family, "fact_f1", mode)
    irr_rows = read_metric(family, "irr", mode)
    irr_by_id = {row.get("item_id"): row for row in irr_rows}
    joined: list[dict[str, Any]] = []
    for row in fact_rows:
        irr = irr_by_id.get(row.get("item_id"), {}).get("metric", {})
        metric = row.get("metric", {})
        reconstruction = reconstruction_by_id.get(row.get("item_id"), {})
        active_detection = detection_by_id.get(row.get("item_id"), {})
        joined.append({
            "item_id": row.get("item_id"), "source_id": row.get("source_id"),
            "operator": row.get("operator"), "strength": row.get("strength"),
            "fact_f1": metric.get("fact_f1"), "fact_precision": metric.get("fact_precision"),
            "fact_recall": metric.get("fact_recall"), "irr": irr.get("irr"),
            "irr_decisive": irr.get("irr_decisive"),
            "irr_reversed_count": irr.get("reversed_count", 0),
            "irr_injected_edit_count": irr.get("injected_edit_count", 0),
            "d0_context_aligned": reconstruction.get("detection_result") == active_detection.get("model_output"),
        })
    aligned_rows = [row for row in joined if row.get("d0_context_aligned")]
    analysis_rows = aligned_rows
    summary: dict[str, Any] = {
        "detection": detection, "reconstruction_n": len(analysis_rows),
        "reconstruction_total_n": len(joined),
        "provider_missingness": {
            "expected_detection_n": len(expected_detection),
            "observed_detection_n": len(observed_detection),
            "missing_detection_n": len(missing_detection),
            "missing_detection_ids": missing_detection,
            "expected_reconstruction_n": len(expected_contaminated),
            "observed_reconstruction_n": len(observed_reconstruction),
            "missing_reconstruction_n": len(missing_reconstruction),
            "missing_reconstruction_ids": missing_reconstruction,
            "treatment": "provider_blocked_or_execution_missing; never imputed as a model label",
        },
        "reconstruction_context_alignment": {
            "aligned_n": len(aligned_rows), "total_n": len(joined),
            "excluded_n": len(joined) - len(aligned_rows),
            "status": "aligned" if len(aligned_rows) == len(joined) else "partial_or_mismatched",
            "note": "Main reconstruction metrics use only rows whose embedded D0 result exactly matches the active detection file.",
        },
        "overall": {}, "by_operator": {}, "by_strength": {}, "by_operator_strength": {},
        "provisional_all_reconstruction_rows": {
            "fact_f1": mean([float(row["fact_f1"]) for row in joined if row.get("fact_f1") is not None]),
            "irr": irr_summary(joined),
            "n": len(joined), "warning": "Not used for the main table when D0 context is mismatched.",
        },
    }
    rows = [row for row in analysis_rows if row.get("fact_f1") is not None]
    entry = grouped_mean(rows, lambda item: item.get("fact_f1"))
    entry["bootstrap_source_clustered"] = cluster_bootstrap(rows, lambda item: item.get("fact_f1")) if bootstrap else None
    summary["overall"]["fact_f1"] = entry
    summary["overall"]["irr"] = irr_summary(analysis_rows, bootstrap=bootstrap)
    for operator in OPERATORS:
        summary["by_operator"][operator] = {}
        rows = [row for row in analysis_rows if row.get("operator") == operator and row.get("fact_f1") is not None]
        summary["by_operator"][operator]["fact_f1"] = grouped_mean(rows, lambda item: item.get("fact_f1"))
        summary["by_operator"][operator]["irr"] = irr_summary([row for row in analysis_rows if row.get("operator") == operator])
    for strength in STRENGTHS:
        summary["by_strength"][strength] = {}
        rows = [row for row in analysis_rows if row.get("strength") == strength and row.get("fact_f1") is not None]
        summary["by_strength"][strength]["fact_f1"] = grouped_mean(rows, lambda item: item.get("fact_f1"))
        summary["by_strength"][strength]["irr"] = irr_summary([row for row in analysis_rows if row.get("strength") == strength])
    table_rows: list[dict[str, Any]] = []
    for operator in OPERATORS:
        for strength in STRENGTHS:
            rows = [row for row in analysis_rows if row.get("operator") == operator and row.get("strength") == strength]
            table_rows.append({
                "family": family, "operator": operator, "strength": strength, "n": len(rows),
                "fact_f1": mean([float(row["fact_f1"]) for row in rows if row.get("fact_f1") is not None]),
                "irr": irr_summary(rows)["mean"],
                "d0_context_alignment": summary["reconstruction_context_alignment"]["status"],
            })
    return summary, table_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze detection and reconstruction outputs.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pilot", action="store_true")
    group.add_argument("--full", action="store_true")
    args = parser.parse_args()
    mode = "pilot" if args.pilot else "full"
    family_summaries: dict[str, Any] = {}
    table_rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        family_summaries[family], rows = summarize_family(family, mode, bootstrap=(mode == "full"))
        table_rows.extend(rows)
    has_provider_missingness = any(
        item["provider_missingness"]["missing_detection_n"]
        or item["provider_missingness"]["missing_reconstruction_n"]
        for item in family_summaries.values()
    )
    summary = {
        "schema_version": "analysis-v1",
        "mode": mode,
        "analysis_status": (
            "pilot_descriptive" if mode == "pilot" else
            ("full_inferential_with_provider_blocked_missingness" if has_provider_missingness else "full_inferential")
        ),
        "pilot_excluded_from_final_statistics": mode == "pilot",
        "bootstrap_plan": {"method": "source_clustered_percentile", "replicates": 5000, "ci": 0.95,
                            "run": mode == "full", "note": "Pilot intervals are intentionally not treated as final inference; provider-blocked rows remain explicit missingness."},
        "families": family_summaries,
        "table_rows": table_rows,
    }
    out = ROOT / "data/results" / "summaries" / f"{mode}_analysis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise SystemExit(f"Refusing to overwrite {out}; remove or archive it before rerunning")
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    table = ROOT / "data/results" / "tables" / f"{mode}_reconstruction_by_condition.csv"
    with table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["family", "operator", "strength", "n", "fact_f1", "irr", "d0_context_alignment"])
        writer.writeheader()
        writer.writerows(table_rows)
    manifest = write_manifest(f"analysis", mode, {"summary": str(out.relative_to(ROOT)),
        "table": str(table.relative_to(ROOT)), "families": FAMILIES,
        "pilot_excluded_from_final_statistics": mode == "pilot"})
    print(f"Analysis complete: {out}; table={table}; manifest={manifest}")

if __name__ == "__main__":
    main()
