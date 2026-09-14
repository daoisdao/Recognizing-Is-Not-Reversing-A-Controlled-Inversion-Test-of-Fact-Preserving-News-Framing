# SPDX-FileCopyrightText: 2026 Yi Liu
# SPDX-License-Identifier: Apache-2.0

"""Compute paired high-strength thinking-minus-direct deltas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from metric_common import load_reconstruction_rows
from pipeline_common import ROOT, jsonl_read, write_manifest


def detection_stats(rows: list[dict]) -> dict:
    labels = ("none", "lexical", "agency", "salience")
    prepared = [{
        "truth_issue": r.get("truth_issue", "none"),
        "predicted_issue": (r.get("model_output") or {}).get("presentation_issue", "none"),
        "truth_direction": r.get("truth_direction", "none"),
        "predicted_direction": (r.get("model_output") or {}).get("direction", "none"),
    } for r in rows]
    f1s = {}
    for label in labels:
        tp = sum(x["predicted_issue"] == label and x["truth_issue"] == label for x in prepared)
        fp = sum(x["predicted_issue"] == label and x["truth_issue"] != label for x in prepared)
        fn = sum(x["predicted_issue"] != label and x["truth_issue"] == label for x in prepared)
        f1s[label] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    contaminated = [x for x in prepared if x["truth_issue"] != "none"]
    return {
        "n": len(prepared),
        "macro_f1": sum(f1s.values()) / len(labels),
        "direction_accuracy": sum(x["truth_direction"] == x["predicted_direction"] for x in contaminated) / len(contaminated),
        "per_class_f1": f1s,
    }


def metric_summary(rows: list[dict]) -> dict:
    facts = [float((r.get("metric") or {}).get("fact_f1")) for r in rows if (r.get("metric") or {}).get("fact_f1") is not None]
    reversed_count = sum(int((r.get("metric") or {}).get("reversed_count", 0) or 0) for r in rows)
    injected_count = sum(int((r.get("metric") or {}).get("injected_edit_count", 0) or 0) for r in rows)
    return {"n": len(rows), "fact_f1": sum(facts) / len(facts) if facts else None,
            "irr": reversed_count / injected_count if injected_count else None,
            "reversed_count": reversed_count, "injected_edit_count": injected_count}


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze high-strength reasoning ablation deltas.")
    parser.add_argument("--full", action="store_true", required=True)
    parser.add_argument("--family", choices=("qwen", "deepseek", "kimi"))
    args = parser.parse_args()
    contaminated = jsonl_read(ROOT / "data" / "contaminated" / "full_glm52.jsonl")
    high_ids = {r["sample_id"] for r in contaminated if r["strength"] == "high"}
    families = (args.family,) if args.family else ("qwen", "deepseek", "kimi")
    output = {}
    for family in families:
        direct_path = ROOT / "data/results" / "evaluated" / family / "detection_full_direct.jsonl"
        thinking_path = ROOT / "data/results" / "evaluated" / family / "detection_full_thinking.jsonl"
        direct_detection = [r for r in jsonl_read(direct_path) if r.get("valid", True) and r.get("item_id") in high_ids]
        thinking_detection = [r for r in jsonl_read(thinking_path) if r.get("valid", True) and r.get("item_id") in high_ids]
        direct_fact = [r for r in jsonl_read(ROOT / "data/results" / "metrics" / family / "fact_f1_full.jsonl") if r.get("valid", True) and r.get("item_id") in high_ids]
        thinking_fact = [r for r in jsonl_read(ROOT / "data/results" / "metrics" / family / "fact_f1_full_thinking.jsonl") if r.get("valid", True) and r.get("item_id") in high_ids]
        direct_irr = [r for r in jsonl_read(ROOT / "data/results" / "metrics" / family / "irr_full.jsonl") if r.get("valid", True) and r.get("item_id") in high_ids]
        thinking_irr = [r for r in jsonl_read(ROOT / "data/results" / "metrics" / family / "irr_full_thinking.jsonl") if r.get("valid", True) and r.get("item_id") in high_ids]
        d0, dt = detection_stats(direct_detection), detection_stats(thinking_detection)
        r0, rt = metric_summary(direct_fact + direct_irr), metric_summary(thinking_fact + thinking_irr)
        output[family] = {
            "availability": {
                "expected_high_n": len(high_ids),
                "direct_detection_n": len(direct_detection),
                "thinking_detection_n": len(thinking_detection),
                "direct_fact_f1_n": len(direct_fact),
                "thinking_fact_f1_n": len(thinking_fact),
                "direct_irr_n": len(direct_irr),
                "thinking_irr_n": len(thinking_irr),
                "missingness_treatment": "provider_blocked_or_timeout rows remain missing; deltas use returned paired rows only",
            },
            "direct_high": {"detection": d0, "fact_f1": metric_summary(direct_fact), "irr": metric_summary(direct_irr)},
            "thinking_high": {"detection": dt, "fact_f1": metric_summary(thinking_fact), "irr": metric_summary(thinking_irr)},
            "delta_thinking_minus_direct": {
                "detection_macro_f1": dt["macro_f1"] - d0["macro_f1"],
                "detection_direction_accuracy": dt["direction_accuracy"] - d0["direction_accuracy"],
                "fact_f1": (metric_summary(thinking_fact)["fact_f1"] - metric_summary(direct_fact)["fact_f1"]),
                "irr": (metric_summary(thinking_irr)["irr"] - metric_summary(direct_irr)["irr"]),
            },
        }
    out = ROOT / "data/results" / "summaries" / "full_reasoning_deltas.json"
    out.write_text(json.dumps({"mode": "full", "high_strength_only": True, "families": output}, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = write_manifest("reasoning_deltas", "full", {"output": str(out.relative_to(ROOT)), "families": list(families), "high_strength_only": True})
    print(f"Reasoning deltas written: {out}; manifest={manifest}")


if __name__ == "__main__":
    main()
