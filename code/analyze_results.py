# SPDX-FileCopyrightText: 2026 Yi Liu
# SPDX-License-Identifier: Apache-2.0

"""Analyze detection, reconstruction, and paired thinking results offline."""
from __future__ import annotations
import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code/scripts"))
from pipeline_common import jsonl_read, OPERATORS, STRENGTHS

FAMILIES = ("deepseek", "qwen", "kimi")
ISSUES = {"lexical_affective": "lexical", "agency_prominence": "agency",
          "salience_order": "salience"}

def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "code/scripts" / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

analysis = module("released_analysis", "13_analyze.py")
reasoning = module("released_reasoning", "16_analyze_reasoning_deltas.py")

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               sort_keys=True) + "\n", encoding="utf-8")

def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def index(rows, key="item_id"):
    result = {row[key]: row for row in rows if row.get("valid", True)}
    assert len(result) == len(rows), "Duplicate or invalid rows"
    return result

def exact(d0):
    output = d0["model_output"]
    return (output.get("presentation_issue") == d0["truth_issue"]
            and output.get("direction") == d0["truth_direction"])

def conditional_summary(rows):
    reverse = sum(r["irr_reversed_count"] for r in rows)
    injected = sum(r["irr_injected_edit_count"] for r in rows)
    return {"n": len(rows), "reversed_count": reverse,
            "injected_edit_count": injected, "irr": reverse/injected if injected else None}

def direct_results():
    output = {}
    table1 = []
    table2 = []
    table3 = []
    strength_rows = []
    pooled = {True: [], False: []}
    for family in FAMILIES:
        summary, conditions = analysis.summarize_family(family, "full", bootstrap=True)
        output[family] = summary
        d0 = index(jsonl_read(ROOT / f"data/results/evaluated/{family}/detection_full_direct.jsonl"))
        recon = index(jsonl_read(ROOT / f"data/results/evaluated/{family}/reconstruction_full_direct.jsonl"))
        fact = index(jsonl_read(ROOT / f"data/results/metrics/{family}/fact_f1_full.jsonl"))
        irr = index(jsonl_read(ROOT / f"data/results/metrics/{family}/irr_full.jsonl"))
        groups = {True: [], False: []}
        for sid, row in recon.items():
            if sid not in d0 or row["detection_result"] != d0[sid]["model_output"]:
                continue
            metric = irr[sid]["metric"]
            joined = {"source_id": row["source_id"],
                      "irr_reversed_count": metric["reversed_count"],
                      "irr_injected_edit_count": metric["injected_edit_count"]}
            correct = exact(d0[sid])
            groups[correct].append(joined)
            pooled[correct].append(joined)
        clean_control_n = len(jsonl_read(ROOT / f"data/results/evaluated/{family}/clean_control_full_direct.jsonl"))
        ds = summary["detection"]
        fs = summary["overall"]["fact_f1"]
        irs = summary["overall"]["irr"]
        table1.append({"model": family, "macro_f1": ds["macro_f1"],
                       "direction_accuracy": ds["direction_accuracy_contaminated"],
                       "clean_fp": ds["clean_false_positive_rate"],
                       "exact_recognition": len(groups[True])/sum(map(len, groups.values())),
                       "d0_n": ds["n"], "clean_control_n": clean_control_n,
                       "fact_f1": fs["mean"], "fact_f1_ci95": fs["bootstrap_source_clustered"]["ci95"],
                       "irr": irs["mean"], "irr_ci95": irs["bootstrap_source_clustered"]["ci95"],
                       "r0_n": summary["reconstruction_n"]})
        good, bad = conditional_summary(groups[True]), conditional_summary(groups[False])
        table2.append({"model": family, "exact_n": good["n"], "other_n": bad["n"],
                       "exact_reversed": good["reversed_count"], "exact_injected": good["injected_edit_count"],
                       "other_reversed": bad["reversed_count"], "other_injected": bad["injected_edit_count"],
                       "irr_exact": good["irr"], "irr_other": bad["irr"], "gain": good["irr"]-bad["irr"]})
        for label, op in (("none", None), *[(ISSUES[o], o) for o in OPERATORS]):
            table3.append({"model": family, "class": label,
                           "detection_f1": ds["per_class_f1"][label],
                           "irr": summary["by_operator"][op]["irr"]["mean"] if op else None})
        for strength in STRENGTHS:
            selected = [r for r in d0.values() if not r["clean"] and f"__{strength}" == r["item_id"][-len(strength)-2:]]
            strength_rows.append({"model": family, "strength": strength,
                            "direction_accuracy": analysis.detection_stats(selected)["direction_accuracy_contaminated"],
                            "irr": summary["by_strength"][strength]["irr"]["mean"]})
    good, bad = conditional_summary(pooled[True]), conditional_summary(pooled[False])
    table2.append({"model": "pooled", "exact_n": good["n"], "other_n": bad["n"],
                   "exact_reversed": good["reversed_count"], "exact_injected": good["injected_edit_count"],
                   "other_reversed": bad["reversed_count"], "other_injected": bad["injected_edit_count"],
                   "irr_exact": good["irr"], "irr_other": bad["irr"], "gain": good["irr"]-bad["irr"]})
    return output, table1, table2, table3, strength_rows

def high_summary(det, fact, irr):
    return {"detection": reasoning.detection_stats(det),
            "fact_f1": reasoning.metric_summary(fact),
            "irr": reasoning.metric_summary(irr)}

def delta(a, b):
    return {"delta_direction": b["detection"]["direction_accuracy"]-a["detection"]["direction_accuracy"],
            "delta_macro_f1": b["detection"]["macro_f1"]-a["detection"]["macro_f1"],
            "delta_fact_f1": b["fact_f1"]["fact_f1"]-a["fact_f1"]["fact_f1"],
            "delta_irr": b["irr"]["irr"]-a["irr"]["irr"]}

def thinking_results(contaminated):
    high_ids = {sid for sid, r in contaminated.items() if r["strength"] == "high"}
    historical, paired = [], []
    detail = {}
    for family in FAMILIES:
        items = {}
        for mode in ("direct", "thinking"):
            suffix = "" if mode == "direct" else "_thinking"
            items[mode] = {
                "d": index([r for r in jsonl_read(ROOT / f"data/results/evaluated/{family}/detection_full_{mode}.jsonl") if r["item_id"] in high_ids]),
                "f": index([r for r in jsonl_read(ROOT / f"data/results/metrics/{family}/fact_f1_full{suffix}.jsonl") if r["item_id"] in high_ids]),
                "i": index([r for r in jsonl_read(ROOT / f"data/results/metrics/{family}/irr_full{suffix}.jsonl") if r["item_id"] in high_ids]),
                "r": index([r for r in jsonl_read(ROOT / f"data/results/evaluated/{family}/reconstruction_full_{mode}.jsonl") if r["item_id"] in high_ids]),
            }
        direct, think = items["direct"], items["thinking"]
        a = high_summary(list(direct["d"].values()), list(direct["f"].values()), list(direct["i"].values()))
        b = high_summary(list(think["d"].values()), list(think["f"].values()), list(think["i"].values()))
        historical.append({"model": family, "direct_d0_n": len(direct["d"]),
                           "thinking_d0_n": len(think["d"]), "direct_r0_n": len(direct["r"]),
                           "thinking_r0_n": len(think["r"]), **delta(a, b)})
        d_ids = set(direct["d"]) & set(think["d"])
        r_ids = set(direct["r"]) & set(think["r"]) & set(direct["f"]) & set(think["f"]) & set(direct["i"]) & set(think["i"])
        r_ids = {sid for sid in r_ids if all(sid in items[m]["d"] and items[m]["r"][sid]["detection_result"] == items[m]["d"][sid]["model_output"] for m in items)}
        def choose(mode):
            x = items[mode]
            return high_summary([x["d"][sid] for sid in sorted(d_ids)],
                                [x["f"][sid] for sid in sorted(r_ids)],
                                [x["i"][sid] for sid in sorted(r_ids)])
        pa, pb = choose("direct"), choose("thinking")
        paired.append({"model": family, "paired_d0_n": len(d_ids), "paired_r0_n": len(r_ids), **delta(pa, pb)})
        detail[family] = {"historical_direct": a, "historical_thinking": b,
                          "paired_direct": pa, "paired_thinking": pb,
                          "paired_d0_ids": sorted(d_ids), "paired_r0_ids": sorted(r_ids)}
    return historical, paired, detail

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/analysis/results")
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit("Choose a new or empty output directory.")
    contaminated = index(
        jsonl_read(ROOT / "data/contaminated/full_glm52.jsonl"), "sample_id"
    )
    full, direct, conditional, operators, strengths = direct_results()
    print("Direct analysis complete (5000 source-clustered bootstrap replicates).", flush=True)
    available, paired, thinking = thinking_results(contaminated)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("direct_metrics", direct),
        ("conditional_irr", conditional),
        ("operator_metrics", operators),
        ("strength_metrics", strengths),
        ("thinking_available_rows", available),
        ("thinking_paired", paired),
    ):
        write_csv(out / (name + ".csv"), rows)
    write_json(out / "direct_analysis.json", full)
    write_json(out / "thinking_analysis.json", thinking)
    print("Results written to: " + str(out), flush=True)

if __name__ == "__main__":
    main()
