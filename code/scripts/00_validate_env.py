# SPDX-FileCopyrightText: 2026 Yi Liu
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
from pathlib import Path

from pipeline_common import ROOT, add_mode_args, ensure_layout, expected_source_count, load_configs, selected_mode


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the frozen experiment environment.")
    add_mode_args(parser, resume=False)
    parser.add_argument("--require-api", action="store_true", help="Require the configured generator API key now.")
    parser.add_argument("--require-evaluated-api", action="store_true",
                        help="Require Qwen, DeepSeek, and Kimi runtime settings now.")
    args = parser.parse_args()
    mode = selected_mode(args)
    cfg, models = load_configs()
    errors = []
    if cfg["sample"]["pilot_sources"] != 10 or cfg["sample"]["final_sources"] != 60:
        errors.append("source counts must remain 10 pilot / 60 final")
    if cfg["models"]["generator_family"] != "glm5.2":
        errors.append("generator must be GLM5.2")
    if cfg["models"]["evaluated_families"] != ["qwen", "deepseek", "kimi"]:
        errors.append("evaluated families must be Qwen, DeepSeek, Kimi")
    if cfg["operators"]["active"] != ["lexical_affective", "agency_prominence", "salience_order"]:
        errors.append("operator scope changed")
    if cfg["operators"]["strengths"] != ["low", "medium", "high"]:
        errors.append("strength scope changed")
    if models["generator"].get("family") != "glm5.2":
        errors.append("configured generator family must be GLM5.2")
    if models["generator"].get("model_id") != "glm-5-2-260617":
        errors.append("pilot GLM5.2 model ID must be glm-5-2-260617")
    for rel in ("docs/PROTOCOL.md", "prompts/P0_atomic_fact_extraction.md",
                "prompts/P1_contamination_generator.md", "prompts/P2_contamination_validator.md",
                "prompts/D0_detection.md", "prompts/R0_reconstruction.md",
                "prompts/R-CLEAN_clean_reconstruction.md"):
        if not (ROOT / "code" / rel).is_file():
            errors.append(f"missing required file: {rel}")
    if args.require_api and not os.environ.get(models["generator"]["api_key_env"]):
        errors.append(f"missing runtime secret: {models['generator']['api_key_env']}")
    if args.require_evaluated_api:
        for entry in models.get("evaluated", []):
            for key in (entry["api_key_env"],):
                if not os.environ.get(key):
                    errors.append(f"missing evaluator runtime setting: {key}")
    if mode == "full" and not (ROOT / "data/splits/final_source_ids.json").is_file():
        errors.append("full mode is locked until data/splits/final_source_ids.json is frozen")
    ensure_layout()
    if errors:
        raise SystemExit("Environment invalid:\n- " + "\n- ".join(errors))
    print(f"Environment valid for {mode}: expected_sources={expected_source_count(mode, cfg)} config_scope=frozen")


if __name__ == "__main__":
    main()
