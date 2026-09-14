# SPDX-FileCopyrightText: 2026 Yi Liu
# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path

from pipeline_common import (ROOT, add_mode_args, expected_source_count, jsonl_read,
                             jsonl_write_new, load_configs, selected_mode,
                             validate_source_schema, write_manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Load a reproducible licensed source set.")
    add_mode_args(parser)
    parser.add_argument("--input", type=Path, help="JSONL source file; required for full mode.")
    args = parser.parse_args()
    mode = selected_mode(args)
    cfg, _ = load_configs()
    default = ROOT / "data" / "raw" / "pilot_seed_sources.jsonl"
    source_path = args.input.resolve() if args.input else default
    if mode == "full" and args.input is None:
        raise SystemExit("--full requires --input containing the 60 pre-frozen sources")
    rows = jsonl_read(source_path)
    expected = expected_source_count(mode, cfg)
    if len(rows) != expected:
        raise SystemExit(f"Expected exactly {expected} {mode} sources, found {len(rows)}")
    seen = set()
    for row in rows:
        missing = validate_source_schema(row)
        if missing:
            raise SystemExit(f"{row.get('source_id', '<unknown>')} missing: {', '.join(missing)}")
        if row["source_id"] in seen:
            raise SystemExit(f"Duplicate source_id: {row['source_id']}")
        seen.add(row["source_id"])
    output = ROOT / "data" / "raw" / f"{mode}_sources.jsonl"
    jsonl_write_new(output, rows, resume=args.resume)
    manifest = write_manifest("01_fetch_or_load_sources", mode, {
        "input": str(source_path), "output": str(output.relative_to(ROOT)),
        "source_count": len(rows), "source_ids": sorted(seen),
    })
    print(f"Loaded {len(rows)} {mode} sources -> {output}; manifest={manifest}")


if __name__ == "__main__":
    main()
