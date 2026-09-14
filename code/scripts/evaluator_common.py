# SPDX-FileCopyrightText: 2026 Yi Liu
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from pipeline_common import (FatalAPIError, ROOT, append_jsonl, expected_source_count,
                             jsonl_read, load_configs, write_manifest)

EVALUATED_FAMILIES = ("qwen", "deepseek", "kimi")


def add_evaluator_args(parser: argparse.ArgumentParser, *, use_candidates: bool = False) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pilot", action="store_true")
    group.add_argument("--full", action="store_true")
    parser.add_argument("--family", choices=EVALUATED_FAMILIES, required=True)
    parser.add_argument("--workers", type=int, default=1, choices=range(1, 9))
    parser.add_argument("--resume", action="store_true")
    if use_candidates:
        parser.add_argument(
            "--use-candidates", action="store_true",
            help="Use manually approved GLM5.2 candidates when no validated file exists.",
        )


def selected_mode(args: argparse.Namespace) -> str:
    return "pilot" if args.pilot else "full"


def contaminated_path(mode: str, *, use_candidates: bool) -> Path:
    _, models = load_configs()
    tag = "".join(ch for ch in str(models["generator"].get("family", "generator")) if ch.isalnum()).lower()
    validated = ROOT / "data" / "contaminated" / f"{mode}_{tag}.jsonl"
    candidates = ROOT / "data" / "contaminated" / f"{mode}_candidates_{tag}.jsonl"
    if validated.exists():
        return validated
    if use_candidates and candidates.exists():
        return candidates
    raise FileNotFoundError(
        f"No approved contamination file at {validated}; use --use-candidates only after human approval."
    )


def load_canonical(mode: str) -> list[dict[str, Any]]:
    path = ROOT / "data" / "canonical" / f"{mode}.jsonl"
    rows = jsonl_read(path)
    if not rows:
        raise FileNotFoundError(path)
    return rows


def load_contaminated(mode: str, *, use_candidates: bool) -> list[dict[str, Any]]:
    rows = jsonl_read(contaminated_path(mode, use_candidates=use_candidates))
    expected = expected_source_count(mode, load_configs()[0]) * 9
    ids = [row.get("sample_id") for row in rows]
    if len(rows) != expected or len(set(ids)) != expected or any(not value for value in ids):
        raise ValueError(f"Expected {expected} unique contamination rows, found {len(rows)}")
    return rows


def output_path(family: str, stage: str, mode: str, reasoning_mode: str = "direct") -> Path:
    return ROOT / "data/results" / "evaluated" / family / f"{stage}_{mode}_{reasoning_mode}.jsonl"


def existing_ids(path: Path) -> set[str]:
    return {str(row.get("item_id") or row.get("sample_id")) for row in jsonl_read(path) if row.get("valid", True)}


def run_parallel_jsonl(*, family: str, stage: str, mode: str, items: list[dict[str, Any]],
                       worker: Callable[[dict[str, Any]], dict[str, Any]], workers: int,
                       resume: bool, reasoning_mode: str = "direct",
                       expected_total: int | None = None) -> tuple[int, list[dict[str, Any]]]:
    path = output_path(family, stage, mode, reasoning_mode)
    if path.exists() and not resume:
        raise SystemExit(f"Refusing to overwrite {path}; use --resume")
    done = existing_ids(path) if resume else set()
    pending = [item for item in items if str(item.get("item_id") or item.get("sample_id")) not in done]
    failures: list[dict[str, Any]] = []
    lock = threading.Lock()
    fatal = threading.Event()

    def wrapped(item: dict[str, Any]) -> dict[str, Any]:
        if fatal.is_set():
            raise FatalAPIError("batch cancelled after a non-retryable API failure")
        return worker(item)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(wrapped, item): item for item in pending}
        for future in as_completed(futures):
            item = futures[future]
            item_id = str(item.get("item_id") or item.get("sample_id"))
            try:
                row = future.result()
                row.setdefault("item_id", item_id)
                row.setdefault("valid", True)
                with lock:
                    from pipeline_common import append_jsonl as _append
                    _append(path, row)
            except FatalAPIError:
                fatal.set()
                raise
            except Exception as exc:
                failure = {"item_id": item_id, "family": family, "stage": stage,
                           "mode": mode, "reasoning_mode": reasoning_mode,
                           "error": f"{type(exc).__name__}: {exc}"}
                failures.append(failure)
                with lock:
                    append_jsonl(ROOT / "code/logs" / "evaluator_failures.jsonl", failure)
    completed = len(existing_ids(path))
    expected = int(expected_total if expected_total is not None else len(items))
    pending_count = max(0, expected - completed)
    manifest = write_manifest(
        f"{stage}_{family}_{reasoning_mode}", mode,
        {"family": family, "output": str(path.relative_to(ROOT)),
         "expected": expected, "scheduled": len(items), "completed": completed,
         "pending": pending_count,
         "failed": len(failures), "workers": workers, "reasoning_mode": reasoning_mode},
    )
    if completed != expected:
        print(f"Completed {completed}/{expected} for {family} {stage}; manifest={manifest}")
    else:
        print(f"Completed {completed}/{expected} for {family} {stage}; output={path}; manifest={manifest}")
    return completed, failures
