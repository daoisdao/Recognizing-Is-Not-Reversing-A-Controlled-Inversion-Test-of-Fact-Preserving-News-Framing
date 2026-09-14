# SPDX-FileCopyrightText: 2026 Yi Liu
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
import uuid
from urllib.parse import urlparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml


ROOT = Path(__file__).resolve().parents[2]
OPERATORS = ("lexical_affective", "agency_prominence", "salience_order")
STRENGTHS = ("low", "medium", "high")
DIRECTIONS = ("favorable", "unfavorable")


class FatalAPIError(RuntimeError):
    """Non-retryable authentication, authorization, or billing failure."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_configs() -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        load_yaml(ROOT / "code/configs" / "experiment.yaml"),
        load_yaml(ROOT / "code/configs" / "models.yaml"),
    )


def config_hash() -> str:
    digest = hashlib.sha256()
    for rel in (
        "configs/experiment.yaml",
        "configs/models.yaml",
        "prompts/C0_canonicalize_source.md",
        "prompts/P0_atomic_fact_extraction.md",
        "prompts/P1_contamination_generator.md",
        "prompts/P2_contamination_validator.md",
        "prompts/D0_detection.md",
        "prompts/R0_reconstruction.md",
        "prompts/R-CLEAN_clean_reconstruction.md",
    ):
        path = ROOT / "code" / rel
        if not path.exists():
            continue
        digest.update(rel.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def ensure_layout() -> None:
    generator_tag = generator_artifact_tag()
    for rel in (
        "data/raw", "data/canonical", "data/facts", "data/contaminated", "data/splits",
        f"data/results/raw/{generator_tag}", f"data/results/parsed/{generator_tag}", "data/results/metrics",
        "data/results/summaries", "data/results/tables", "data/results/figures",
        "code/logs/run_manifests", "code/paper",
    ):
        (ROOT / rel).mkdir(parents=True, exist_ok=True)
    for rel in ("code/logs/invalid_samples.jsonl", "code/logs/api_errors.jsonl"):
        path = ROOT / rel
        if not path.exists():
            path.touch()


def mode_from_args(parser: argparse.ArgumentParser) -> str:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pilot", action="store_true", help="Use exactly 10 pilot sources.")
    group.add_argument("--full", action="store_true", help="Use exactly 60 frozen final sources.")
    args, _ = parser.parse_known_args()
    return "pilot" if args.pilot else "full"


def add_mode_args(parser: argparse.ArgumentParser, *, resume: bool = True) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pilot", action="store_true")
    group.add_argument("--full", action="store_true")
    if resume:
        parser.add_argument("--resume", action="store_true")


def selected_mode(args: argparse.Namespace) -> str:
    return "pilot" if args.pilot else "full"


def expected_source_count(mode: str, cfg: dict[str, Any]) -> int:
    # CLI mode is ``full`` while the frozen config names that stratum
    # ``final_sources``.
    key = "pilot_sources" if mode == "pilot" else "final_sources"
    return int(cfg["sample"][key])


def jsonl_read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def jsonl_write_new(path: Path, rows: Iterable[dict[str, Any]], *, resume: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if resume:
            return
        raise FileExistsError(f"Refusing to overwrite {path}; use --resume or move it aside.")
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_unique_json(directory: Path, stem: str, value: Any) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)[:120]
    path = directory / f"{safe}__{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}__{uuid.uuid4().hex}.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def write_manifest(stage: str, mode: str, payload: dict[str, Any]) -> Path:
    cfg, _ = load_configs()
    manifest = {
        "stage": stage,
        "mode": mode,
        "timestamp": utc_now(),
        "prompt_version": cfg["project"]["prompt_version"],
        "config_hash": config_hash(),
        **payload,
    }
    return write_unique_json(ROOT / "code/logs" / "run_manifests", f"{stage}_{mode}", manifest)


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w’'-]+\b", text, flags=re.UNICODE))


def split_prompt(path: Path) -> tuple[str, str]:
    text = path.read_text(encoding="utf-8").strip()
    system_match = re.search(r"(?ms)^SYSTEM:\s*(.*?)\s*^USER:\s*(.*)$", text)
    if not system_match:
        raise ValueError(f"Prompt must contain SYSTEM: and USER: sections: {path}")
    return system_match.group(1).strip(), system_match.group(2).strip()


def render(template: str, values: dict[str, Any]) -> str:
    output = template
    for key, value in values.items():
        if not isinstance(value, str):
            value = json.dumps(value, ensure_ascii=False, indent=2)
        output = output.replace("{{" + key + "}}", value)
    missing = re.findall(r"{{[A-Z0-9_]+}}", output)
    if missing:
        raise ValueError(f"Unfilled prompt placeholders: {sorted(set(missing))}")
    return output


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        try:
            value = json.loads(cleaned, strict=False)
        except json.JSONDecodeError:
            start, end = cleaned.find("{"), cleaned.rfind("}")
            if start < 0 or end <= start:
                raise
            value = json.loads(cleaned[start:end + 1], strict=False)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def generator_artifact_tag() -> str:
    _, models = load_configs()
    family = str(models["generator"].get("family", "generator"))
    return re.sub(r"[^A-Za-z0-9]+", "", family).lower() or "generator"


@dataclass(frozen=True)
class GeneratorSettings:
    api_key: str
    model_id: str
    base_url: str
    endpoint_label: str
    temperature: float
    reasoning_mode: str
    timeout_seconds: int
    max_output_tokens: int


def generator_settings() -> GeneratorSettings:
    _, models = load_configs()
    gen = models["generator"]
    api_key = os.environ.get(gen["api_key_env"], "").strip()
    if not api_key:
        raise RuntimeError(f"Missing {gen['api_key_env']}; provide it only as an environment variable.")
    return GeneratorSettings(
        api_key=api_key,
        model_id=os.environ.get(gen["model_id_env"], gen["model_id"]).strip(),
        base_url=os.environ.get(gen["base_url_env"], gen["base_url"]).strip(),
        endpoint_label=gen["endpoint_label"],
        temperature=float(gen["temperature"]),
        reasoning_mode=str(gen["reasoning"]),
        timeout_seconds=int(gen.get("timeout_seconds", 120)),
        max_output_tokens=int(gen.get("max_output_tokens", 2500)),
    )


def _response_text(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        raise ValueError("API response has no choices")
    content = (choices[0].get("message") or {}).get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    raise ValueError("API response has no textual message content")


def call_generator(
    *, stage: str, item_id: str, system: str, user: str,
    expected_json: bool = True, attempt: int = 1,
) -> tuple[Any, dict[str, Any]]:
    ensure_layout()
    cfg, models = load_configs()
    settings = generator_settings()
    payload: dict[str, Any] = {
        "model": settings.model_id,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False,
        "temperature": settings.temperature,
    }
    if settings.reasoning_mode.lower() in {"off", "false", "disabled", "minimum", "min"}:
        # Ark exposes an explicit switch for thinking-capable models. Sending it
        # prevents provider-side automatic reasoning enablement in the main run.
        payload["thinking"] = {"type": "disabled"}
    request = urllib.request.Request(
        settings.base_url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {settings.api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    started = utc_now()
    raw_response: dict[str, Any] | None = None
    try:
        with urllib.request.urlopen(request, timeout=settings.timeout_seconds) as response:
            body = response.read().decode("utf-8")
            raw_response = json.loads(body)
            header_request_id = response.headers.get("x-request-id") or response.headers.get("request-id")
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        append_jsonl(ROOT / "code/logs" / "api_errors.jsonl", {
            "stage": stage, "item_id": item_id, "attempt": attempt, "timestamp": utc_now(),
            "requested_model_id": settings.model_id, "provider": settings.endpoint_label,
            "reasoning_mode": settings.reasoning_mode, "temperature": settings.temperature,
            "config_hash": config_hash(), "error_type": type(exc).__name__, "error": str(exc),
        })
        if isinstance(exc, urllib.error.HTTPError) and exc.code in (401, 402, 403):
            raise FatalAPIError(f"HTTP {exc.code}: API authorization or billing rejected the request") from exc
        raise

    # Save an exact relay response in an exclusive, unique path before parsing.
    raw_path = write_unique_json(ROOT / "data/results" / "raw" / generator_artifact_tag() / stage, f"{item_id}_a{attempt}", raw_response)
    returned_model = raw_response.get("model")
    accepted = set(models["generator"].get("accepted_returned_model_ids", [settings.model_id]))
    identity_valid = returned_model is None or returned_model in accepted
    usage = raw_response.get("usage") or {}
    metadata = {
        "requested_model_id": settings.model_id,
        "returned_model_id": returned_model,
        "provider": settings.endpoint_label,
        "request_id": raw_response.get("id") or header_request_id,
        "timestamp": started,
        "temperature": settings.temperature,
        "reasoning_mode": settings.reasoning_mode,
        "input_tokens": usage.get("prompt_tokens") or usage.get("input_tokens"),
        "output_tokens": usage.get("completion_tokens") or usage.get("output_tokens"),
        "prompt_version": cfg["project"]["prompt_version"],
        "config_hash": config_hash(),
        "raw_response_path": str(raw_path.relative_to(ROOT)),
        "identity_valid": identity_valid,
    }
    if not identity_valid:
        append_jsonl(ROOT / "code/logs" / "invalid_samples.jsonl", {
            **metadata, "stage": stage, "item_id": item_id, "attempt": attempt,
            "failure_reason": "unexpected_returned_model_identity",
        })
        raise RuntimeError(f"Unexpected returned model identity {returned_model!r}; expected {sorted(accepted)!r}")
    text = _response_text(raw_response)
    parsed: Any = parse_json_object(text) if expected_json else text.strip()
    parsed_path = write_unique_json(
        ROOT / "data/results" / "parsed" / generator_artifact_tag() / stage,
        f"{item_id}_a{attempt}", {"output": parsed, "metadata": metadata},
    )
    metadata["parsed_response_path"] = str(parsed_path.relative_to(ROOT))
    return parsed, metadata


def retry_generator(*, max_attempts: int = 3, **kwargs: Any) -> tuple[Any, dict[str, Any], int]:
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            value, metadata = call_generator(attempt=attempt, **kwargs)
            return value, metadata, attempt
        except Exception as exc:  # failure is logged by caller with condition context
            last_error = exc
            if attempt < max_attempts:
                time.sleep(min(2 ** (attempt - 1), 4))
    assert last_error is not None
    raise last_error


@dataclass(frozen=True)
class EvaluatedSettings:
    family: str
    api_key: str
    model_id: str
    base_url: str
    endpoint_label: str
    temperature: float
    reasoning_mode: str
    timeout_seconds: int
    max_output_tokens: int
    accepted_returned_model_ids: tuple[str, ...]


def evaluated_settings(family: str, *, reasoning_mode: str = "off",
                       max_output_tokens: int | None = None) -> EvaluatedSettings:
    """Resolve one frozen evaluated-model configuration from runtime secrets."""
    _, models = load_configs()
    entry = next((row for row in models.get("evaluated", []) if row.get("family") == family), None)
    if not entry:
        raise ValueError(f"Unknown evaluated family {family!r}")
    prefix = family.upper()
    api_key = os.environ.get(entry["api_key_env"], "").strip()
    model_id = os.environ.get(entry["model_id_env"], entry.get("model_id", "")).strip()
    base_url = os.environ.get(entry["base_url_env"], entry.get("base_url", "")).strip()
    missing = [name for name, value in ((entry["api_key_env"], api_key),
                                         (entry["model_id_env"], model_id),
                                         (entry["base_url_env"], base_url)) if not value]
    if missing:
        raise RuntimeError(f"Missing evaluator runtime settings: {', '.join(missing)}")
    endpoint_label = os.environ.get(
        f"{prefix}_ENDPOINT_LABEL",
        entry.get("endpoint_label") or urlparse(base_url).netloc or family,
    ).strip()
    accepted = tuple(entry.get("accepted_returned_model_ids") or [model_id])
    temperature = float(entry.get("temperature", 0.0))
    if family == "kimi":
        # Kimi K2.6 fixes non-thinking sampling at 0.6 and thinking at 1.0.
        temperature = 0.6 if reasoning_mode.lower() not in {"on", "enabled", "high", "thinking"} else 1.0
    return EvaluatedSettings(
        family=family,
        api_key=api_key,
        model_id=model_id,
        base_url=base_url,
        endpoint_label=endpoint_label,
        temperature=temperature,
        reasoning_mode=reasoning_mode,
        timeout_seconds=int(entry.get("timeout_seconds", 120)),
        max_output_tokens=int(max_output_tokens or entry.get("max_output_tokens", 1000)),
        accepted_returned_model_ids=accepted,
    )


def _evaluated_payload(settings: EvaluatedSettings, system: str, user: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": settings.model_id,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False,
        "temperature": settings.temperature,
        "max_tokens": settings.max_output_tokens,
    }
    mode = settings.reasoning_mode.lower()
    # Never silently enable thinking in the direct condition.
    if settings.family == "qwen":
        payload["enable_thinking"] = mode in {"on", "enabled", "high", "thinking"}
    elif settings.family in {"deepseek", "kimi"}:
        payload["thinking"] = {"type": "enabled" if mode in {"on", "enabled", "high", "thinking"} else "disabled"}
    return payload


def call_evaluated(*, family: str, stage: str, item_id: str, system: str, user: str,
                   expected_json: bool, reasoning_mode: str = "off", attempt: int = 1,
                   max_output_tokens: int | None = None) -> tuple[Any, dict[str, Any]]:
    """Call exactly the requested evaluated model and archive raw/parsed output."""
    ensure_layout()
    cfg, _ = load_configs()
    settings = evaluated_settings(family, reasoning_mode=reasoning_mode,
                                  max_output_tokens=max_output_tokens)
    payload = _evaluated_payload(settings, system, user)
    request = urllib.request.Request(
        settings.base_url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {settings.api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    started = utc_now()
    try:
        with urllib.request.urlopen(request, timeout=settings.timeout_seconds) as response:
            body = response.read().decode("utf-8")
            raw_response = json.loads(body)
            header_request_id = response.headers.get("x-request-id") or response.headers.get("request-id")
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        detail = str(exc)
        if isinstance(exc, urllib.error.HTTPError):
            try:
                detail += " response=" + exc.read().decode("utf-8", errors="replace")[:2000]
            except Exception:
                pass
        detail = detail.replace(settings.api_key, "[REDACTED]")
        append_jsonl(ROOT / "code/logs" / "api_errors.jsonl", {
            "stage": stage, "item_id": item_id, "attempt": attempt, "timestamp": utc_now(),
            "family": family, "requested_model_id": settings.model_id,
            "provider": settings.endpoint_label, "reasoning_mode": reasoning_mode,
            "temperature": settings.temperature, "config_hash": config_hash(),
            "error_type": type(exc).__name__, "error": detail,
        })
        if isinstance(exc, urllib.error.HTTPError) and exc.code in (401, 402, 403):
            raise FatalAPIError(f"{family} HTTP {exc.code}: authorization or billing rejected") from exc
        raise

    raw_path = write_unique_json(
        ROOT / "data/results" / "raw" / "evaluated" / family / stage,
        f"{item_id}_a{attempt}_{reasoning_mode}", raw_response,
    )
    returned_model = raw_response.get("model")
    identity_valid = returned_model is None or returned_model in settings.accepted_returned_model_ids
    usage = raw_response.get("usage") or {}
    metadata = {
        "family": family,
        "requested_model_id": settings.model_id,
        "returned_model_id": returned_model,
        "provider": settings.endpoint_label,
        "request_id": raw_response.get("id") or header_request_id,
        "timestamp": started,
        "temperature": settings.temperature,
        "reasoning_mode": reasoning_mode,
        "input_tokens": usage.get("prompt_tokens") or usage.get("input_tokens"),
        "output_tokens": usage.get("completion_tokens") or usage.get("output_tokens"),
        "reasoning_tokens": usage.get("reasoning_tokens"),
        "prompt_version": cfg["project"]["prompt_version"],
        "config_hash": config_hash(),
        "raw_response_path": str(raw_path.relative_to(ROOT)),
        "identity_valid": identity_valid,
    }
    if not identity_valid:
        append_jsonl(ROOT / "code/logs" / "invalid_samples.jsonl", {
            **metadata, "stage": stage, "item_id": item_id, "attempt": attempt,
            "failure_reason": "unexpected_returned_model_identity",
        })
        raise RuntimeError(
            f"Unexpected {family} returned model {returned_model!r}; "
            f"expected {sorted(settings.accepted_returned_model_ids)!r}"
        )
    text = _response_text(raw_response)
    parsed: Any = parse_json_object(text) if expected_json else text.strip()
    parsed_path = write_unique_json(
        ROOT / "data/results" / "parsed" / "evaluated" / family / stage,
        f"{item_id}_a{attempt}_{reasoning_mode}", {"output": parsed, "metadata": metadata},
    )
    metadata["parsed_response_path"] = str(parsed_path.relative_to(ROOT))
    return parsed, metadata


def retry_evaluated(*, max_attempts: int = 2, **kwargs: Any) -> tuple[Any, dict[str, Any], int]:
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            value, metadata = call_evaluated(attempt=attempt, **kwargs)
            return value, metadata, attempt
        except FatalAPIError:
            raise
        except Exception as exc:
            last_error = exc
            if attempt < max_attempts:
                time.sleep(min(2 ** (attempt - 1), 4))
    assert last_error is not None
    raise last_error


# Backward-compatible names for the canonical/fact scripts and prior manifests.
# They resolve to the configured generator, which is now the GLM5.2 pollution adapter.
HunyuanSettings = GeneratorSettings
hunyuan_settings = generator_settings
call_hunyuan = call_generator
retry_hunyuan = retry_generator


def validate_source_schema(row: dict[str, Any]) -> list[str]:
    required = ("source_id", "title", "provenance_url", "revision_id", "publication_date",
                "license", "raw_article_text", "domain", "target_actor")
    return [key for key in required if not row.get(key)]


def validate_facts(source_id: str, value: dict[str, Any], minimum: int, maximum: int) -> list[dict[str, Any]]:
    facts = value.get("facts")
    if not isinstance(facts, list) or not minimum <= len(facts) <= maximum:
        raise ValueError(f"Expected {minimum}-{maximum} facts, got {len(facts) if isinstance(facts, list) else 'non-list'}")
    normalized: list[dict[str, Any]] = []
    for index, fact in enumerate(facts, 1):
        if not isinstance(fact, dict) or not str(fact.get("proposition", "")).strip():
            raise ValueError(f"Malformed fact {index}")
        normalized.append({
            "fact_id": f"{source_id}_F{index:02d}",
            "proposition": str(fact["proposition"]).strip(),
            "entities": list(fact.get("entities") or []),
            "numbers_dates": list(fact.get("numbers_dates") or []),
        })
    return normalized


def condition_rows(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_index, source in enumerate(sorted(sources, key=lambda x: x["source_id"])):
        for operator_index, operator in enumerate(OPERATORS):
            direction = DIRECTIONS[(source_index + operator_index) % 2]
            for strength in STRENGTHS:
                rows.append({
                    "sample_id": f"{source['source_id']}__{operator}__{strength}",
                    "source_id": source["source_id"],
                    "operator": operator,
                    "strength": strength,
                    "direction": direction,
                })
    return rows


def check_generator_output(value: dict[str, Any], condition: dict[str, Any], source_text: str) -> list[str]:
    failures: list[str] = []
    article = value.get("transformed_article")
    edits = value.get("edit_plan")
    self_check = value.get("self_check")
    if not isinstance(article, str) or not article.strip():
        failures.append("missing_transformed_article")
    if not isinstance(edits, list) or not edits:
        failures.append("missing_edit_plan")
    if not isinstance(self_check, dict):
        failures.append("missing_self_check")
    else:
        expected = {
            "all_facts_preserved": True, "new_facts_added": False,
            "facts_removed": False, "cross_operator_changes": False,
        }
        for key, expected_value in expected.items():
            if self_check.get(key) is not expected_value:
                failures.append(f"self_check_{key}")
    if isinstance(edits, list):
        for edit in edits:
            if not isinstance(edit, dict) or edit.get("operator") != condition["operator"]:
                failures.append("edit_plan_operator_mismatch")
                break
            before = str(edit.get("before", "")).strip()
            after = str(edit.get("after", "")).strip()
            if not before or not after or before == after:
                failures.append("edit_plan_noop_or_empty")
            elif before not in source_text:
                failures.append("edit_before_not_in_source")
            elif after not in article:
                failures.append("edit_after_not_in_output")
        expected_counts = {"low": 2, "medium": 4, "high": 6} if condition["operator"] == "lexical_affective" else (
            {"low": 1, "medium": 2, "high": 3} if condition["operator"] == "agency_prominence" else None
        )
        if expected_counts and len(edits) != expected_counts[condition["strength"]]:
            failures.append("edit_budget_mismatch")
    if isinstance(article, str):
        ratio = word_count(article) / max(word_count(source_text), 1)
        if ratio < 0.90 or ratio > 1.10:
            failures.append("length_outside_10_percent")
    return sorted(set(failures))


def validator_passed(value: dict[str, Any], fact_ids: list[str]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    fp = value.get("fact_preservation")
    if value.get("pass") is not True:
        failures.append("validator_pass_false")
    def fact_value(fact_id: str) -> Any:
        if not isinstance(fp, dict):
            return None
        if fact_id in fp:
            return fp[fact_id]
        # P2's frozen output example uses local IDs (F01), while stored inventories
        # use globally unique IDs (S001_F01). A validator sees one source per call,
        # so suffix mapping is unambiguous and changes no substantive judgement.
        match = re.search(r"(F\d+)$", fact_id)
        return fp.get(match.group(1)) if match else None

    if not isinstance(fp, dict) or any(fact_value(fid) is not True for fid in fact_ids):
        failures.append("fact_not_preserved")
    if value.get("unsupported_new_facts") not in ([], None):
        failures.append("unsupported_new_facts")
    if value.get("altered_facts") not in ([], None):
        failures.append("altered_facts")
    for key in ("operator_valid", "direction_valid", "strength_plausible", "professionally_plausible"):
        if value.get(key) is not True:
            failures.append(key)
    if value.get("cross_operator_leakage") is not False:
        failures.append("cross_operator_leakage")
    return not failures, failures
