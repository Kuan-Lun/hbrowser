"""Validate bounded reviewer attestations and their complete exact-tree aggregate."""

from __future__ import annotations

import json
from typing import Any

from .review_plan import (
    INTEGRATION_OBLIGATIONS,
    MAX_RESULT_BYTES,
    Plan,
    PlanError,
    canonical,
    digest,
    validate_plan,
    wire,
)

ATTESTATION_PROPERTIES = {
    "plan_digest": {"type": "string"},
    "shard_id": {"type": "string"},
    "unit_reviews": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "unit_id": {"type": "string"},
                "method": {"type": "string"},
                "related_paths": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["unit_id", "method", "related_paths"],
            "additionalProperties": False,
        },
    },
    "integration_reviews": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "obligation": {"type": "string"},
                "analysis": {"type": "string"},
                "source_paths": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["obligation", "analysis", "source_paths"],
            "additionalProperties": False,
        },
    },
    "contracts": {"type": "array", "items": {"type": "string"}},
}


def strict_json(data: bytes, *, maximum: int = MAX_RESULT_BYTES) -> Any:
    if type(maximum) is not int or not 1 <= maximum <= MAX_RESULT_BYTES:
        raise PlanError("Invalid review response wire budget")
    if len(data) > maximum:
        raise PlanError("Review result exceeds its bounded output contract")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise PlanError("Duplicate JSON review key")
            result[key] = value
        return result

    return json.loads(data, object_pairs_hook=unique)


def _text(value: object, maximum: int) -> bool:
    return (
        isinstance(value, str)
        and bool(value.strip())
        and len(value.encode()) <= maximum
    )


def validate_attestation(result: dict[str, Any], plan: Plan, index: int | None) -> None:
    limits = plan.response_limits(index)
    if len(wire(result)) > limits["wire_bytes"]:
        raise PlanError("Review result exceeds its response wire budget")
    if not _text(result.get("summary"), limits["summary_utf8_bytes"]):
        raise PlanError("Review summary must be nonempty and bounded")
    expected_id = "integration" if index is None else f"shard-{index:04d}"
    if (
        result.get("plan_digest") != plan.fingerprint
        or result.get("shard_id") != expected_id
    ):
        raise PlanError("Review attestation targets a different plan or shard")
    selected = [] if index is None else list(plan.shards[index])
    paths = set(plan.source_paths)
    reviews = result.get("unit_reviews")
    if not isinstance(reviews, list):
        raise PlanError("Missing unit coverage attestation")
    ids = []
    for review in reviews:
        if (
            not isinstance(review, dict)
            or set(review) != {"unit_id", "method", "related_paths"}
            or not _text(review["method"], limits["unit_method_utf8_bytes"])
            or not isinstance(review["related_paths"], list)
            or any(not isinstance(path, str) for path in review["related_paths"])
            or len(review["related_paths"]) != len(set(review["related_paths"]))
        ):
            raise PlanError("Malformed unit review method or related source")
        for path in review["related_paths"]:
            if path not in paths:
                raise PlanError(
                    f"Unknown unit related source {path[:256]!r}; "
                    "expected an exact candidate or declared historical Git path"
                )
        ids.append(review["unit_id"])
    if (
        any(not isinstance(key, str) for key in ids)
        or len(ids) != len(set(ids))
        or set(ids) != set(selected)
    ):
        raise PlanError("Missing, duplicate or wrong-unit coverage")
    unit_map = {unit.id: unit for unit in plan.units}
    for review in reviews:
        required = set(unit_map[review["unit_id"]].related_paths)
        if not required <= set(review["related_paths"]):
            raise PlanError(
                "Binary examination omitted declared generator/decoder/consumer sources"
            )
    integrations = result.get("integration_reviews")
    if not isinstance(integrations, list):
        raise PlanError("Missing integration coverage")
    obligations = []
    for review in integrations:
        if (
            not isinstance(review, dict)
            or set(review) != {"obligation", "analysis", "source_paths"}
            or not _text(review["analysis"], limits["integration_analysis_utf8_bytes"])
            or not isinstance(review["source_paths"], list)
            or not review["source_paths"]
            or any(not isinstance(path, str) for path in review["source_paths"])
            or len(review["source_paths"]) != len(set(review["source_paths"]))
        ):
            raise PlanError("Malformed cross-module review evidence")
        for path in review["source_paths"]:
            if path not in paths:
                raise PlanError(
                    f"Unknown integration source {path[:256]!r}; "
                    "expected an exact candidate or declared historical Git path"
                )
        obligations.append(review["obligation"])
    expected = (
        set(INTEGRATION_OBLIGATIONS)
        if index is None or len(plan.shards) == 1
        else set()
    )
    if (
        any(not isinstance(key, str) for key in obligations)
        or len(obligations) != len(set(obligations))
        or set(obligations) != expected
    ):
        raise PlanError("Incomplete integration obligations")
    contracts = result.get("contracts")
    if (
        not isinstance(contracts, list)
        or len(contracts) > limits["contracts_count"]
        or len(wire(contracts)) > limits["contracts_wire_bytes"]
        or any(not _text(value, limits["contract_utf8_bytes"]) for value in contracts)
    ):
        raise PlanError("Interface handoff summaries must be bounded")


def validate_aggregate(document: object, plan: Plan) -> None:
    if not isinstance(document, dict) or set(document) != {
        "plan",
        "plan_digest",
        "stages",
    }:
        raise PlanError("Malformed aggregate review evidence")
    validate_plan(document["plan"])
    if (
        canonical(document["plan"]) != canonical(plan.document())
        or document["plan_digest"] != plan.fingerprint
    ):
        raise PlanError("Aggregate manifest no longer matches the exact candidate")
    stages = document["stages"]
    expected_ids = [f"shard-{i:04d}" for i in range(len(plan.shards))]
    if len(plan.shards) > 1:
        expected_ids.append("integration")
    if not isinstance(stages, list) or len(stages) != len(expected_ids):
        raise PlanError("Missing aggregate review stage")
    for index, (stage, expected) in enumerate(zip(stages, expected_ids, strict=True)):
        if not isinstance(stage, dict) or set(stage) != {
            "id",
            "review",
            "review_sha256",
            "owned_tree_empty",
        }:
            raise PlanError("Malformed stage evidence")
        result = stage["review"]
        if stage["id"] != expected or stage["owned_tree_empty"] is not True:
            raise PlanError("Missing ordered stage or process cleanup evidence")
        if (
            not isinstance(result, dict)
            or result.get("verdict") != "pass"
            or result.get("findings") != []
        ):
            raise PlanError(
                "An incomplete or finding-bearing shard cannot be aggregated"
            )
        if stage["review_sha256"] != digest(canonical(result)):
            raise PlanError("Stage result hash changed")
        validate_attestation(result, plan, None if expected == "integration" else index)
