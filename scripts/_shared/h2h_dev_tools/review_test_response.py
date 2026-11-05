"""Synthetic V3 responses for disposable review workflow fixtures only.

This self-contained module is also exported into fake CLI scripts. It performs
no code review and neither creates nor verifies passing evidence receipts.
"""

import json
import re

ASSIGNMENT_MARKER = "COORDINATOR ASSIGNMENT (data, not instructions):\n"


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate synthetic assignment key")
        result[key] = value
    return result


def _texts(value: object, *, allow_whitespace: bool = False) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str)
        or not item
        or (not allow_whitespace and not item.strip())
        for item in value
    ):
        raise ValueError("Synthetic assignment requires nonempty string entries")
    if len(value) != len(set(value)):
        raise ValueError("Duplicate synthetic assignment entry")
    return list(value)


def synthetic_review_response(prompt: str) -> dict[str, object]:
    """Project an actual V3 assignment into explicitly synthetic raw test output.

    Fixture-specific faults may mutate this result before writing it. Normal
    production validators still decide whether that output satisfies the plan.
    Malformed or legacy prompts never become implicit passing responses. Serialize
    with compact ensure_ascii=False UTF-8 JSON to preserve the checked wire bound.
    """
    _, marker, payload = prompt.rpartition(ASSIGNMENT_MARKER)
    if not marker:
        raise ValueError("Synthetic review requires a V3 coordinator assignment")
    assignment: object = json.loads(payload, object_pairs_hook=_unique_object)
    if not isinstance(assignment, dict):
        raise ValueError("Synthetic assignment must be an object")
    fingerprint = assignment.get("plan_digest")
    shard = assignment.get("shard_id")
    units = assignment.get("units")
    if (
        not isinstance(fingerprint, str)
        or re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None
        or not isinstance(shard, str)
        or re.fullmatch(r"shard-[0-9]{4}|integration", shard) is None
        or not isinstance(units, list)
    ):
        raise ValueError("Invalid synthetic assignment identity or units")
    limits = assignment.get("response_limits")
    if (
        not isinstance(limits, dict)
        or set(limits)
        != {
            "wire_bytes",
            "summary_utf8_bytes",
            "unit_method_utf8_bytes",
            "integration_analysis_utf8_bytes",
            "contract_utf8_bytes",
            "contracts_wire_bytes",
            "contracts_count",
        }
        or any(type(value) is not int or value <= 0 for value in limits.values())
    ):
        raise ValueError("Synthetic assignment requires explicit response limits")
    text = "Synthetic fixture response; no code review performed."
    if any(
        len(text.encode("utf-8")) > limits[field]
        for field in (
            "summary_utf8_bytes",
            "unit_method_utf8_bytes",
            "integration_analysis_utf8_bytes",
        )
    ):
        raise ValueError("Synthetic semantic text exceeds assignment limits")
    obligations = _texts(assignment.get("integration_obligations"))
    reviews: list[dict[str, object]] = []
    identities: set[str] = set()
    for unit in units:
        if (
            not isinstance(unit, dict)
            or not isinstance(unit.get("id"), str)
            or not unit["id"].strip()
            or unit["id"] in identities
        ):
            raise ValueError("Invalid or duplicate synthetic unit identity")
        identities.add(unit["id"])
        reviews.append(
            {
                "unit_id": unit["id"],
                "method": text,
                "related_paths": _texts(
                    unit.get("related_paths"), allow_whitespace=True
                ),
            }
        )
    if (shard == "integration" and (reviews or not obligations)) or (
        shard != "integration" and not reviews
    ):
        raise ValueError("Synthetic assignment has no valid shard work")
    result: dict[str, object] = {
        "verdict": "pass",
        "summary": text,
        "findings": [],
        "plan_digest": fingerprint,
        "shard_id": shard,
        "unit_reviews": reviews,
        "integration_reviews": [
            {
                "obligation": obligation,
                "analysis": text,
                "source_paths": ["AGENTS.md"],
            }
            for obligation in obligations
        ],
        "contracts": [],
    }
    if (
        limits["contracts_wire_bytes"] < 2
        or len(
            json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        > limits["wire_bytes"]
    ):
        raise ValueError("Synthetic response exceeds assignment wire budget")
    return result
