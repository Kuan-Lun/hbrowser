"""Deterministic review units and bounded, non-executing artifact examination.

Coverage records are reviewer attestations. Hashes and ranges establish what was
assigned; they do not prove that a model understood or even read those bytes.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import tarfile
import time
import tomllib
from dataclasses import asdict, dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Never

from . import review_execution

UNIT_BYTES = 64 * 1024
SHARD_BYTES = 384 * 1024
SHARD_UNITS = 32
BLOB_BATCH_BYTES = 8 * 1024 * 1024
MAX_BLOB_BYTES = 64 * 1024 * 1024
MAX_EXPANDED_BYTES = 128 * 1024 * 1024
MAX_MEMBERS = 4096
MAX_GIT_METADATA_BYTES = 2 * 1024 * 1024
MAX_INVENTORY_ENTRIES = 4096
MAX_TOTAL_BLOB_BYTES = 64 * 1024 * 1024
MAX_TOTAL_EXPANDED_BYTES = 128 * 1024 * 1024
MAX_TOTAL_VIEW_BYTES = 32 * 1024 * 1024
MAX_REVIEW_UNITS = 4096
MAX_UNIT_METADATA_BYTES = 8 * 1024 * 1024
MAX_SAMPLED_JSON_BYTES = 16 * 1024 * 1024
MAX_JSON_SLOTS = 1_000_000
MAX_JSON_DEPTH = 64
MAX_INTEGRATION_BYTES = 512 * 1024
MAX_PROMPT_BYTES = 640 * 1024
MAX_HANDOFF_BYTES = 8 * 1024 + 64
MIN_HANDOFF_BYTES = 1028
MAX_RESULT_BYTES = 1024 * 1024
SHARD_RESULT_BYTES = 512 * 1024
SUMMARY_UTF8_BYTES = 8 * 1024
UNIT_METHOD_UTF8_BYTES = 2 * 1024
INTEGRATION_ANALYSIS_UTF8_BYTES = 8 * 1024
CONTRACT_UTF8_BYTES = 1024
MAX_CONTRACTS = 8
INTEGRATION_OBLIGATIONS = (
    "public-interfaces-and-callers",
    "state-transactions-fencing-and-recovery",
    "resource-ownership-and-cleanup",
    "schema-generated-artifacts-and-source-correspondence",
    "configuration-packaging-and-deployment",
    "tests-verification-and-cross-shard-contracts",
)


class PlanError(RuntimeError):
    """An exact and safely inspectable assignment cannot be established."""


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=True, separators=(",", ":")
    ).encode()


def wire(value: object) -> bytes:
    """The exact compact UTF-8 representation used in model assignments."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def response_limits(
    *, integration: bool, contracts_wire_bytes: int = MAX_HANDOFF_BYTES
) -> dict[str, int]:
    """One wire contract shared by planning, assignments and validation.

    Text limits measure decoded UTF-8 bytes; wire limits include JSON escaping,
    keys, arrays and whitespace. Integration output has no downstream handoff,
    so its contracts retain the fixed maximum rather than a shard's smaller cap.
    """
    return {
        "wire_bytes": MAX_RESULT_BYTES if integration else SHARD_RESULT_BYTES,
        "summary_utf8_bytes": SUMMARY_UTF8_BYTES,
        "unit_method_utf8_bytes": UNIT_METHOD_UTF8_BYTES,
        "integration_analysis_utf8_bytes": INTEGRATION_ANALYSIS_UTF8_BYTES,
        "contract_utf8_bytes": CONTRACT_UTF8_BYTES,
        "contracts_wire_bytes": contracts_wire_bytes,
        "contracts_count": MAX_CONTRACTS,
    }


@dataclass(frozen=True)
class Unit:
    id: str
    path: str
    blob: str
    mode: str
    origin: str
    member: str | None
    kind: str
    start: int
    end: int
    size: int
    sha256: str
    examination: str
    related_paths: tuple[str, ...]


def response_capacity(
    units: tuple[Unit, ...],
    *,
    source_paths: tuple[str, ...],
    shard_id: str,
    integration: bool,
    contracts_wire_bytes: int = MAX_HANDOFF_BYTES,
) -> int:
    """Reserve complete mandatory coverage and bounded semantic text.

    A decoded UTF-8 byte costs at most six bytes in compact UTF-8 JSON (a control
    character escape). Exact keys, unit identities and required paths are counted
    from their actual wire representation. Each integration obligation can cite
    every known source once. Optional extra unit references, findings and pretty
    printing share the wire cap; they are not promised additional reserved space.
    """
    result = {
        "verdict": "pass",
        "summary": "",
        "findings": [],
        "plan_digest": "0" * 64,
        "shard_id": shard_id,
        "unit_reviews": [
            {"unit_id": unit.id, "method": "", "related_paths": unit.related_paths}
            for unit in units
        ],
        "integration_reviews": [
            {"obligation": obligation, "analysis": "", "source_paths": source_paths}
            for obligation in INTEGRATION_OBLIGATIONS
        ]
        if integration
        else [],
        "contracts": [],
    }
    text_bytes = SUMMARY_UTF8_BYTES + len(units) * UNIT_METHOD_UTF8_BYTES
    if integration:
        text_bytes += len(INTEGRATION_OBLIGATIONS) * INTEGRATION_ANALYSIS_UTF8_BYTES
    return len(wire(result)) + 6 * text_bytes + contracts_wire_bytes - 2


@dataclass(frozen=True)
class Plan:
    target: dict[str, Any]
    scope: dict[str, Any]
    inventory: tuple[dict[str, Any], ...]
    units: tuple[Unit, ...]
    shards: tuple[tuple[str, ...], ...]
    payloads: dict[str, str]
    historical_related: tuple[dict[str, Any], ...] = ()

    def document(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "coverage": self.scope,
            "inventory": list(self.inventory),
            "historical_related": list(self.historical_related),
            "units": [asdict(unit) for unit in self.units],
            "shards": [list(shard) for shard in self.shards],
            "integration_obligations": list(INTEGRATION_OBLIGATIONS),
            "limits": {
                "unit_bytes": UNIT_BYTES,
                "shard_bytes": SHARD_BYTES,
                "shard_units": SHARD_UNITS,
                "blob_bytes": MAX_BLOB_BYTES,
                "expanded_bytes": MAX_EXPANDED_BYTES,
                "members": MAX_MEMBERS,
                "git_metadata_bytes": MAX_GIT_METADATA_BYTES,
                "inventory_entries": MAX_INVENTORY_ENTRIES,
                "total_blob_bytes": MAX_TOTAL_BLOB_BYTES,
                "total_expanded_bytes": MAX_TOTAL_EXPANDED_BYTES,
                "total_view_bytes": MAX_TOTAL_VIEW_BYTES,
                "review_units": MAX_REVIEW_UNITS,
                "unit_metadata_bytes": MAX_UNIT_METADATA_BYTES,
                "sampled_json_bytes": MAX_SAMPLED_JSON_BYTES,
                "json_slots": MAX_JSON_SLOTS,
                "json_depth": MAX_JSON_DEPTH,
                "handoff_wire_bytes": self.handoff_budget,
                "integration_bytes": MAX_INTEGRATION_BYTES,
                "prompt_bytes": MAX_PROMPT_BYTES,
                "shard_response": response_limits(
                    integration=False, contracts_wire_bytes=self.handoff_budget
                ),
                "integration_response": response_limits(integration=True),
            },
        }

    @property
    def fingerprint(self) -> str:
        return digest(canonical(self.document()))

    def assignment(self, index: int) -> dict[str, Any]:
        ids = self.shards[index]
        selected = {unit.id: unit for unit in self.units}
        return {
            "plan_digest": self.fingerprint,
            "shard_id": f"shard-{index:04d}",
            "response_limits": self.response_limits(index),
            "historical_related": list(self.historical_related),
            "units": [
                {**asdict(selected[key]), "content": self.payloads[key]} for key in ids
            ],
        }

    def response_limits(self, index: int | None) -> dict[str, int]:
        return response_limits(
            integration=index is None or len(self.shards) == 1,
            contracts_wire_bytes=(
                MAX_HANDOFF_BYTES if index is None else self.handoff_budget
            ),
        )

    @property
    def source_paths(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {entry["path"] for entry in (*self.inventory, *self.historical_related)}
                | {unit.path for unit in self.units}
            )
        )

    def response_capacity(self, index: int | None) -> int:
        selected = set() if index is None else set(self.shards[index])
        return response_capacity(
            tuple(unit for unit in self.units if unit.id in selected),
            source_paths=self.source_paths,
            shard_id="integration" if index is None else f"shard-{index:04d}",
            integration=index is None or len(self.shards) == 1,
            contracts_wire_bytes=self.response_limits(index)["contracts_wire_bytes"],
        )

    def validate_response_capacity(self) -> None:
        stages: list[int | None] = list(range(len(self.shards)))
        if len(self.shards) > 1:
            stages.append(None)
        for index in stages:
            if (
                self.response_capacity(index)
                > self.response_limits(index)["wire_bytes"]
            ):
                raise PlanError(
                    "Complete review coverage and semantic text exceed "
                    "the response wire budget before review"
                )

    def _integration(
        self, contracts: list[list[str]], fingerprint: str
    ) -> dict[str, Any]:
        return {
            "plan_digest": fingerprint,
            "shard_id": "integration",
            "response_limits": self.response_limits(None),
            "units": [],
            "integration_obligations": list(INTEGRATION_OBLIGATIONS),
            "handoffs": [
                {
                    "shard_id": f"shard-{index:04d}",
                    "unit_ids": list(ids),
                    "contracts": summaries,
                }
                for index, (ids, summaries) in enumerate(
                    zip(self.shards, contracts, strict=True)
                )
            ],
            "inventory": list(self.inventory),
            "historical_related": list(self.historical_related),
            "coverage_map": [
                {
                    key: getattr(unit, key)
                    for key in (
                        "id",
                        "path",
                        "origin",
                        "member",
                        "kind",
                        "start",
                        "end",
                    )
                }
                for unit in self.units
            ],
            "integration_instruction": (
                "Read raw source using this exact "
                "id/path/origin/member/range map across every reported "
                "contract and all shard boundaries. Read deleted source "
                "and its related contracts from their historical origin. "
                "Reconcile conflicting assumptions, large-file splits, "
                "generator/validator/runtime, fixtures/tests and "
                "callers. This stage attests interactions, not merely "
                "summary consistency."
            ),
        }

    def _integration_base_bytes(self) -> int:
        # A fixed-width digest breaks the budget/document/fingerprint dependency.
        return len(wire(self._integration([[] for _ in self.shards], "0" * 64)))

    @property
    def handoff_budget(self) -> int:
        if len(self.shards) == 1:
            return MAX_HANDOFF_BYTES
        if not self.shards:
            raise PlanError("Missing review shards")
        available = MAX_INTEGRATION_BYTES - self._integration_base_bytes()
        # Each empty array already contributes two bytes to the base assignment.
        budget = min(MAX_HANDOFF_BYTES, 2 + available // len(self.shards))
        if budget < MIN_HANDOFF_BYTES:
            raise PlanError(
                "Exact inventory/coverage map leaves insufficient "
                "bounded integration handoff capacity"
            )
        return budget

    @property
    def integration_max_bytes(self) -> int:
        return self._integration_base_bytes() + len(self.shards) * (
            self.handoff_budget - 2
        )

    def integration_assignment(self, contracts: list[list[str]]) -> dict[str, Any]:
        if len(contracts) != len(self.shards) or any(
            len(wire(summaries)) > self.handoff_budget for summaries in contracts
        ):
            raise PlanError(
                "Integration handoff exceeds the preflight per-shard wire budget"
            )
        assignment = self._integration(contracts, self.fingerprint)
        if len(wire(assignment)) > MAX_INTEGRATION_BYTES:
            raise PlanError("Integration assignment exceeds its preflight bound")
        return assignment

    def validate_prompt_capacity(self, prefix: str) -> None:
        self.validate_response_capacity()
        if (
            len(self.shards) > 1
            and len(prefix.encode("utf-8")) + self.integration_max_bytes
            > MAX_PROMPT_BYTES
        ):
            raise PlanError(
                "Root policy and worst-case integration exceed the "
                "prompt budget before review"
            )


def check_deadline(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise PlanError("Aggregate review deadline expired")
    return remaining


def _git(repository: Path, *args: str, deadline: float) -> bytes:
    check_deadline(deadline)
    try:
        data = review_execution.git_output(
            repository,
            *args,
            deadline=deadline,
            max_output_bytes=MAX_GIT_METADATA_BYTES,
        )
    except (OSError, review_execution.ExecutionError) as error:
        raise PlanError(f"Cannot read immutable Git input: {args}: {error}") from error
    check_deadline(deadline)
    return data


def _blobs(
    repository: Path, entries: list[dict[str, Any]], deadline: float
) -> dict[str, bytes]:
    """Read exact inventory objects in bounded batches, validating Git framing."""
    # Count paths, not only unique objects: repeated content still has distinct
    # examination coverage. Reject the complete input before reading any batch.
    if len(entries) > MAX_INVENTORY_ENTRIES:
        raise PlanError("Selected inventory exceeds review entry budget")
    total = 0
    for entry in entries:
        check_deadline(deadline)
        if not 0 <= entry["size"] <= MAX_BLOB_BYTES:
            raise PlanError("Git blob exceeds review safety bound")
        total += entry["size"]
        if total > MAX_TOTAL_BLOB_BYTES:
            raise PlanError("Selected Git blobs exceed aggregate review byte budget")
    pending: list[dict[str, Any]] = []
    result: dict[str, bytes] = {}

    def read_batch() -> None:
        bound = sum(
            len(f"{entry['blob']} blob {entry['size']}\n".encode("ascii"))
            + entry["size"]
            + 1
            for entry in pending
        )
        try:
            output = review_execution.git_output(
                repository,
                "cat-file",
                "--batch",
                deadline=deadline,
                input_bytes="".join(f"{entry['blob']}\n" for entry in pending).encode(
                    "ascii"
                ),
                max_output_bytes=bound,
            )
        except (OSError, review_execution.ExecutionError) as error:
            raise PlanError(f"Cannot read immutable Git blob batch: {error}") from error
        check_deadline(deadline)
        offset = 0
        for entry in pending:
            header = f"{entry['blob']} blob {entry['size']}\n".encode("ascii")
            if not output.startswith(header, offset):
                raise PlanError("Git blob batch identity/type/size mismatch")
            start = offset + len(header)
            end = start + entry["size"]
            if output[end : end + 1] != b"\n":
                raise PlanError("Truncated Git blob batch payload")
            result[entry["blob"]] = output[start:end]
            offset = end + 1
        if offset != len(output):
            raise PlanError("Unexpected trailing Git blob batch data")
        pending.clear()

    size = 0
    unique = {entry["blob"]: entry for entry in entries}
    for entry in unique.values():
        if pending and size + entry["size"] > BLOB_BATCH_BYTES:
            read_batch()
            size = 0
        pending.append(entry)
        size += entry["size"]
    if pending:
        read_batch()
    return result


def _inventory(repository: Path, tree: str, deadline: float) -> list[dict[str, Any]]:
    records = _git(repository, "ls-tree", "-rzl", tree, deadline=deadline)
    result: list[dict[str, Any]] = []
    for record in records.split(b"\0"):
        if not record:
            continue
        if len(result) >= MAX_INVENTORY_ENTRIES:
            raise PlanError("Git inventory exceeds review entry budget")
        metadata, raw_path = record.split(b"\t", 1)
        mode, kind, blob, size = metadata.decode("ascii").split()
        if kind != "blob" or mode not in {"100644", "100755", "120000"}:
            raise PlanError(
                "Unsupported tracked object; submodules require explicit review support"
            )
        path = raw_path.decode("utf-8", errors="strict")
        if int(size) > MAX_BLOB_BYTES:
            raise PlanError(f"Tracked blob exceeds review safety bound: {path}")
        result.append({"path": path, "mode": mode, "blob": blob, "size": int(size)})
    return result


def _declarations(data: bytes | None, paths: set[str]) -> dict[str, dict[str, Any]]:
    if data is None:
        return {}
    try:
        config = tomllib.loads(data.decode("utf-8"))
        entries = (
            config.get("tool", {})
            .get("h2h-dev-tools", {})
            .get("review", {})
            .get("artifacts", [])
        )
    except (ValueError, AttributeError) as error:
        raise PlanError("Invalid review artifact configuration") from error
    if not isinstance(entries, list):
        raise PlanError("review.artifacts must be an array of exact-path declarations")
    result = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) - {"sampled-data-members"} != {
            "path",
            "role",
            "format",
            "related-paths",
            "reason",
            "max-expanded-bytes",
            "max-members",
        }:
            raise PlanError("Malformed binary examination declaration")
        path = entry["path"]
        related = entry["related-paths"]
        if (
            not isinstance(path, str)
            or path not in paths
            or path in result
            or entry["role"] not in {"generated", "historical-evidence", "fixture"}
            or entry["format"] not in {"opaque", "gzip", "tar-gzip"}
            or not isinstance(entry["reason"], str)
            or not entry["reason"].strip()
            or not isinstance(related, list)
            or not related
            or any(
                not isinstance(p, str) or p not in paths or p == path for p in related
            )
            or len(set(related)) != len(related)
        ):
            raise PlanError(f"Invalid artifact declaration: {path!r}")
        for field, maximum in (
            ("max-expanded-bytes", MAX_EXPANDED_BYTES),
            ("max-members", MAX_MEMBERS),
        ):
            if type(entry[field]) is not int or not 1 <= entry[field] <= maximum:
                raise PlanError(f"Unsafe artifact limit: {path!r} {field}")
        if entry["format"] in {"opaque", "gzip"} and entry["max-members"] != 1:
            raise PlanError("Single-payload artifact must declare max-members=1")
        sampled = entry.get("sampled-data-members", [])
        if (
            not isinstance(sampled, list)
            or any(
                not isinstance(name, str) or not name.endswith(".json")
                for name in sampled
            )
            or len(sampled) != len(set(sampled))
            or (
                sampled
                and (
                    entry["role"] != "historical-evidence"
                    or entry["format"] != "tar-gzip"
                )
            )
        ):
            raise PlanError(
                "Only exact historical JSON evidence members may be sampled"
            )
        seen: set[str] = set()
        for name in sampled:
            _safe_member(name, seen)
        result[path] = entry
    return result


def _safe_member(name: str, seen: set[str]) -> None:
    path = PurePosixPath(name)
    if (
        not name
        or name.startswith("/")
        or "\\" in name
        or "\0" in name
        or any(part in {"", ".", ".."} for part in name.rstrip("/").split("/"))
        or path.as_posix().rstrip("/") in seen
    ):
        raise PlanError(f"Unsafe or duplicate archive member: {name!r}")
    seen.add(path.as_posix().rstrip("/"))


def _is_textual(data: bytes) -> bool:
    try:
        data.decode("utf-8")
    except UnicodeError:
        return False
    return b"\0" not in data


def _sample(data: bytes) -> dict[str, Any]:
    return {
        "size": len(data),
        "sha256": digest(data),
        "head_hex": data[:256].hex(),
        "tail_hex": data[-256:].hex(),
    }


class _JSONNumber(str):
    """An already parsed JSON number lexeme, without float/int conversion loss."""


def _reject_json_constant(constant: str) -> Never:
    raise ValueError(f"Non-JSON numeric constant: {constant}")


def _data_shape(value: Any, depth: int = 0) -> Any:
    """Bound displayed samples; parsing validates syntax, not numerical truth."""
    if isinstance(value, (dict, list)):
        if depth >= 4:
            return {"type": type(value).__name__, "length": len(value)}
        if isinstance(value, dict):
            keys = list(value)
            selected = list(dict.fromkeys(keys[:3] + keys[-3:]))
            return {
                "type": "object",
                "length": len(value),
                "samples": {
                    str(k)[:160]: _data_shape(value[k], depth + 1) for k in selected
                },
            }
        indices = sorted(
            set(range(min(3, len(value))))
            | set(range(max(0, len(value) - 3), len(value)))
        )
        return {
            "type": "array",
            "length": len(value),
            "samples": {str(i): _data_shape(value[i], depth + 1) for i in indices},
        }
    if isinstance(value, str):
        return {
            "type": "number" if isinstance(value, _JSONNumber) else "string",
            "length": len(value),
            "sample": value[:160],
        }
    return value


@dataclass
class _PlanBudget:
    expanded_bytes: int = 0
    view_bytes: int = 0
    unit_metadata_bytes: int = 0

    def expand(
        self,
        data: bytes,
        limit: int,
        *,
        limit_error: str = "Compressed payload exceeds its declared expanded limit",
    ) -> bytes:
        remaining = MAX_TOTAL_EXPANDED_BYTES - self.expanded_bytes
        with gzip.GzipFile(fileobj=io.BytesIO(data)) as stream:
            payload = stream.read(min(limit, remaining) + 1)
        if len(payload) > remaining:
            raise PlanError("Expanded artifacts exceed aggregate review byte budget")
        if len(payload) > limit:
            raise PlanError(limit_error)
        self.expanded_bytes += len(payload)
        return payload

    def retain_view(self, size: int) -> None:
        if self.view_bytes + size > MAX_TOTAL_VIEW_BYTES:
            raise PlanError("Examination views exceed aggregate review byte budget")
        self.view_bytes += size

    def retain_unit(self, unit: Unit, count: int) -> None:
        if count >= MAX_REVIEW_UNITS:
            raise PlanError("Review units exceed aggregate count budget")
        size = len(canonical(asdict(unit)))
        if self.unit_metadata_bytes + size > MAX_UNIT_METADATA_BYTES:
            raise PlanError("Review unit metadata exceeds aggregate byte budget")
        self.unit_metadata_bytes += size


def _check_json_budget(data: bytes, *, deadline: float) -> None:
    """Bound DOM construction; json.loads remains the complete syntax validator.

    Each colon permits at most a key and value, each comma another item, and
    each array opener its first item. This conservative slot count also bounds
    nested containers. Quoted/escaped punctuation contributes no structure.
    """
    if len(data) > MAX_SAMPLED_JSON_BYTES:
        raise PlanError("Sampled JSON exceeds its parse byte budget")
    # Match json.loads(bytes), including BOMs, UTF-16/32 and escaped surrogates.
    # The input byte cap bounds this temporary string, which is released before
    # json.loads constructs a DOM. Scanning raw bytes would misread UTF-16 escapes.
    try:
        text = data.decode(json.detect_encoding(data), "surrogatepass")
    except UnicodeError as error:
        raise PlanError("Cannot safely inspect archived JSON evidence") from error
    slots, depth = 1, 0
    quoted = escaped = False
    for offset in range(0, len(text), UNIT_BYTES):
        check_deadline(deadline)
        for character in text[offset : offset + UNIT_BYTES]:
            if quoted:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    quoted = False
            elif character == '"':
                quoted = True
            elif character in ("[", "{"):
                depth += 1
                slots += character == "["
            elif character in ("]", "}"):
                depth -= 1
            elif character == ":":
                slots += 2
            elif character == ",":
                slots += 1
            if slots > MAX_JSON_SLOTS or depth > MAX_JSON_DEPTH:
                raise PlanError("Sampled JSON exceeds its structure budget")
    check_deadline(deadline)


def _json_sample(data: bytes, *, deadline: float) -> bytes:
    # Keep the DOM local so the next member cannot coexist with the old DOM.
    _check_json_budget(data, deadline=deadline)
    try:
        parsed = json.loads(
            data,
            parse_constant=_reject_json_constant,
            parse_int=_JSONNumber,
            parse_float=_JSONNumber,
        )
        return canonical(
            {
                **_sample(data),
                "structure_and_value_samples": _data_shape(parsed),
                "limits": (
                    "Entire JSON syntax parsed; number samples retain "
                    "bounded original lexemes, not numerical recomputation "
                    "or full semantic proof."
                ),
            }
        )
    except (ValueError, RecursionError) as error:
        raise PlanError("Cannot safely inspect archived JSON evidence") from error


def _binary_views(
    data: bytes, entry: dict[str, Any], *, deadline: float, budget: _PlanBudget
) -> list[tuple[str | None, str, bytes]]:
    """Never extract, follow links, execute sources or unpickle candidate data."""
    check_deadline(deadline)
    limit = entry["max-expanded-bytes"]
    members: list[tuple[str, bytes, int]] = []
    inventory: list[dict[str, Any]] = []
    if entry["format"] == "opaque":
        if len(data) > limit:
            raise PlanError("Opaque artifact exceeds its declared byte limit")
        payload = data
    elif entry["format"] == "gzip":
        payload = budget.expand(data, limit)
    elif entry["format"] == "tar-gzip":
        total = 0
        seen: set[str] = set()

        # Streaming limits include tar headers/padding, not only member sizes.
        expanded = budget.expand(
            data,
            limit + MAX_MEMBERS * 1024,
            limit_error="Tar stream exceeds its bounded envelope",
        )
        with tarfile.open(fileobj=io.BytesIO(expanded), mode="r:") as archive:
            for member in archive:
                if not member.isfile() and not member.isdir():
                    raise PlanError(
                        "Archive links/devices/special members are unsupported"
                    )

                check_deadline(deadline)
                _safe_member(member.name, seen)
                total += member.size
                if len(seen) > entry["max-members"] or total > limit:
                    raise PlanError(
                        "Archive inventory exceeds its declared safety budget"
                    )
                content = b""
                if member.isfile():
                    member_stream = archive.extractfile(member)
                    if member_stream is None:
                        raise PlanError("Archive member has no readable payload")
                    with member_stream:
                        content = member_stream.read(limit + 1)
                if len(content) != member.size:
                    raise PlanError("Archive member size disagrees with its payload")
                inventory.append(
                    {
                        "path": member.name,
                        "type": "directory" if member.isdir() else "file",
                        "mode": member.mode,
                        "size": member.size,
                        "sha256": digest(content),
                    }
                )
                if member.isfile():
                    members.append((member.name, content, member.mode))
            # tarfile stops at the first zero block and can also silently stop
            # at a later invalid/truncated header. Its next-header offset must
            # therefore leave only the complete two-block end marker and zero
            # block padding, never another stream or unexamined payload.
            tail_size = len(expanded) - archive.offset
            if (
                archive.offset % tarfile.BLOCKSIZE
                or tail_size < 2 * tarfile.BLOCKSIZE
                or tail_size % tarfile.BLOCKSIZE
                or expanded.count(0, archive.offset) != tail_size
            ):
                raise PlanError("Tar framing is incomplete or contains unexamined data")
            check_deadline(deadline)
        views: list[tuple[str | None, str, bytes]] = [
            (
                None,
                "archive-inventory",
                canonical({"declaration": entry, "members": inventory}),
            )
        ]
        sampled = set(entry.get("sampled-data-members", []))
        available = {name for name, _content, mode in members if not mode & 0o111}
        if not sampled <= available:
            raise PlanError(
                "Sampled member is absent, executable or not a regular file"
            )
        for name, content, _mode in members:
            check_deadline(deadline)
            if name in sampled:
                views.append(
                    (name, "data-samples", _json_sample(content, deadline=deadline))
                )
            else:
                try:
                    content.decode("utf-8")
                except UnicodeError as error:
                    raise PlanError(
                        f"Unknown embedded binary requires a supported "
                        f"examination strategy: {name}"
                    ) from error
                if b"\0" in content:
                    raise PlanError(
                        f"Embedded binary cannot masquerade as text: {name}"
                    )
                views.append((name, "archive-text", content))
        check_deadline(deadline)
        return views
    else:
        raise PlanError(f"Unsupported binary artifact format: {entry['format']!r}")
    check_deadline(deadline)
    if entry["format"] == "gzip" and _is_textual(payload):
        return [
            (
                None,
                "binary-inventory",
                canonical(
                    {
                        "declaration": entry,
                        "compressed": _sample(data),
                        "payload": _sample(payload),
                    }
                ),
            ),
            (None, "gzip-text", payload),
        ]
    return [
        (
            None,
            "binary-contract",
            canonical(
                {
                    "declaration": entry,
                    "payload": _sample(payload),
                    "limits": (
                        "Opaque data was not executed/deserialized. Review "
                        "generator, restricted decoder, validation, source "
                        "correspondence and consumers; report incomplete if "
                        "these cannot establish examination."
                    ),
                }
            ),
        )
    ]


def _affected_artifacts(
    declarations: dict[str, dict[str, Any]],
    historical: dict[str, dict[str, Any]],
    changed: set[str],
    *,
    deadline: float,
) -> set[str]:
    """Include changed examination contracts and their source-dependent artifacts."""
    selected = changed | {
        path
        for path in declarations.keys() | historical.keys()
        if declarations.get(path) != historical.get(path)
    }
    dependents: dict[str, set[str]] = {}
    # Current paths use current contracts. Removed artifacts retain the exact
    # parent's contract, including when their related source is also removed.
    for path, declaration in (historical | declarations).items():
        check_deadline(deadline)
        for related in declaration["related-paths"]:
            dependents.setdefault(related, set()).add(path)
    pending = list(selected)
    while pending:
        check_deadline(deadline)
        for path in dependents.get(pending.pop(), set()):
            if path not in selected:
                selected.add(path)
                pending.append(path)
    return selected


def build_plan(
    repository: Path, target: dict[str, Any], scope: dict[str, Any], *, deadline: float
) -> Plan:
    tree = target["tree"]
    inventory = _inventory(repository, tree, deadline)
    paths = {item["path"] for item in inventory}
    by_path = {item["path"]: item for item in inventory}
    project = by_path.get("pyproject.toml")
    config = (
        None
        if project is None
        else _git(repository, "cat-file", "blob", project["blob"], deadline=deadline)
    )
    declarations = _declarations(config, paths)
    selected = paths
    origins = dict.fromkeys(paths, tree)
    historical_declarations: dict[str, dict[str, Any]] = {}
    historical_related: dict[str, dict[str, Any]] = {}
    if scope["scope"] == "incremental":
        base = target["parents"][0]
        selected = {
            item.decode("utf-8")
            for item in _git(
                repository,
                "diff",
                "--name-only",
                "--no-renames",
                "-z",
                base,
                tree,
                "--",
                deadline=deadline,
            ).split(b"\0")
            if item
        }
        previous = {
            item["path"]: item for item in _inventory(repository, base, deadline)
        }
        previous_project = previous.get("pyproject.toml")
        previous_config = (
            None
            if previous_project is None
            else _git(
                repository,
                "cat-file",
                "blob",
                previous_project["blob"],
                deadline=deadline,
            )
        )
        historical_declarations = _declarations(previous_config, set(previous))
        selected = _affected_artifacts(
            declarations, historical_declarations, selected, deadline=deadline
        )
        for path in selected - paths:
            for related_path in historical_declarations.get(path, {}).get(
                "related-paths", []
            ):
                historical_related[related_path] = {
                    **previous[related_path],
                    "origin": base,
                }
            by_path[path] = previous[path]
            origins[path] = base
    units: list[Unit] = []
    payloads: dict[str, str] = {}
    binary_paths = set()
    budget = _PlanBudget()
    blobs = _blobs(repository, [by_path[path] for path in sorted(selected)], deadline)
    for path in sorted(selected):
        check_deadline(deadline)
        applicable = declarations if path in paths else historical_declarations
        item = by_path[path]
        data = blobs[item["blob"]]
        if len(data) != item["size"]:
            raise PlanError("Git blob size drift")
        if _is_textual(data):
            if path in applicable:
                raise PlanError(
                    f"Text cannot be excluded through a binary declaration: {path}"
                )
            views: list[tuple[str | None, str, bytes]] = [
                (None, "text" if path in paths else "deleted-text", data)
            ]
            related: tuple[str, ...] = ()
            examination = (
                "Read every assigned byte range and its complete "
                "file/related contracts; symlink payload is a link "
                "target, never follow it."
            )
        else:
            if path not in applicable:
                raise PlanError(
                    f"Undeclared binary requires an exact-path examination "
                    f"contract: {path}"
                )
            binary_paths.add(path)
            declaration = applicable[path]
            views = _binary_views(data, declaration, deadline=deadline, budget=budget)
            related = tuple(declaration["related-paths"])
            examination = declaration["reason"]
        for member, kind, content in views:
            budget.retain_view(len(content))
            content_digest = digest(content)
            start = 0
            while start < len(content) or (not content and start == 0):
                check_deadline(deadline)
                end = min(start + UNIT_BYTES, len(content))
                while end < len(content) and content[end] & 0xC0 == 0x80:
                    end -= 1
                key = f"u{len(units):06d}"
                unit = Unit(
                    key,
                    path,
                    item["blob"],
                    item["mode"],
                    origins[path],
                    member,
                    kind,
                    start,
                    end,
                    len(content),
                    content_digest,
                    examination,
                    related,
                )
                budget.retain_unit(unit, len(units))
                units.append(unit)
                payloads[key] = content[start:end].decode("utf-8")
                if end == len(content):
                    break
                start = end
    if scope["scope"] == "repository" and set(declarations) != binary_paths:
        raise PlanError("Artifact declarations must exactly match the binary inventory")
    shards: list[tuple[str, ...]] = []
    pending: list[str] = []
    pending_units: list[Unit] = []
    size = 0
    for unit in units:
        cost = len(canonical({**asdict(unit), "content": payloads[unit.id]}))
        if (
            response_capacity(
                (unit,), source_paths=(), shard_id="shard-0000", integration=False
            )
            > SHARD_RESULT_BYTES
        ):
            raise PlanError("A single review unit exceeds the response wire budget")
        if pending and (
            size + cost > SHARD_BYTES
            or len(pending) >= SHARD_UNITS
            or response_capacity(
                (*pending_units, unit),
                source_paths=(),
                shard_id="shard-0000",
                integration=False,
            )
            > SHARD_RESULT_BYTES
        ):
            shards.append(tuple(pending))
            pending, pending_units, size = [], [], 0
        pending.append(unit.id)
        pending_units.append(unit)
        size += cost
    if pending:
        shards.append(tuple(pending))
    if not units:
        raise PlanError("Empty review assignment is not a passing scope")
    check_deadline(deadline)
    plan = Plan(
        target,
        scope,
        tuple(inventory),
        tuple(units),
        tuple(shards),
        payloads,
        tuple(historical_related[path] for path in sorted(historical_related)),
    )
    # One shard also performs integration. If the combined response cannot fit,
    # preserve all units and use separate shard and integration invocations.
    if len(plan.shards) == 1 and plan.response_capacity(0) > MAX_RESULT_BYTES:
        if len(units) == 1:
            raise PlanError("A single complete review exceeds the response wire budget")
        middle = len(units) // 2
        plan = replace(
            plan,
            shards=(
                tuple(unit.id for unit in units[:middle]),
                tuple(unit.id for unit in units[middle:]),
            ),
        )
    plan.validate_prompt_capacity("")
    check_deadline(deadline)
    return plan


def validate_plan(document: object) -> None:
    """Reject malformed or overlapping/gapped range coverage, without model claims."""
    if not isinstance(document, dict):
        raise PlanError("Malformed review plan")
    units = document.get("units")
    shards = document.get("shards")
    if not isinstance(units, list) or not units or not isinstance(shards, list):
        raise PlanError("Missing review units/shards")
    identities = set()
    ranges: dict[tuple[str, str | None, str], list[dict[str, Any]]] = {}
    for unit in units:
        if not isinstance(unit, dict) or set(unit) != set(Unit.__dataclass_fields__):
            raise PlanError("Malformed review unit")
        key = unit["id"]
        if not isinstance(key, str) or key in identities:
            raise PlanError("Duplicate/invalid review unit")
        identities.add(key)
        for field in ("start", "end", "size"):
            if type(unit[field]) is not int:
                raise PlanError("Review byte ranges must be exact integers")
        if (
            not 0 <= unit["start"] <= unit["end"] <= unit["size"]
            or unit["end"] - unit["start"] > UNIT_BYTES
        ):
            raise PlanError("Malformed review byte range")
        ranges.setdefault((unit["path"], unit["member"], unit["kind"]), []).append(unit)
    for group in ranges.values():
        end = 0
        for unit in sorted(group, key=lambda item: item["start"]):
            if (
                unit["start"] != end
                or unit["sha256"] != group[0]["sha256"]
                or unit["size"] != group[0]["size"]
            ):
                raise PlanError("Incomplete or overlapping review range coverage")
            end = unit["end"]
        if end != group[0]["size"]:
            raise PlanError("Review ranges do not cover their entire examination view")
    assigned = [key for shard in shards for key in shard]
    if len(assigned) != len(set(assigned)) or set(assigned) != identities:
        raise PlanError("Missing, duplicate or unknown unit assignment")
