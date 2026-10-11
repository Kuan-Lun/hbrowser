"""Private admission ledger for cooperating nested POSIX process owners.

An obligation is registered before a nested session can exist. Closing admission
and inspecting obligations share one bounded lock; no PID observation substitutes
for the nested owner's proof that its own group and obligations are empty.
"""

from __future__ import annotations

import json
import os
import secrets
import tempfile
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

ENVIRONMENT_KEY = "H2H_PRIVATE_POSIX_PROCESS_SCOPE"
_POLL_SECONDS = 0.01


class ScopeError(RuntimeError):
    """Admission or recursive cleanup proof is unavailable."""


@dataclass(frozen=True)
class _Capability:
    directory: Path
    token: str

    @classmethod
    def decode(cls, value: str) -> _Capability:
        try:
            data = json.loads(value)
            if (
                not isinstance(data, dict)
                or set(data) != {"directory", "token"}
                or not isinstance(data["directory"], str)
                or not Path(data["directory"]).is_absolute()
                or not isinstance(data["token"], str)
                or len(data["token"]) != 64
                or any(c not in "0123456789abcdef" for c in data["token"])
            ):
                raise ValueError("Invalid scope capability")
            return cls(Path(data["directory"]), data["token"])
        except (TypeError, ValueError) as error:
            raise ScopeError("Invalid inherited POSIX process scope") from error

    def encode(self) -> str:
        return json.dumps({"directory": str(self.directory), "token": self.token})

    @contextmanager
    def locked(self, *, deadline: float) -> Iterator[None]:
        import fcntl

        try:
            with (self.directory / "lock").open("rb") as lock:
                while True:
                    if time.monotonic() >= deadline:
                        raise ScopeError(
                            f"Process scope lock deadline: {self.directory}"
                        )
                    try:
                        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        time.sleep(
                            min(_POLL_SECONDS, max(0.0, deadline - time.monotonic()))
                        )
                try:
                    with (self.directory / "token").open(
                        "r", encoding="ascii"
                    ) as source:
                        token = source.read(65)
                    if token != self.token:
                        raise ScopeError(
                            f"Process scope identity changed: {self.directory}"
                        )
                    yield
                finally:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        except OSError as error:
            raise ScopeError(f"Process scope unavailable: {self.directory}") from error

    def pending(self) -> tuple[Path, ...]:
        return tuple((self.directory / "pending").iterdir())


@dataclass
class PosixScope:
    capability: _Capability
    parent: _Capability | None
    proven_empty: bool = False
    cleanup_deadline: float = 0.0

    @classmethod
    def create(cls, *, deadline: float) -> PosixScope:
        inherited = os.environ.get(ENVIRONMENT_KEY)
        parent = None if inherited is None else _Capability.decode(inherited)
        directory = Path(tempfile.mkdtemp(prefix="h2h-process-scope-"))
        capability = _Capability(directory, secrets.token_hex(32))
        try:
            (directory / "lock").touch()
            (directory / "token").write_text(capability.token, encoding="ascii")
            (directory / "pending").mkdir()
            if parent is not None:
                with parent.locked(deadline=deadline):
                    if (parent.directory / "closed").exists():
                        raise ScopeError("Outer process owner has closed admission")
                    with (parent.directory / "pending" / capability.token).open(
                        "x", encoding="utf-8"
                    ) as obligation:
                        obligation.write(capability.encode())
        except BaseException:
            # No child was launched. An interrupted partial parent obligation
            # remains fail-closed; never erase an uncertain registration.
            for name in ("lock", "token"):
                (directory / name).unlink(missing_ok=True)
            (directory / "pending").rmdir()
            directory.rmdir()
            raise
        return cls(capability, parent)

    def environment(self, original: Mapping[str, str] | None) -> dict[str, str]:
        environment = dict(os.environ if original is None else original)
        environment[ENVIRONMENT_KEY] = self.capability.encode()
        return environment

    def close_admission(self, *, deadline: float) -> None:
        with self.capability.locked(deadline=deadline):
            (self.capability.directory / "closed").touch()

    def has_pending(self, *, deadline: float) -> bool:
        with self.capability.locked(deadline=deadline):
            return bool(self.capability.pending())

    def close(self, *, deadline: float) -> None:
        if not self.proven_empty:
            raise ScopeError(
                f"Unproven process scope retained: {self.capability.directory}"
            )
        with self.capability.locked(deadline=deadline):
            if self.capability.pending():
                raise ScopeError("Nested cleanup obligations remain")
            (self.capability.directory / "closed").touch()
            if self.parent is not None:
                with self.parent.locked(deadline=deadline):
                    obligation = (
                        self.parent.directory / "pending" / self.capability.token
                    )
                    if (
                        obligation.read_text(encoding="utf-8")
                        != self.capability.encode()
                    ):
                        raise ScopeError("Nested cleanup obligation identity changed")
                    obligation.unlink()
        # The capability is closed before removal; a queued admission cannot
        # succeed through either the old lock descriptor or the removed path.
        for name in ("closed", "token", "lock"):
            (self.capability.directory / name).unlink()
        (self.capability.directory / "pending").rmdir()
        self.capability.directory.rmdir()
