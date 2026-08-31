"""Start-gated subprocess supervisor used by hbrowser process owners."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Never, cast

from ._process_control import (
    MAX_CONTROL_LINE_BYTES,
    MAX_START_LINE_BYTES,
    PROVEN_CLEANUP_FAILURE_EXIT_CODE,
    PROVEN_PROTOCOL_FAILURE_EXIT_CODE,
    PROVEN_TARGET_NOT_STARTED_EXIT_CODE,
    ControlIntent,
    ControlRequest,
    StartRequest,
)

_POLL_SECONDS = 0.05
_FALLBACK_TERM_SECONDS = 2.0
_FALLBACK_KILL_SECONDS = 2.0
_POSIX_PS_EXECUTABLE = shutil.which("ps", path=os.defpath)
_TARGET_ONLY_ENVIRONMENT_KEYS = (
    "HBROWSER_LOG_FORWARD_ENDPOINT",
    "HBROWSER_LOG_FORWARD_TOKEN",
)


class _TargetGroupIdentity(Enum):
    LIVE_OWNED = auto()
    EXITED_PINNED = auto()


class _SignalDisposition(Enum):
    DELIVERED = auto()
    GROUP_ABSENT = auto()
    DENIED = auto()


class _PhaseDeadlineExpired(RuntimeError):
    """The active control phase ended before its ownership receipt."""


class _OwnershipProofInvalid(RuntimeError):
    """The retained identity can no longer prove the assigned ownership set."""


@dataclass(frozen=True, slots=True)
class _ControllerState:
    """One atomically observed controller generation."""

    request: ControlRequest | None
    revision: int


@dataclass(frozen=True, slots=True)
class _ActionAttempt:
    """Result of linearizing one signal attempt against a controller state."""

    attempted: bool
    disposition: _SignalDisposition | None
    state: _ControllerState


class _ShutdownController:
    """Merge active control plans and admit fresh phases after expiry."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._request: ControlRequest | None = None
        self._revision = 0
        self._protocol_failed = False
        self._autonomous = False

    def _replace_request_locked(
        self,
        request: ControlRequest,
        *,
        force_revision: bool = False,
    ) -> None:
        if request == self._request and not force_revision:
            return
        self._request = request
        self._revision += 1
        self._condition.notify_all()

    def _promote_due_terminate_locked(self, now_ns: int) -> bool:
        request = self._request
        if (
            request is None
            or request.intent is ControlIntent.KILL
            or request.overall_deadline_ns is None
            or now_ns < request.phase_deadline_ns
        ):
            return False
        kill_deadline_ns = request.overall_deadline_ns
        if self._autonomous and kill_deadline_ns <= now_ns:
            kill_deadline_ns = now_ns + int(_FALLBACK_KILL_SECONDS * 1_000_000_000)
        self._replace_request_locked(
            ControlRequest(
                intent=ControlIntent.KILL,
                phase_deadline_ns=kill_deadline_ns,
                overall_deadline_ns=kill_deadline_ns,
                allow_immediate=(
                    request.phase_deadline_ns == request.overall_deadline_ns
                ),
            )
        )
        return True

    def apply(self, request: ControlRequest) -> None:
        with self._condition:
            now_ns = time.monotonic_ns()
            # Once a finite TERM phase is due, its KILL escalation is a
            # commitment. A later standalone TERM must not make the result
            # depend on whether the supervisor loop promoted it first.
            self._promote_due_terminate_locked(now_ns)
            current = self._request
            if current is None or (
                request.intent is current.intent
                and self._phase_is_exhausted(current, now_ns)
            ):
                # A completed phase may be retried with a fresh absolute
                # deadline. Preserve an unconsumed immediate entitlement;
                # deadline expiry alone does not prove that its action ran.
                if current is not None and current.allow_immediate:
                    request = ControlRequest(
                        intent=request.intent,
                        phase_deadline_ns=request.phase_deadline_ns,
                        overall_deadline_ns=request.overall_deadline_ns,
                        allow_immediate=True,
                    )
                self._replace_request_locked(request, force_revision=True)
            else:
                self._replace_request_locked(
                    current.merge(request),
                    force_revision=request.intent >= current.intent,
                )

    @staticmethod
    def _phase_is_exhausted(request: ControlRequest, now_ns: int) -> bool:
        return now_ns >= request.phase_deadline_ns

    def request_fallback(self, intent: ControlIntent) -> None:
        now_ns = time.monotonic_ns()
        with self._condition:
            if not self._autonomous:
                self._autonomous = True
                self._revision += 1
                self._condition.notify_all()
            self._promote_due_terminate_locked(now_ns)
            current = self._request
            effective_intent = (
                intent if current is None else max(intent, current.intent)
            )
            if effective_intent is ControlIntent.TERMINATE:
                phase_deadline_ns = now_ns + int(_FALLBACK_TERM_SECONDS * 1_000_000_000)
                overall_deadline_ns = phase_deadline_ns + int(
                    _FALLBACK_KILL_SECONDS * 1_000_000_000
                )
            else:
                phase_deadline_ns = now_ns + int(_FALLBACK_KILL_SECONDS * 1_000_000_000)
                overall_deadline_ns = phase_deadline_ns
            request = ControlRequest(
                intent=effective_intent,
                phase_deadline_ns=phase_deadline_ns,
                overall_deadline_ns=overall_deadline_ns,
            )
            current = self._request
            if current is None or self._phase_is_exhausted(current, now_ns):
                if current is not None and current.allow_immediate:
                    request = ControlRequest(
                        intent=request.intent,
                        phase_deadline_ns=request.phase_deadline_ns,
                        overall_deadline_ns=request.overall_deadline_ns,
                        allow_immediate=True,
                    )
                self._replace_request_locked(request, force_revision=True)
            else:
                self._replace_request_locked(
                    current.merge(request),
                    force_revision=True,
                )

    def promote_planned_kill(self) -> bool:
        with self._condition:
            return self._promote_due_terminate_locked(time.monotonic_ns())

    def renew_autonomous_kill(self) -> bool:
        """Give an expired owner-death KILL plan one fresh bounded phase."""

        with self._condition:
            self._promote_due_terminate_locked(time.monotonic_ns())
            request = self._request
            now_ns = time.monotonic_ns()
            if (
                not self._autonomous
                or request is None
                or request.intent is not ControlIntent.KILL
                or now_ns < request.phase_deadline_ns
            ):
                return False
            kill_deadline_ns = now_ns + int(_FALLBACK_KILL_SECONDS * 1_000_000_000)
            self._replace_request_locked(
                ControlRequest(
                    intent=ControlIntent.KILL,
                    phase_deadline_ns=kill_deadline_ns,
                    overall_deadline_ns=kill_deadline_ns,
                )
            )
            return True

    def fail_protocol(self) -> None:
        with self._condition:
            if not self._protocol_failed:
                self._protocol_failed = True
                self._revision += 1
                self._condition.notify_all()
        self.request_fallback(ControlIntent.KILL)

    @property
    def protocol_failed(self) -> bool:
        with self._condition:
            return self._protocol_failed

    def snapshot(self) -> ControlRequest | None:
        return self.state().request

    def state(self) -> _ControllerState:
        with self._condition:
            self._promote_due_terminate_locked(time.monotonic_ns())
            return _ControllerState(self._request, self._revision)

    def wait_for_change(
        self,
        observed: _ControllerState,
        timeout: float,
    ) -> _ControllerState:
        """Wait only while the exact observed generation is still current."""

        with self._condition:
            self._promote_due_terminate_locked(time.monotonic_ns())
            if self._revision == observed.revision:
                self._condition.wait_for(
                    lambda: self._revision != observed.revision,
                    timeout=max(0.0, timeout),
                )
            self._promote_due_terminate_locked(time.monotonic_ns())
            return _ControllerState(self._request, self._revision)

    def perform_if_actionable(
        self,
        observed: _ControllerState,
        intent: ControlIntent,
        action: Callable[[ControlRequest], _SignalDisposition],
    ) -> _ActionAttempt:
        """Linearize a signal action against deadline shrink and escalation."""

        with self._condition:
            self._promote_due_terminate_locked(time.monotonic_ns())
            request = self._request
            if (
                self._revision != observed.revision
                or request is None
                or request.intent is not intent
                or (
                    time.monotonic_ns() >= request.phase_deadline_ns
                    and not request.allow_immediate
                )
            ):
                return _ActionAttempt(
                    attempted=False,
                    disposition=None,
                    state=_ControllerState(self._request, self._revision),
                )
            if request.allow_immediate:
                # The entitlement permits exactly one action attempt after an
                # expired deadline. A failed attempt needs a fresh request.
                self._replace_request_locked(
                    ControlRequest(
                        intent=request.intent,
                        phase_deadline_ns=request.phase_deadline_ns,
                        overall_deadline_ns=request.overall_deadline_ns,
                    )
                )
            try:
                disposition = action(request)
            except _PhaseDeadlineExpired:
                disposition = None
            return _ActionAttempt(
                attempted=True,
                disposition=disposition,
                state=_ControllerState(self._request, self._revision),
            )

    def start_target_if_authorized(
        self,
        *,
        deadline_ns: int,
        action: Callable[[], subprocess.Popen[bytes]],
    ) -> tuple[subprocess.Popen[bytes] | None, str | None]:
        """Linearize target creation after the deadline and cancel barrier."""

        with self._condition:
            now_ns = time.monotonic_ns()
            if now_ns >= deadline_ns:
                return None, "StartupDeadlineExpired"
            if self._request is not None:
                return None, "StartupCancelled"
            # The controller lock is the start/cancel linearization point. A
            # request accepted before this point prevents Popen; one accepted
            # afterwards observes an owned target and drives ordinary cleanup.
            return action(), None


def _posix_exit_receipts_supported() -> bool:
    waitid = getattr(os, "waitid", None)
    fork = getattr(os, "fork", None)
    waitpid = getattr(os, "waitpid", None)
    primitives = tuple(
        getattr(os, name, None) for name in ("P_PID", "WEXITED", "WNOHANG", "WNOWAIT")
    )
    if (
        not callable(waitid)
        or not callable(fork)
        or not callable(waitpid)
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in primitives
        )
    ):
        return False
    process_id, exited, nohang, nowait = (
        cast(int, primitive) for primitive in primitives
    )
    try:
        # SIG_IGN/SA_NOCLDWAIT can survive exec and auto-reap the child before
        # WNOWAIT can pin its identity. The target requires default semantics.
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)
    except AttributeError, OSError, ValueError:
        return False
    try:
        probe_pid = fork()
    except OSError:
        return False
    if probe_pid == 0:
        os._exit(0)
    reaped = False
    try:
        # Exercise the exact operation against a disposable child, then prove
        # WNOWAIT really retained its waitable identity.
        receipt = waitid(process_id, probe_pid, exited | nowait)
        if receipt is None or getattr(receipt, "si_pid", None) != probe_pid:
            return False
        retained_receipt = waitid(
            process_id,
            probe_pid,
            exited | nohang | nowait,
        )
        if (
            retained_receipt is None
            or getattr(retained_receipt, "si_pid", None) != probe_pid
        ):
            return False
        waited_pid, _ = waitpid(probe_pid, 0)
        reaped = True
        return bool(waited_pid == probe_pid)
    except ChildProcessError, NotImplementedError, OSError:
        return False
    finally:
        if not reaped:
            try:
                waited_pid, _ = waitpid(probe_pid, nohang)
            except ChildProcessError, OSError:
                pass
            else:
                if waited_pid == 0:
                    # waitpid(WNOHANG) proved this numeric PID is still our
                    # unreaped child, so a cleanup signal cannot hit PID reuse.
                    try:
                        os.kill(probe_pid, signal.SIGKILL)
                    except PermissionError, ProcessLookupError:
                        pass
                    try:
                        waitpid(probe_pid, 0)
                    except ChildProcessError, OSError:
                        pass


def _write_status(status_path: Path, value: str) -> None:
    temporary = status_path.with_suffix(".tmp")
    temporary.write_text(f"{value}\n", encoding="utf-8")
    os.replace(temporary, status_path)


def _posix_target_has_exit_receipt(target: subprocess.Popen[bytes]) -> bool:
    """Observe child exit without releasing its PID or process-group identity."""

    try:
        result = os.waitid(
            os.P_PID,
            target.pid,
            os.WEXITED | os.WNOHANG | os.WNOWAIT,
        )
    except ChildProcessError:
        raise _OwnershipProofInvalid(
            "Owned target identity was reaped unexpectedly"
        ) from None
    return result is not None


def _state_or_raise(controller: _ShutdownController) -> _ControllerState:
    state = controller.state()
    if state.request is None:
        raise _OwnershipProofInvalid(
            "Process cleanup began without a shutdown deadline"
        )
    return state


def _remaining_seconds(deadline_ns: int) -> float:
    return max(0.0, (deadline_ns - time.monotonic_ns()) / 1_000_000_000)


def _wait_for_target_exit_proof(
    target: subprocess.Popen[bytes],
    *,
    controller: _ShutdownController,
) -> None:
    """Reconcile an ESRCH lookup without signalling an unproven numeric PID."""

    while True:
        state = controller.state()
        if _posix_target_has_exit_receipt(target):
            return
        controller.wait_for_change(state, _POLL_SECONDS)


def _wait_for_kill_request(
    controller: _ShutdownController,
    *,
    settle_if_exited: Callable[[], bool] | None = None,
) -> _ControllerState | None:
    """Wait for an actionable KILL while continuing natural-exit proof."""

    while True:
        state = _state_or_raise(controller)
        request = state.request
        assert request is not None
        if request.intent is ControlIntent.KILL:
            if (
                time.monotonic_ns() < request.phase_deadline_ns
                or request.allow_immediate
            ):
                return state
            if controller.renew_autonomous_kill():
                continue
            if settle_if_exited is not None and settle_if_exited():
                return None
            controller.wait_for_change(state, _POLL_SECONDS)
            continue
        if settle_if_exited is not None and settle_if_exited():
            return None
        if (
            time.monotonic_ns() >= request.phase_deadline_ns
            and controller.promote_planned_kill()
        ):
            continue
        controller.wait_for_change(
            state,
            min(_POLL_SECONDS, _remaining_seconds(request.phase_deadline_ns))
            if time.monotonic_ns() < request.phase_deadline_ns
            else _POLL_SECONDS,
        )


def _wait_for_posix_control_request(
    controller: _ShutdownController,
    *,
    attempted_term_revision: int | None,
    settle_if_exited: Callable[[], bool],
) -> _ControllerState | None:
    """Return each fresh TERM once, or the next actionable KILL."""

    while True:
        state = _state_or_raise(controller)
        request = state.request
        assert request is not None
        actionable = (
            time.monotonic_ns() < request.phase_deadline_ns or request.allow_immediate
        )
        if request.intent is ControlIntent.TERMINATE:
            if state.revision != attempted_term_revision and actionable:
                return state
        elif actionable:
            return state
        elif controller.renew_autonomous_kill():
            continue

        if settle_if_exited():
            return None
        state_after_proof = controller.state()
        if state_after_proof.revision != state.revision:
            continue
        remaining = _remaining_seconds(request.phase_deadline_ns)
        controller.wait_for_change(
            state,
            min(_POLL_SECONDS, remaining) if remaining > 0 else _POLL_SECONDS,
        )


def _process_group_members(
    process_group: int,
    *,
    deadline_ns: int,
) -> tuple[int, ...]:
    if _POSIX_PS_EXECUTABLE is None:
        raise _PhaseDeadlineExpired(
            "Owned target process-group proof executable is unavailable"
        )
    remaining = _remaining_seconds(deadline_ns)
    if remaining <= 0:
        raise _PhaseDeadlineExpired(
            "Owned target cleanup deadline expired before group proof"
        )
    try:
        result = subprocess.run(
            [_POSIX_PS_EXECUTABLE, "-axo", "pid=,pgid="],
            check=True,
            capture_output=True,
            env={"LC_ALL": "C", "PATH": os.defpath},
            text=True,
            timeout=min(1.0, remaining),
        )
    except subprocess.TimeoutExpired:
        raise _PhaseDeadlineExpired(
            "Owned target cleanup deadline expired during group proof"
        ) from None
    except OSError, subprocess.CalledProcessError:
        raise _PhaseDeadlineExpired(
            "Owned target process-group proof is temporarily unavailable"
        ) from None
    members: list[int] = []
    for line in result.stdout.splitlines():
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 2:
            raise _PhaseDeadlineExpired(
                "Owned target process-group proof returned malformed output"
            )
        try:
            pid, pgid = (int(field) for field in fields)
        except ValueError:
            raise _PhaseDeadlineExpired(
                "Owned target process-group proof returned malformed output"
            ) from None
        if pid <= 0 or pgid < 0:
            raise _PhaseDeadlineExpired(
                "Owned target process-group proof returned invalid identities"
            )
        if pgid == process_group:
            members.append(pid)
    if time.monotonic_ns() >= deadline_ns:
        raise _PhaseDeadlineExpired(
            "Owned target cleanup deadline expired during group proof"
        )
    return tuple(members)


def _posix_process_group_snapshots_supported() -> bool:
    """Functionally validate the exact strict ps proof before target spawn."""

    try:
        members = _process_group_members(
            os.getpgrp(),
            deadline_ns=time.monotonic_ns() + 1_000_000_000,
        )
    except _PhaseDeadlineExpired:
        return False
    return os.getpid() in members


def _target_group_identity(
    target: subprocess.Popen[bytes],
    *,
    controller: _ShutdownController,
) -> _TargetGroupIdentity:
    """Validate a live target or return its pinned post-exit identity state."""

    if _posix_target_has_exit_receipt(target):
        return _TargetGroupIdentity.EXITED_PINNED
    try:
        process_group = os.getpgid(target.pid)
        session = os.getsid(target.pid)
    except ProcessLookupError as error:
        # The target can exit after the caller's waitid(WNOWAIT) probe but
        # before either identity lookup.  Because it remains our unreaped
        # child, a positive second probe pins the PID and makes the cached
        # process-group identity safe to inspect during cleanup.
        try:
            _wait_for_target_exit_proof(target, controller=controller)
        except RuntimeError as ownership_error:
            raise ownership_error from error
        return _TargetGroupIdentity.EXITED_PINNED
    if process_group != target.pid or session != os.getsid(0):
        raise _OwnershipProofInvalid("Owned target escaped its assigned process group")
    return _TargetGroupIdentity.LIVE_OWNED


def _snapshot_proves_pinned_group(
    members: tuple[int, ...],
    target_pid: int,
) -> bool:
    """Require the retained target zombie as the sole group member."""

    return members == (target_pid,)


def _posix_group_is_clean(
    target: subprocess.Popen[bytes],
    *,
    controller: _ShutdownController,
) -> bool:
    if not _posix_target_has_exit_receipt(target):
        return False
    state = _state_or_raise(controller)
    request = state.request
    assert request is not None
    members = _process_group_members(
        target.pid,
        deadline_ns=request.phase_deadline_ns,
    )
    return _snapshot_proves_pinned_group(members, target.pid)


def _reap_posix_target(
    target: subprocess.Popen[bytes],
    *,
    controller: _ShutdownController,
) -> None:
    state = _state_or_raise(controller)
    request = state.request
    assert request is not None
    remaining = _remaining_seconds(request.phase_deadline_ns)
    if remaining <= 0:
        raise _PhaseDeadlineExpired("Owned target cleanup deadline expired before reap")
    target.wait(timeout=remaining)


def _signal_posix_group(
    target: subprocess.Popen[bytes],
    signal_number: int,
    *,
    request: ControlRequest,
) -> _SignalDisposition:
    if time.monotonic_ns() >= request.phase_deadline_ns and not request.allow_immediate:
        raise _PhaseDeadlineExpired(
            "Owned target control phase expired before process-group signal"
        )
    try:
        os.killpg(target.pid, signal_number)
    except PermissionError:
        return _SignalDisposition.DENIED
    except ProcessLookupError:
        return _SignalDisposition.GROUP_ABSENT
    return _SignalDisposition.DELIVERED


def _wait_for_posix_phase(
    target: subprocess.Popen[bytes],
    *,
    controller: _ShutdownController,
    intent: ControlIntent,
    observed: _ControllerState,
) -> bool:
    """Wait for a clean group, returning false when this phase must escalate."""

    while True:
        state = _state_or_raise(controller)
        request = state.request
        assert request is not None
        if state.revision != observed.revision or request.intent > intent:
            return False
        phase_deadline_ns = request.phase_deadline_ns
        remaining = _remaining_seconds(phase_deadline_ns)
        if remaining <= 0:
            return False
        try:
            if _posix_group_is_clean(target, controller=controller):
                _reap_posix_target(target, controller=controller)
                return True
        except _PhaseDeadlineExpired:
            controller.wait_for_change(state, min(_POLL_SECONDS, remaining))
            continue
        controller.wait_for_change(state, min(_POLL_SECONDS, remaining))


def _try_settle_posix_target(target: subprocess.Popen[bytes]) -> bool:
    """Prove and reap an exited target without releasing a live group identity."""

    if not _posix_target_has_exit_receipt(target):
        return False
    proof_deadline_ns = time.monotonic_ns() + 1_000_000_000
    try:
        members = _process_group_members(
            target.pid,
            deadline_ns=proof_deadline_ns,
        )
    except OSError, subprocess.CalledProcessError, _PhaseDeadlineExpired:
        return False
    if not _snapshot_proves_pinned_group(members, target.pid):
        return False
    target.wait()
    return True


def _settle_posix_target(
    target: subprocess.Popen[bytes],
    *,
    controller: _ShutdownController,
) -> None:
    """Retain ownership after SIGKILL until proof becomes available."""

    while True:
        state = controller.state()
        if _try_settle_posix_target(target):
            return
        controller.wait_for_change(state, _POLL_SECONDS)


def _wait_after_denied_posix_signal(
    target: subprocess.Popen[bytes],
    *,
    controller: _ShutdownController,
    observed: _ControllerState,
) -> bool:
    """Await a fresh request after denial while still accepting natural exit."""

    while True:
        state = controller.state()
        if state.revision != observed.revision:
            return False
        if _try_settle_posix_target(target):
            return True
        state = controller.state()
        if state.revision != observed.revision:
            return False
        request = state.request
        if request is None:
            raise _OwnershipProofInvalid("Process cleanup lost its shutdown request")
        remaining = _remaining_seconds(request.phase_deadline_ns)
        if remaining <= 0:
            return False
        controller.wait_for_change(state, min(_POLL_SECONDS, remaining))


def _terminate_posix_target(
    target: subprocess.Popen[bytes],
    *,
    controller: _ShutdownController,
) -> None:
    identity = _target_group_identity(target, controller=controller)
    if identity is _TargetGroupIdentity.EXITED_PINNED and _try_settle_posix_target(
        target
    ):
        return
    attempted_term_revision: int | None = None
    while True:
        control_state = _wait_for_posix_control_request(
            controller,
            attempted_term_revision=attempted_term_revision,
            settle_if_exited=lambda: _try_settle_posix_target(target),
        )
        if control_state is None:
            return
        request = control_state.request
        assert request is not None
        if request.intent is ControlIntent.TERMINATE:
            terminate_attempt = controller.perform_if_actionable(
                control_state,
                ControlIntent.TERMINATE,
                lambda active_request: _signal_posix_group(
                    target,
                    signal.SIGTERM,
                    request=active_request,
                ),
            )
            if terminate_attempt.attempted:
                attempted_term_revision = terminate_attempt.state.revision
                if _wait_for_posix_phase(
                    target,
                    controller=controller,
                    intent=ControlIntent.TERMINATE,
                    observed=terminate_attempt.state,
                ):
                    return
            continue
        kill_attempt = controller.perform_if_actionable(
            control_state,
            ControlIntent.KILL,
            lambda request: _signal_posix_group(
                target,
                signal.SIGKILL,
                request=request,
            ),
        )
        if kill_attempt.disposition in {
            _SignalDisposition.DELIVERED,
            _SignalDisposition.GROUP_ABSENT,
        }:
            break
        if (
            kill_attempt.disposition is _SignalDisposition.DENIED
            and _wait_after_denied_posix_signal(
                target,
                controller=controller,
                observed=kill_attempt.state,
            )
        ):
            return
    _settle_posix_target(target, controller=controller)


def _windows_target_exited(target: subprocess.Popen[bytes]) -> bool:
    return target.poll() is not None


def _terminate_windows_target(
    target: subprocess.Popen[bytes],
    *,
    controller: _ShutdownController,
) -> None:
    if _windows_target_exited(target):
        return

    def terminate_target(_: ControlRequest) -> _SignalDisposition:
        try:
            target.terminate()
        except ProcessLookupError:
            return _SignalDisposition.GROUP_ABSENT
        return _SignalDisposition.DELIVERED

    terminate_state = _state_or_raise(controller)
    terminate_attempt = controller.perform_if_actionable(
        terminate_state,
        ControlIntent.TERMINATE,
        terminate_target,
    )
    if terminate_attempt.disposition is not None:
        while True:
            if _windows_target_exited(target):
                return
            state = _state_or_raise(controller)
            request = state.request
            assert request is not None
            if (
                request.intent is ControlIntent.KILL
                or time.monotonic_ns() >= request.phase_deadline_ns
            ):
                break
            controller.wait_for_change(
                state, min(_POLL_SECONDS, _remaining_seconds(request.phase_deadline_ns))
            )

    def kill_target(_: ControlRequest) -> _SignalDisposition:
        try:
            target.kill()
        except ProcessLookupError:
            return _SignalDisposition.GROUP_ABSENT
        return _SignalDisposition.DELIVERED

    while True:
        kill_state = _wait_for_kill_request(
            controller,
            settle_if_exited=lambda: _windows_target_exited(target),
        )
        if kill_state is None:
            return
        kill_attempt = controller.perform_if_actionable(
            kill_state,
            ControlIntent.KILL,
            kill_target,
        )
        if kill_attempt.disposition is not None:
            break
    while not _windows_target_exited(target):
        state = controller.state()
        controller.wait_for_change(state, _POLL_SECONDS)


def _parse_arguments(arguments: Sequence[str]) -> tuple[Path, tuple[str, ...]]:
    if len(arguments) < 3 or arguments[1] != "--":
        raise ValueError("invalid supervisor arguments")
    status_path = Path(arguments[0])
    command = tuple(arguments[2:])
    if not command:
        raise ValueError("invalid supervisor arguments")
    return status_path, command


def _read_start_gate(file_descriptor: int) -> bytes:
    command = bytearray()
    while len(command) <= MAX_START_LINE_BYTES:
        chunk = os.read(file_descriptor, 1)
        if not chunk:
            break
        command.extend(chunk)
        if chunk == b"\n":
            break
    return bytes(command)


def _take_target_environment() -> dict[str, str]:
    """Consume target-only capabilities before the supervisor spawns helpers."""

    target_only: dict[str, str] = {}
    for key in _TARGET_ONLY_ENVIRONMENT_KEYS:
        value = os.environ.pop(key, None)
        if value is not None:
            target_only[key] = value
    target_environment = os.environ.copy()
    target_environment.update(target_only)
    return target_environment


def _watch_control_pipe(
    control_file_descriptor: int,
    controller: _ShutdownController,
    initial_drain_complete: threading.Event | None = None,
) -> None:
    """Consume bounded frames; optionally drain queued cancellation first."""

    pending = bytearray()
    initial_drain_pending = initial_drain_complete is not None
    try:
        if initial_drain_pending:
            os.set_blocking(control_file_descriptor, False)
        while True:
            try:
                chunk = os.read(
                    control_file_descriptor,
                    MAX_CONTROL_LINE_BYTES + 1,
                )
            except BlockingIOError:
                if initial_drain_pending and not pending:
                    os.set_blocking(control_file_descriptor, True)
                    initial_drain_pending = False
                    assert initial_drain_complete is not None
                    initial_drain_complete.set()
                    continue
                time.sleep(_POLL_SECONDS)
                continue
            if not chunk:
                break
            pending.extend(chunk)
            while b"\n" in pending:
                raw_request, _, remainder = pending.partition(b"\n")
                pending = bytearray(remainder)
                try:
                    controller.apply(ControlRequest.parse(bytes(raw_request)))
                except TypeError, ValueError:
                    controller.fail_protocol()
                    return
            if len(pending) > MAX_CONTROL_LINE_BYTES:
                controller.fail_protocol()
                return
        if pending:
            controller.fail_protocol()
    except OSError:
        controller.fail_protocol()
    finally:
        controller.request_fallback(ControlIntent.TERMINATE)
        if initial_drain_pending:
            assert initial_drain_complete is not None
            initial_drain_complete.set()


def _start_control_watcher(
    control_file_descriptor: int,
    controller: _ShutdownController,
) -> threading.Event:
    """Start the reader and return a receipt for its initial queue drain."""

    initial_drain_complete = threading.Event()
    threading.Thread(
        target=_watch_control_pipe,
        args=(control_file_descriptor, controller, initial_drain_complete),
        name="hbrowser-owner-control",
        daemon=True,
    ).start()
    return initial_drain_complete


def _report_cleanup_failure(error: BaseException) -> None:
    try:
        print(
            "hbrowser process supervisor retained unresolved ownership "
            f"after {type(error).__name__}",
            file=sys.stderr,
            flush=True,
        )
    except BaseException:
        pass


def _park_unresolved_posix_cleanup(
    controller: _ShutdownController,
    error: BaseException,
) -> Never:
    """Never release an identity whose ownership set became unverifiable."""

    _report_cleanup_failure(error)
    state = controller.state()
    while True:
        state = controller.wait_for_change(state, _POLL_SECONDS)


def _complete_posix_cleanup(
    target: subprocess.Popen[bytes],
    *,
    controller: _ShutdownController,
) -> bool:
    """Contain cleanup faults; return true only after fallback proof succeeds."""

    try:
        _terminate_posix_target(target, controller=controller)
        return False
    except BaseException as cleanup_error:
        if isinstance(cleanup_error, _OwnershipProofInvalid):
            _park_unresolved_posix_cleanup(controller, cleanup_error)
        _report_cleanup_failure(cleanup_error)

    state = controller.state()
    while True:
        try:
            if _try_settle_posix_target(target):
                return True
        except BaseException as proof_error:
            if isinstance(proof_error, _OwnershipProofInvalid):
                _park_unresolved_posix_cleanup(controller, proof_error)
        state = controller.wait_for_change(state, _POLL_SECONDS)


def main(arguments: Sequence[str] | None = None) -> int:
    try:
        status_path, command = _parse_arguments(
            tuple(sys.argv[1:] if arguments is None else arguments)
        )
    except OSError, ValueError:
        return PROVEN_TARGET_NOT_STARTED_EXIT_CODE

    control_file_descriptor = sys.stdin.fileno()
    try:
        start_request = StartRequest.parse(_read_start_gate(control_file_descriptor))
    except OSError, TypeError, ValueError:
        _write_status(status_path, "error InvalidStartGate")
        return PROVEN_TARGET_NOT_STARTED_EXIT_CODE

    # Consume target-only capabilities before any functional preflight creates
    # a disposable child or invokes ps. The retained mapping is passed only to
    # the authorized target and cleared on every pre-target return path.
    target_environment = _take_target_environment()
    platform_name = os.name
    controller = _ShutdownController()
    if platform_name == "posix":
        signal.signal(
            signal.SIGINT,
            lambda *_: controller.request_fallback(ControlIntent.TERMINATE),
        )
        signal.signal(
            signal.SIGTERM,
            lambda *_: controller.request_fallback(ControlIntent.TERMINATE),
        )
        if (
            not _posix_exit_receipts_supported()
            or not _posix_process_group_snapshots_supported()
        ):
            target_environment.clear()
            _write_status(status_path, "error UnsupportedOwnershipPrimitive")
            return PROVEN_TARGET_NOT_STARTED_EXIT_CODE
    elif platform_name == "nt":
        pass
    else:
        target_environment.clear()
        _write_status(status_path, "error UnsupportedPlatform")
        return PROVEN_TARGET_NOT_STARTED_EXIT_CODE

    if time.monotonic_ns() >= start_request.deadline_ns:
        target_environment.clear()
        _write_status(status_path, "error StartupDeadlineExpired")
        return PROVEN_TARGET_NOT_STARTED_EXIT_CODE

    try:
        initial_drain_complete = _start_control_watcher(
            control_file_descriptor,
            controller,
        )
    except RuntimeError:
        target_environment.clear()
        _write_status(status_path, "error ControlReaderUnavailable")
        return PROVEN_TARGET_NOT_STARTED_EXIT_CODE
    if not initial_drain_complete.wait(
        timeout=_remaining_seconds(start_request.deadline_ns)
    ):
        target_environment.clear()
        _write_status(status_path, "error StartupDeadlineExpired")
        return PROVEN_TARGET_NOT_STARTED_EXIT_CODE

    try:
        try:
            if platform_name == "posix":
                target, start_error = controller.start_target_if_authorized(
                    deadline_ns=start_request.deadline_ns,
                    action=lambda: subprocess.Popen(
                        command,
                        stdin=subprocess.DEVNULL,
                        close_fds=True,
                        env=target_environment,
                        process_group=0,
                    ),
                )
            else:
                target, start_error = controller.start_target_if_authorized(
                    deadline_ns=start_request.deadline_ns,
                    action=lambda: subprocess.Popen(
                        command,
                        stdin=subprocess.DEVNULL,
                        close_fds=True,
                        env=target_environment,
                    ),
                )
        finally:
            target_environment.clear()
    except (OSError, ValueError) as error:
        _write_status(status_path, f"error {type(error).__name__}")
        return PROVEN_TARGET_NOT_STARTED_EXIT_CODE
    if target is None:
        assert start_error is not None
        _write_status(status_path, f"error {start_error}")
        return PROVEN_TARGET_NOT_STARTED_EXIT_CODE
    execution_error: BaseException | None = None
    try:
        _write_status(status_path, f"ready {target.pid}")

        while True:
            state = controller.state()
            if state.request is not None:
                break
            target_exited = (
                _posix_target_has_exit_receipt(target)
                if platform_name == "posix"
                else _windows_target_exited(target)
            )
            if target_exited:
                controller.request_fallback(ControlIntent.TERMINATE)
                break
            controller.wait_for_change(state, _POLL_SECONDS)
    except BaseException as error:
        execution_error = error
    finally:
        if controller.snapshot() is None:
            controller.request_fallback(ControlIntent.KILL)
        if platform_name == "posix":
            if isinstance(execution_error, _OwnershipProofInvalid):
                _park_unresolved_posix_cleanup(controller, execution_error)
            cleanup_failure_proven = _complete_posix_cleanup(
                target,
                controller=controller,
            )
        else:
            cleanup_failure_proven = False
            _terminate_windows_target(
                target,
                controller=controller,
            )
    if execution_error is not None:
        _report_cleanup_failure(execution_error)
    if execution_error is not None or cleanup_failure_proven:
        return PROVEN_CLEANUP_FAILURE_EXIT_CODE
    if controller.protocol_failed:
        return PROVEN_PROTOCOL_FAILURE_EXIT_CODE
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
