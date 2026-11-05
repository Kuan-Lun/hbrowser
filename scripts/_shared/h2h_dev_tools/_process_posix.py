"""Keep native group identity valid through observation, termination and cleanup."""

from __future__ import annotations

import ctypes
import errno
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, BinaryIO, cast


class IdentityError(RuntimeError):
    """The original child or its complete native cleanup proof is unavailable."""


class AcquisitionError(IdentityError):
    """A launched child still needs its existing owner's bounded cleanup."""

    def __init__(self, process: PinnedProcess) -> None:
        self.process = process
        super().__init__(
            f"Cannot retain kernel group identity for owned PID {process.pid}; "
            "cleanup is unproven"
        )


_PIDFD_SIGNAL_PROCESS_GROUP = 4  # Linux 6.9: scope the retained struct pid as PGID.
_PROC_PGRP_ONLY = 2


def _pidfd_open(pid: int) -> int:
    opener = cast(Callable[[int], int] | None, getattr(os, "pidfd_open", None))
    if opener is None:
        raise ValueError("Linux ownership requires pidfd_open")
    return opener(pid)


def _pidfd_signal(descriptor: int, number: int) -> None:
    sender = cast(
        Callable[[int, int, None, int], None] | None,
        getattr(signal, "pidfd_send_signal", None),
    )
    if sender is None:
        raise ValueError("Linux ownership requires pidfd_send_signal")
    sender(descriptor, number, None, _PIDFD_SIGNAL_PROCESS_GROUP)


def _darwin_group_members(group: int) -> tuple[int, ...]:
    """A two-slot, kernel-locked list of live AND zombie PGID members.

    XNU proc_listpids holds proc_list_lock across both lists. A one-record result
    is therefore complete, unlike /proc or ps enumeration. A two-record result
    only means more than one member; no full inventory allocation is needed.
    """
    try:
        library: Any = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        query = library.proc_listpids
    except (OSError, AttributeError) as error:
        raise IdentityError("Native libproc inventory is unavailable") from error
    query.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_int]
    query.restype = ctypes.c_int
    members = (ctypes.c_int * 2)()
    ctypes.set_errno(0)
    size = query(_PROC_PGRP_ONLY, group, members, ctypes.sizeof(members))
    if size not in (ctypes.sizeof(ctypes.c_int), ctypes.sizeof(members)):
        raise IdentityError(
            f"Native group inventory is unavailable: bytes={size}, errno={ctypes.get_errno()}"
        )
    result = tuple(members[: size // ctypes.sizeof(ctypes.c_int)])
    if any(pid <= 0 for pid in result) or len(set(result)) != len(result):
        raise IdentityError("Native group inventory contains invalid identities")
    return result


def _preflight_native_identity() -> None:
    if sys.platform == "darwin":
        try:
            _darwin_group_members(os.getpgrp())
        except IdentityError as error:
            raise ValueError(
                "macOS ownership requires native libproc inventory"
            ) from error
    elif sys.platform == "linux":
        descriptor: int | None = None
        try:
            descriptor = _pidfd_open(os.getpid())
            try:
                _pidfd_signal(descriptor, 0)
            except ProcessLookupError:
                # This process need not lead a group. ESRCH still establishes
                # support for the PGID scope; an older kernel returns EINVAL.
                pass
        except OSError as error:
            if error.errno in {errno.EINVAL, errno.ENOSYS}:
                raise ValueError(
                    "Linux ownership requires Linux 6.9+ pidfd process-group signalling"
                ) from error
            raise
        finally:
            if descriptor is not None:
                os.close(descriptor)
    else:
        raise ValueError("POSIX ownership requires macOS or Linux 6.9+ native identity")


class PinnedProcess(subprocess.Popen[bytes]):
    """Exit status observations do not surrender the owned group identity.

    On macOS a waitable leader pins the numeric PGID through the final locked
    inventory. Linux transfers that responsibility to a pidfd before reaping;
    subsequent operations use that kernel object, never the reused group number.
    The caller closes admission and proves nested obligations empty first.
    """

    def __init__(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        env: dict[str, str] | None,
        stdout: BinaryIO | int | None,
        stderr: BinaryIO | int | None,
        start_new_session: bool,
    ) -> None:
        if not start_new_session or not all(
            hasattr(os, name) for name in ("waitid", "P_PID", "WEXITED", "WNOWAIT")
        ):
            raise ValueError(
                "POSIX ownership requires waitid(WNOWAIT) and a new session"
            )
        if signal.getsignal(signal.SIGCHLD) != signal.SIG_DFL:
            raise ValueError("POSIX ownership requires default SIGCHLD handling")
        _preflight_native_identity()  # Reject unsupported platforms before spawn.
        self._identity_lock = threading.RLock()
        self._blocking_observers: set[object] = set()
        self._identity_retired = False
        self._leader_reaped = False
        self._pidfd: int | None = None
        self._pidfd_acquisition_failed = False
        self._close_error: OSError | None = None
        super().__init__(
            command,
            cwd=cwd,
            env=env,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        if sys.platform == "linux":
            try:
                self._pidfd = _pidfd_open(self.pid)
            except OSError as error:
                # The target may already own nested sessions. Only its existing
                # scope owner may close admission and arrange bounded cleanup;
                # killing this group here could destroy those nested owners.
                self._pidfd_acquisition_failed = True
                raise AcquisitionError(self) from error

    def poll(self) -> int | None:
        with self._identity_lock:
            if self._identity_retired or self._leader_reaped:
                return self.returncode
            try:
                receipt = os.waitid(
                    os.P_PID, self.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT
                )
            except ChildProcessError as error:
                raise IdentityError(
                    "Owned group leader was reaped outside its owner"
                ) from error
            if receipt is None:
                if self.returncode is not None:
                    raise IdentityError("Owned exit receipt disappeared")
                return None
            if receipt.si_pid != self.pid:
                raise IdentityError("Owned exit receipt names another process")
            if receipt.si_code in (
                os.CLD_STOPPED,
                os.CLD_CONTINUED,
                os.CLD_TRAPPED,
            ):
                # Darwin can report a stopped child despite WEXITED. A live
                # transition is not an exit or permission to release identity.
                if self.returncode is not None:
                    raise IdentityError("Owned exit receipt disappeared")
                return None
            if receipt.si_code == os.CLD_EXITED:
                code = receipt.si_status
            elif receipt.si_code in (os.CLD_KILLED, os.CLD_DUMPED):
                code = -receipt.si_status
            else:
                raise IdentityError("Unexpected owned exit receipt")
            if self.returncode is not None and self.returncode != code:
                raise IdentityError("Owned exit status changed")
            self.returncode = code
            return code

    def _internal_poll(self, _deadstate: int | None = None) -> int | None:
        del _deadstate
        return self.poll()

    def wait(self, timeout: float | None = None) -> int:
        if timeout is None:
            return self._wait_unbounded()
        deadline = time.monotonic() + timeout
        delay = 0.0005
        while True:
            code = self.poll()
            if code is not None:
                return code
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(self.args, timeout)
            # Keep fast commands responsive without busy-spinning on long ones.
            delay = min(delay * 2, 0.05, remaining)
            time.sleep(delay)

    def _wait_unbounded(self) -> int:
        while True:
            observer = object()
            try:
                with self._identity_lock:
                    # Reserve the waitable PID before releasing the lock. Even
                    # a waiter paused before waitid must prevent retirement and
                    # numeric PID reuse until its observation has finished.
                    self._blocking_observers.add(observer)
                    code = self.poll()
                    if code is not None:
                        return code
                try:
                    os.waitid(os.P_PID, self.pid, os.WEXITED | os.WNOWAIT)
                except ChildProcessError as error:
                    raise IdentityError(
                        "Owned group leader was reaped outside its owner"
                    ) from error
                # The blocking receipt is only a wake-up: another waiter may
                # already have observed a later state. Revalidate under the
                # lock rather than recording a potentially stale STOP receipt.
                code = self.poll()
            finally:
                with self._identity_lock:
                    self._blocking_observers.discard(observer)
            if code is not None:
                return code
            # Darwin may repeatedly return a nonterminal receipt with WEXITED
            # and WNOWAIT. Back off only for that case, never for ordinary exits.
            time.sleep(0.05)

    def _wait(self, timeout: float | None) -> int:
        return self.wait(timeout)

    def send_signal(self, sig: int) -> None:
        with self._identity_lock:
            if self._identity_retired or self.poll() is not None:
                return
            try:
                os.kill(self.pid, sig)
            except ProcessLookupError:
                pass

    def __del__(self) -> None:
        # Popen's opportunistic global reap must never release a still-owned
        # group identity. Explicit owner cleanup is the only retirement path.
        pass

    def signal_group(self, number: int, *, send: Callable[[int, int], None]) -> None:
        with self._identity_lock:
            if self._identity_retired:
                return
            self.poll()  # Detect a foreign reap before sending any signal.
            if self._pidfd is not None:
                _pidfd_signal(self._pidfd, number)
            else:
                send(self.pid, number)

    def _reap_leader(self) -> None:
        if self._leader_reaped:
            return
        code = self.poll()
        if code is None:
            raise IdentityError("Owned leader is still running")
        try:
            pid, status = os.waitpid(self.pid, os.WNOHANG)
        except ChildProcessError as error:
            raise IdentityError(
                "Owned leader identity disappeared before retirement"
            ) from error
        if pid != self.pid or os.waitstatus_to_exitcode(status) != code:
            raise IdentityError(
                "Owned leader retirement did not match its exit receipt"
            )
        self._leader_reaped = True

    def retire_clean_group(self, *, deadline: float) -> bool:
        with self._identity_lock:
            if self._identity_retired:
                return True
            if self._blocking_observers:
                return False
            if self._pidfd_acquisition_failed:
                # A waitable leader permits safe signals, but does not establish
                # Linux group absence. Never fall back to Darwin inventory or
                # reap the only identity still retained after failed acquisition.
                return False
            if self.poll() is None or time.monotonic() >= deadline:
                return False
            if self._pidfd is not None:
                self._reap_leader()
                try:
                    _pidfd_signal(self._pidfd, 0)
                except ProcessLookupError:
                    pass  # Original kernel PGID has no members, even after reuse.
                except OSError as error:
                    raise IdentityError("Kernel group absence is unproven") from error
                else:
                    return False
            else:
                if _darwin_group_members(self.pid) != (self.pid,):
                    return False
                if time.monotonic() >= deadline:
                    return False
                self._reap_leader()
            self._identity_retired = True
            return time.monotonic() < deadline

    def close_identity(self) -> None:
        with self._identity_lock:
            if self._close_error is not None:
                raise IdentityError(
                    "Native identity handle closure is unproven"
                ) from self._close_error
            if not self._identity_retired:
                raise IdentityError("Cannot close an unproven group identity")
            descriptor = self._pidfd
            if descriptor is not None:
                # Do not retry a possibly reused descriptor after close failure.
                self._pidfd = None
                try:
                    os.close(descriptor)
                except OSError as error:
                    self._close_error = error
                    raise IdentityError(
                        "Native identity handle closure is unproven"
                    ) from error
