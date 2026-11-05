"""Native pins for private-directory generations and cleanup-worker handoff."""

from __future__ import annotations

import array
import ctypes
import os
import socket
import stat
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast


class DirectoryIdentityError(RuntimeError):
    """A pinned directory generation cannot be verified or transferred."""


class _WindowsDirectoryApi:
    _FILE_READ_ATTRIBUTES = 0x80
    _SHARE_READ_WRITE_DELETE = 0x7
    _OPEN_EXISTING = 3
    _OPEN_DIRECTORY_WITHOUT_FOLLOWING_REPARSE = 0x02000000 | 0x00200000
    _DUPLICATE_SAME_ACCESS = 2
    _FILE_STANDARD_INFO = 1
    _FILE_ID_INFO = 18

    def __init__(self) -> None:
        from ctypes import wintypes

        class FileStandardInfo(ctypes.Structure):
            _fields_ = [
                ("AllocationSize", ctypes.c_longlong),
                ("EndOfFile", ctypes.c_longlong),
                ("NumberOfLinks", wintypes.DWORD),
                ("DeletePending", ctypes.c_ubyte),
                ("Directory", ctypes.c_ubyte),
            ]

        class FileIdInfo(ctypes.Structure):
            _fields_ = [
                ("VolumeSerialNumber", ctypes.c_ulonglong),
                ("FileId", ctypes.c_ubyte * 16),
            ]

        loader = getattr(ctypes, "WinDLL", None)
        if loader is None:
            raise DirectoryIdentityError("Native Windows directory APIs unavailable")
        self.kernel32: Any = loader("kernel32", use_last_error=True)
        self.standard_info_type = FileStandardInfo
        self.file_id_type = FileIdInfo
        self.kernel32.CreateFileW.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            ctypes.c_void_p,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        self.kernel32.CreateFileW.restype = wintypes.HANDLE
        self.kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel32.CloseHandle.restype = wintypes.BOOL
        self.kernel32.GetCurrentProcess.argtypes = []
        self.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        self.kernel32.DuplicateHandle.argtypes = [
            wintypes.HANDLE,
            wintypes.HANDLE,
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.HANDLE),
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        ]
        self.kernel32.DuplicateHandle.restype = wintypes.BOOL
        self.kernel32.GetFileInformationByHandleEx.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        self.kernel32.GetFileInformationByHandleEx.restype = wintypes.BOOL

    @staticmethod
    def error(operation: str) -> OSError:
        get_error = cast(Any, getattr(ctypes, "get_last_error", lambda: 0))
        code = int(get_error())
        return OSError(code, f"{operation} failed with Windows error {code}")

    def open(self, path: Path) -> int:
        handle = self.kernel32.CreateFileW(
            str(path),
            self._FILE_READ_ATTRIBUTES,
            self._SHARE_READ_WRITE_DELETE,
            None,
            self._OPEN_EXISTING,
            self._OPEN_DIRECTORY_WITHOUT_FOLLOWING_REPARSE,
            None,
        )
        if handle is None or handle == ctypes.c_void_p(-1).value:
            raise self.error("CreateFileW directory pin")
        return int(handle)

    def close(self, handle: int) -> None:
        if not self.kernel32.CloseHandle(handle):
            raise self.error("CloseHandle directory pin")

    def duplicate(
        self, handle: int, *, source: int | None = None, inheritable: bool = False
    ) -> int:
        from ctypes import wintypes

        result = wintypes.HANDLE()
        current = self.kernel32.GetCurrentProcess()
        if not self.kernel32.DuplicateHandle(
            current if source is None else source,
            handle,
            current,
            ctypes.byref(result),
            0,
            inheritable,
            self._DUPLICATE_SAME_ACCESS,
        ):
            raise self.error("DuplicateHandle directory pin")
        assert result.value is not None
        return int(result.value)

    def deleted(self, handle: int) -> bool:
        information = self.standard_info_type()
        if not self.kernel32.GetFileInformationByHandleEx(
            handle,
            self._FILE_STANDARD_INFO,
            ctypes.byref(information),
            ctypes.sizeof(information),
        ):
            raise self.error("GetFileInformationByHandleEx directory deletion")
        if not information.Directory:
            raise DirectoryIdentityError("Native directory pin is not a directory")
        return bool(information.DeletePending)

    def identity(self, handle: int) -> tuple[int, int]:
        information = self.file_id_type()
        if not self.kernel32.GetFileInformationByHandleEx(
            handle,
            self._FILE_ID_INFO,
            ctypes.byref(information),
            ctypes.sizeof(information),
        ):
            # CPython fstat may fall back to a 64-bit ID. That fallback is not
            # unique on every Windows filesystem, so it cannot authorize removal.
            raise self.error("GetFileInformationByHandleEx full directory identity")
        return int(information.VolumeSerialNumber), int.from_bytes(
            bytes(information.FileId), "little"
        )


def _windows_descriptor(handle: int, api: _WindowsDirectoryApi) -> int:
    try:
        import msvcrt

        convert = getattr(msvcrt, "open_osfhandle", None)
        noinherit = getattr(os, "O_NOINHERIT", None)
        if not callable(convert) or noinherit is None:
            raise DirectoryIdentityError(
                "Non-inheritable Windows fd conversion unavailable"
            )
        return cast(Callable[[int, int], int], convert)(handle, os.O_RDONLY | noinherit)
    except BaseException:
        api.close(handle)
        raise


def _windows_handle(descriptor: int) -> int:
    import msvcrt

    convert = getattr(msvcrt, "get_osfhandle", None)
    if not callable(convert):
        raise DirectoryIdentityError("Windows fd HANDLE conversion unavailable")
    return cast(Callable[[int], int], convert)(descriptor)


class DirectoryPin:
    """One native generation shared by path aliases until explicit retirement.

    Finalization only closes a handle; it is never filesystem cleanup proof.
    Owners and pending-cleanup registries retain live pins across failures.
    """

    def __init__(self) -> None:
        # Allocate all Python coordination state before accepting a descriptor.
        # A retained constructor traceback must not keep an acquired fd alive.
        self._descriptor: int | None = None
        self._retired = False
        self._delegate: DirectoryPin | None = None
        self._lock = threading.RLock()
        self.device = 0
        self.inode = 0

    @classmethod
    def _adopt_descriptor(
        cls, descriptor: int, *, windows_api: _WindowsDirectoryApi | None = None
    ) -> DirectoryPin:
        pin = None
        try:
            pin = cls()
            os.set_inheritable(descriptor, False)
            metadata = os.fstat(descriptor)
            if not stat.S_ISDIR(metadata.st_mode) or (
                getattr(metadata, "st_file_attributes", 0) & 0x400
            ):
                raise DirectoryIdentityError("Native pin is not an ordinary directory")
            pin.device = metadata.st_dev
            pin.inode = metadata.st_ino
            if os.name == "nt":
                api = windows_api if windows_api is not None else _WindowsDirectoryApi()
                native_identity = api.identity(_windows_handle(descriptor))
                if native_identity != (pin.device, pin.inode):
                    raise DirectoryIdentityError(
                        "Directory stat does not match its full native identity"
                    )
            pin._descriptor = descriptor
            return pin
        except BaseException:
            # This is the sole failure-close owner, including cls.__new__ and
            # lock allocation. Disarm any retained object before closing once.
            if pin is not None:
                pin._descriptor = None
            os.close(descriptor)
            raise

    @classmethod
    def capture(cls, path: Path) -> DirectoryPin:
        api = _WindowsDirectoryApi() if os.name == "nt" else None
        descriptor = (
            _windows_descriptor(api.open(path), api)
            if api is not None
            else os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        )
        return cls._adopt_descriptor(descriptor, windows_api=api)

    @classmethod
    def inherit(
        cls, native: int, *, windows_api: _WindowsDirectoryApi | None = None
    ) -> DirectoryPin:
        if os.name == "nt":
            # The worker prepares its API while the inherited-handle inventory
            # still owns the resource; this method then consumes it exactly once.
            if windows_api is None:
                raise DirectoryIdentityError("Inherited HANDLE has no prepared API")
            descriptor = _windows_descriptor(native, windows_api)
        else:
            descriptor = native
        return cls._adopt_descriptor(descriptor, windows_api=windows_api)

    @staticmethod
    def close_inherited(
        native: int, *, windows_api: _WindowsDirectoryApi | None = None
    ) -> None:
        if os.name == "nt":
            if windows_api is None:
                raise DirectoryIdentityError("Inherited HANDLE has no prepared API")
            windows_api.close(native)
        else:
            os.close(native)

    @classmethod
    def from_worker(cls, native: int, process_handle: int) -> DirectoryPin:
        if os.name != "nt":
            raise DirectoryIdentityError("Remote HANDLE transfer requires Windows")
        api = _WindowsDirectoryApi()
        descriptor = _windows_descriptor(
            api.duplicate(native, source=process_handle), api
        )
        return cls._adopt_descriptor(descriptor, windows_api=api)

    @classmethod
    def completed(cls, device: int, inode: int) -> DirectoryPin:
        pin = cls()
        pin._retired = True
        pin.device, pin.inode = device, inode
        return pin

    @property
    def authority(self) -> DirectoryPin:
        pin = self
        while pin._delegate is not None:
            pin = pin._delegate
        return pin

    def share_authority(self, other: DirectoryPin) -> None:
        """Coalesce pins owned by one admitted cleanup set, keeping all aliases."""
        authority = self.authority
        redundant = other.authority
        if authority is redundant:
            return
        first, second = sorted((authority, redundant), key=id)
        with first._lock, second._lock:
            authority._require_descriptor()
            descriptor = redundant._require_descriptor()
            if (authority.device, authority.inode) != (
                redundant.device,
                redundant.inode,
            ):
                raise DirectoryIdentityError(
                    "Cannot share different native generations"
                )
            # Both descriptors still pin this generation, so matching native IDs
            # cannot be a reuse. Keep one pin throughout the redundant close.
            # An ambiguous close leaves the other wrapper failed, not delegated.
            redundant._descriptor = None
            os.close(descriptor)
            redundant._delegate = authority

    @property
    def retired(self) -> bool:
        authority = self.authority
        if authority is not self:
            return authority.retired
        with self._lock:
            return self._retired

    def _require_descriptor(self) -> int:
        if self._descriptor is None:
            raise DirectoryIdentityError("Directory pin no longer has native authority")
        return self._descriptor

    def deleted(self) -> bool:
        """Return positive native deletion evidence, or a retired authority.

        False is inconclusive: some filesystems retain a directory's link count
        after rmdir. It must never authorize adopting a replacement generation.
        """
        authority = self.authority
        if authority is not self:
            return authority.deleted()
        with self._lock:
            if self._retired:
                return True
            descriptor = self._require_descriptor()
            if os.name == "nt":
                return _WindowsDirectoryApi().deleted(_windows_handle(descriptor))
            return os.fstat(descriptor).st_nlink == 0

    def duplicate_descriptor(self) -> int:
        authority = self.authority
        if authority is not self:
            return authority.duplicate_descriptor()
        with self._lock:
            descriptor = self._require_descriptor()
            if os.name == "posix":
                import fcntl

                return fcntl.fcntl(descriptor, fcntl.F_DUPFD_CLOEXEC, 3)
            return os.dup(descriptor)

    @contextmanager
    def exported(self) -> Iterator[int]:
        authority = self.authority
        if authority is not self:
            with authority.exported() as native:
                yield native
            return
        with self._lock:
            descriptor = self._require_descriptor()
            api = _WindowsDirectoryApi() if os.name == "nt" else None
            if api is not None:
                native = api.duplicate(_windows_handle(descriptor), inheritable=True)
            else:
                native = self.duplicate_descriptor()
        try:
            yield native
        finally:
            if api is not None:
                api.close(native)
            else:
                os.close(native)

    def native_handle(self) -> int:
        """Expose a still-held worker HANDLE for parent-side DuplicateHandle."""
        if os.name != "nt":
            raise DirectoryIdentityError("Native HANDLE requires Windows")
        authority = self.authority
        if authority is not self:
            return authority.native_handle()
        with self._lock:
            return _windows_handle(self._require_descriptor())

    def retire(self) -> None:
        authority = self.authority
        if authority is not self:
            authority.retire()
            return
        with self._lock:
            if self._retired:
                return
            descriptor = self._require_descriptor()
            # Never retry close against a potentially reused descriptor after an
            # ambiguous close error. Such a pin remains failed, not completed.
            self._descriptor = None
            os.close(descriptor)
            self._retired = True

    def __del__(self) -> None:
        descriptor = getattr(self, "_descriptor", None)
        if descriptor is not None:
            self._descriptor = None
            try:
                os.close(descriptor)
            except OSError:
                pass


def send_pin(channel: socket.socket, pin: DirectoryPin) -> None:
    with pin.exported() as descriptor:
        descriptors = array.array("i", [descriptor])
        sent = channel.sendmsg(
            [b"P"], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, descriptors)]
        )
        if sent != 1:
            raise DirectoryIdentityError("Incomplete directory pin transfer")


def receive_pin(channel: socket.socket) -> DirectoryPin:
    descriptors = array.array("i")
    data, ancillary, flags, _address = channel.recvmsg(
        1, socket.CMSG_SPACE(2 * descriptors.itemsize)
    )
    try:
        unexpected = False
        for level, kind, payload in ancillary:
            if level != socket.SOL_SOCKET or kind != socket.SCM_RIGHTS:
                unexpected = True
            else:
                descriptors.frombytes(
                    payload[: len(payload) - len(payload) % descriptors.itemsize]
                )
        if (
            unexpected
            or data != b"P"
            or flags & socket.MSG_CTRUNC
            or len(descriptors) != 1
        ):
            raise DirectoryIdentityError("Invalid directory pin transfer")
        descriptor = descriptors.pop()
        return DirectoryPin._adopt_descriptor(descriptor)
    finally:
        for descriptor in descriptors:
            os.close(descriptor)
