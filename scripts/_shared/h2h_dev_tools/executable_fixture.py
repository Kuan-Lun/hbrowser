"""Create disposable Python command fixtures without interpreter shebang limits."""

from __future__ import annotations

import os
import shlex
import sys
import tempfile
from pathlib import Path


def _windows_launcher(source: str, interpreter: Path) -> bytes:
    # Distlib ships no typing metadata; this is its documented script API.
    from distlib.scripts import ScriptMaker, enquote_executable  # type: ignore[import-untyped]

    with tempfile.TemporaryDirectory(prefix="h2h-python-fixture-") as temporary:
        directory = Path(temporary)
        outputs = directory / "output"
        outputs.mkdir()
        maker = ScriptMaker(None, str(outputs))
        maker.executable = enquote_executable(str(interpreter))
        maker.variants = {""}
        # Generate directly from the in-memory template. Distlib's file-copy
        # path retains an open input in its traceback when writing fails,
        # preventing Windows from cleaning the temporary directory. The export
        # selects the output name; our literal template replaces its stub.
        maker.script_template = ("#!/usr/bin/env python\n" + source).replace("%", "%%")
        generated = maker.make("fixture = builtins:exec")
        executable = outputs / "fixture.exe"
        if generated != [str(executable)] or not executable.is_file():
            raise RuntimeError("Distlib did not generate the expected native launcher")
        return executable.read_bytes()


def write_python_executable(
    path: Path,
    source: str,
    *,
    python: str | Path | None = None,
    windows_cmd: bool = False,
) -> Path:
    """Write a disposable fixture and return the actual executable path.

    POSIX writes a quoted shell launcher plus adjacent ``<name>.py`` payload;
    ``exec`` preserves the launcher PID and the payload is ``sys.argv[0]`` and
    ``__file__``. Windows uses distlib's native console launcher (``.exe``), with
    source in its embedded ``__main__.py``; ``sys.argv[0]`` is the executable and
    ``__file__`` is that embedded source path. Callers must not infer the command
    name from these platform-specific identities. Default native Windows
    launchers preserve arguments without a command shell. ``windows_cmd=True``
    adds and returns an ASCII npm-style ``.cmd`` wrapper and therefore uses batch
    argument semantics. Python startup flags and stdin are unchanged. POSIX
    launcher/payload pairs may move together; standalone launcher symlinks are
    not supported. Only the explicit launcher/payload/wrapper paths are replaced.
    """
    target = path.absolute()
    interpreter = Path(sys.executable if python is None else python).absolute()
    if os.name == "nt":
        wrapper = None
        if windows_cmd:
            wrapper = (
                target
                if target.suffix.lower() == ".cmd"
                else target.with_suffix(".cmd")
                if target.suffix.lower() == ".exe"
                else target.with_name(target.name + ".cmd")
            )
            launcher = wrapper.with_suffix(".exe")
        else:
            launcher = (
                target
                if target.suffix.lower() == ".exe"
                else target.with_name(target.name + ".exe")
            )
        content = _windows_launcher(source, interpreter)
        target.parent.mkdir(parents=True, exist_ok=True)
        launcher.write_bytes(content)
        if wrapper is not None:
            wrapper.write_text(
                "@echo off\r\n"
                "setlocal DisableDelayedExpansion\r\n"
                '"%~dpn0.exe" %*\r\n'
                "exit /b %errorlevel%\r\n",
                encoding="ascii",
                newline="",
            )
            return wrapper
    else:
        launcher = target
        payload = target.with_name(target.name + ".py")
        command = f'#!/bin/sh\nexec {shlex.quote(str(interpreter))} "$0.py" "$@"\n'
        target.parent.mkdir(parents=True, exist_ok=True)
        payload.write_text(source, encoding="utf-8", newline="")
        payload.chmod(0o644)
        launcher.write_text(command, encoding="utf-8", newline="")
        launcher.chmod(0o755)
    return launcher
