"""Build a candidate wheel and run a repository-owned, offline smoke callback.

Only the candidate is installed in the temporary environment. Dependencies are
borrowed explicitly from the repository's development site, so this is artifact
validation and never evidence of clean dependency resolution.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory


def _python(environment: Path) -> Path:
    return environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _site(python: Path, *, cwd: Path) -> Path:
    result = subprocess.run(
        [
            str(python),
            "-I",
            "-c",
            "import sysconfig; print(sysconfig.get_path('purelib'))",
        ],
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    )
    return Path(result.stdout.strip())


def run_offline_smoke(
    repository: Path, smoke_script: Path, arguments: Sequence[str] = ()
) -> None:
    """Own temporary build/install resources while the consumer owns its oracle."""
    repository = repository.resolve(strict=True)
    callback = (repository / smoke_script).resolve(strict=True)
    if not callback.is_relative_to(repository) or not callback.is_file():
        raise ValueError("Smoke callback must be a file inside the repository")
    development_python = _python(repository / ".venv")
    if not development_python.is_file():
        raise ValueError(f"Missing repository-local Python: {development_python}")
    development_site = _site(development_python, cwd=repository)
    with TemporaryDirectory(prefix="h2h-offline-wheel-") as temporary:
        root = Path(temporary)
        dist = root / "dist"
        environment = os.environ | {
            "UV_CACHE_DIR": str(root / "uv-cache"),
            "UV_OFFLINE": "1",
        }
        subprocess.run(
            [
                str(development_python),
                "-m",
                "build",
                "--no-isolation",
                "--outdir",
                str(dist),
            ],
            cwd=repository,
            env=environment,
            check=True,
        )
        wheels = list(dist.glob("*.whl"))
        if len(wheels) != 1 or not wheels[0].is_file() or wheels[0].is_symlink():
            raise ValueError("Build must produce exactly one regular wheel")
        virtual_environment = root / "smoke-venv"
        subprocess.run(
            [
                "uv",
                "venv",
                "--python",
                str(development_python),
                str(virtual_environment),
            ],
            cwd=root,
            env=environment,
            check=True,
        )
        smoke_python = _python(virtual_environment)
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(smoke_python),
                "--no-deps",
                str(wheels[0]),
            ],
            cwd=root,
            env=environment,
            check=True,
        )
        smoke_site = _site(smoke_python, cwd=root)
        (smoke_site / "development-dependencies.pth").write_text(
            f"import site; site.addsitedir({str(development_site)!r})\n",
            encoding="utf-8",
        )
        print(
            "Offline wheel smoke borrows development dependencies; "
            "it does not validate clean dependency resolution.",
            flush=True,
        )
        subprocess.run(
            [str(smoke_python), "-I", str(callback), *arguments],
            cwd=root,
            check=True,
        )


def main(arguments: list[str] | None = None) -> int:
    """Run one callback with its arguments after an optional -- separator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--smoke-script", required=True, type=Path)
    parser.add_argument("callback_arguments", nargs=argparse.REMAINDER)
    options = parser.parse_args(arguments)
    callback_arguments = options.callback_arguments
    if callback_arguments[:1] == ["--"]:
        callback_arguments = callback_arguments[1:]
    try:
        run_offline_smoke(options.repository, options.smoke_script, callback_arguments)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"offline-wheel: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
