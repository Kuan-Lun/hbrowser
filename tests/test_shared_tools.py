from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("preserve_snapshot_attributes", [True, False])
def test_autocrlf_checkout_preserves_snapshot_bytes_and_source_entry(
    tmp_path: Path, preserve_snapshot_attributes: bool
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    shutil.copytree(
        _ROOT / "scripts/_shared",
        source / "scripts/_shared",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    for name in (".gitattributes", "pyproject.toml"):
        shutil.copy2(_ROOT / name, source / name)
    (source / "ordinary.txt").write_bytes(b"first line\nsecond line\n")
    environment = {
        name: value for name, value in os.environ.items() if not name.startswith("GIT_")
    }
    environment.update(
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_CONFIG_SYSTEM=os.devnull,
        GIT_CONFIG_NOSYSTEM="1",
    )

    def git(repository: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ("git", *arguments),
            cwd=repository,
            env=environment,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )

    git(source, "init", "-b", "main")
    git(source, "config", "core.autocrlf", "false")
    git(source, "add", ".")
    git(
        source,
        "-c",
        "user.name=Snapshot Checkout Test",
        "-c",
        "user.email=snapshot-checkout@example.invalid",
        "-c",
        "commit.gpgSign=false",
        "commit",
        "-m",
        "test: commit exact generated snapshot",
    )
    checkout = tmp_path / "checkout with autocrlf"
    git(
        tmp_path,
        "clone",
        "--no-hardlinks",
        "--no-checkout",
        str(source),
        str(checkout),
    )
    git(checkout, "config", "core.autocrlf", "true")
    if not preserve_snapshot_attributes:
        # Remove only this fixture's snapshot protection. Git must really
        # convert the payload; a hand-edited CRLF file is not a checkout test.
        (checkout / ".git/info/attributes").write_bytes(
            b"scripts/_shared/** !text !eol\n"
        )
    git(checkout, "checkout", "--force", "HEAD")
    assert git(checkout, "status", "--porcelain=v1").stdout == ""
    assert (checkout / "ordinary.txt").read_bytes() == b"first line\r\nsecond line\r\n"

    bundle = checkout / "scripts/_shared/h2h_dev_tools"
    expected_hashes: dict[str, str] = json.loads(
        (_ROOT / "scripts/_shared/h2h_dev_tools/manifest.json").read_bytes()
    )["files"]
    actual_hashes = {
        name: hashlib.sha256((bundle / name).read_bytes()).hexdigest()
        for name in expected_hashes
    }
    assert (actual_hashes == expected_hashes) is preserve_snapshot_attributes
    if preserve_snapshot_attributes:
        for original in (_ROOT / "scripts/_shared").rglob("*"):
            if original.is_file() and "__pycache__" not in original.parts:
                relative = original.relative_to(_ROOT)
                assert (checkout / relative).read_bytes() == original.read_bytes()

    loader = (
        "import runpy\n"
        "from pathlib import Path\n"
        "package = Path('scripts/_shared/h2h_dev_tools').resolve()\n"
        "runpy.run_path(str(package / 'source_loader.py'))['load_package'](\n"
        "    package, '_shared.h2h_dev_tools'\n"
        ")\n"
    )
    for command in (
        (str(bundle / "bundle_verify.py"), "--repository", str(checkout)),
        ("-c", loader),
    ):
        result = subprocess.run(
            (sys.executable, "-I", "-B", *command),
            cwd=checkout,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if preserve_snapshot_attributes:
            assert result.returncode == 0, result.stdout + result.stderr
        else:
            assert result.returncode != 0
            assert "Generated source was modified:" in result.stderr
    assert git(checkout, "status", "--porcelain=v1").stdout == ""


@pytest.mark.skipif(
    os.name != "posix", reason="The repository shell gate is POSIX-only"
)
@pytest.mark.parametrize("damage", ("modified", "missing", "unexpected"))
def test_fast_gate_rejects_damaged_snapshot_before_quality_tools(
    tmp_path: Path, damage: str
) -> None:
    repository = tmp_path / "checkout"
    scripts = repository / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy2(_ROOT / "scripts/check-fast.sh", scripts / "check-fast.sh")
    shutil.copytree(_ROOT / "scripts/_shared", scripts / "_shared")
    shutil.copy2(_ROOT / "pyproject.toml", repository / "pyproject.toml")
    bundle = scripts / "_shared/h2h_dev_tools"
    marker = repository / "quality-tool-started"
    executables = {
        ".venv/bin/python": f'exec {shlex.quote(sys.executable)} "$@"\n',
        ".venv/bin/ruff": f"touch {shlex.quote(str(marker))}\n",
        ".venv/bin/mypy": f"touch {shlex.quote(str(marker))}\n",
        "node_modules/.bin/markdownlint-cli2": f"touch {shlex.quote(str(marker))}\n",
    }
    for name, body in executables.items():
        path = repository / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#!/usr/bin/env bash\n" + body, encoding="utf-8")
        path.chmod(0o755)
    environment = {
        name: value for name, value in os.environ.items() if not name.startswith("GIT_")
    }
    environment.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
    subprocess.run(
        ("git", "init", "-b", "main"),
        cwd=repository,
        env=environment,
        capture_output=True,
        check=True,
        timeout=30,
    )

    baseline = subprocess.run(
        ("bash", "scripts/check-fast.sh"),
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert baseline.returncode == 0, baseline.stdout + baseline.stderr
    assert marker.exists()
    marker.unlink()

    if damage == "modified":
        with (bundle / "dependency_manifest.py").open("a", encoding="utf-8") as stream:
            stream.write("\nraise AssertionError('modified snapshot must not run')\n")
    elif damage == "missing":
        (bundle / "dependency_manifest.py").unlink()
    else:
        (bundle / "unexpected.py").write_text("unexpected = True\n", encoding="utf-8")

    result = subprocess.run(
        ("bash", "scripts/check-fast.sh"),
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode != 0
    expected_errors = {
        "modified": "Generated source was modified: dependency_manifest.py",
        "missing": "missing=['dependency_manifest.py']",
        "unexpected": "extra=['unexpected.py']",
    }
    assert expected_errors[damage] in result.stderr
    assert not marker.exists()
