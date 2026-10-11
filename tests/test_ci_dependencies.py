from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import cast

import pytest

_INSTALLER = Path(__file__).parents[1] / "scripts" / "install-ci-dependencies.py"
_COMPONENTS = tomllib.loads(
    (_INSTALLER.parents[1] / "pyproject.toml").read_text(encoding="utf-8")
)["tool"]["h2h-dev-tools"]["components"]


def _fixture_installer(tmp_path: Path, manifest: str) -> Path:
    repository = tmp_path / "checkout"
    scripts = repository / "scripts"
    scripts.mkdir(parents=True)
    installer = scripts / _INSTALLER.name
    installer.write_bytes(_INSTALLER.read_bytes())
    shutil.copytree(
        _INSTALLER.parent / "_shared",
        scripts / "_shared",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    (repository / "pyproject.toml").write_text(
        manifest
        + "\n[tool.h2h-dev-tools]\ncomponents = "
        + json.dumps(_COMPONENTS)
        + "\n",
        encoding="utf-8",
    )
    return installer


def _run_installer(
    installer: Path,
    *arguments: str,
    pip_status: int = 0,
) -> tuple[subprocess.CompletedProcess[str], list[list[str]]]:
    # Invoke the real CLI in a fresh interpreter. Only its child `python -m pip`
    # resolves to this offline recorder; no package is installed or downloaded.
    directory = installer.parents[2] / "offline-pip"
    directory.mkdir()
    calls = directory / "calls.jsonl"
    (directory / "pip.py").write_text(
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "with Path(os.environ['FIXTURE_PIP_CALLS']).open('a') as stream:\n"
        "    stream.write(json.dumps([sys.executable, *sys.argv[1:]]) + '\\n')\n"
        "raise SystemExit(int(os.environ['FIXTURE_PIP_STATUS']))\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        (sys.executable, "-B", str(installer), *arguments),
        cwd=installer.parents[2],
        env=os.environ
        | {
            "PYTHONPATH": str(directory),
            "FIXTURE_PIP_CALLS": str(calls),
            "FIXTURE_PIP_STATUS": str(pip_status),
        },
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    invocations = (
        [
            cast(list[str], json.loads(line))
            for line in calls.read_text(encoding="utf-8").splitlines()
        ]
        if calls.exists()
        else []
    )
    return result, invocations


@pytest.mark.parametrize("pytest_plugins", [False, True])
def test_ci_install_uses_manifest_constraints_and_required_plugins(
    tmp_path: Path, *, pytest_plugins: bool
) -> None:
    installer = _fixture_installer(
        tmp_path,
        """
[project]
dependencies = ['example-package[extra]>=2; python_version >= "3.14"']
[project.optional-dependencies]
dev = ["pytest>=99,!=99.1", "hypothesis>=88,<89", "unrelated-dev-tool>=42"]
[tool.pytest.ini_options]
required_plugins = ["pytest-xdist>=77,<78", "pytest-custom>=66"]
""",
    )
    arguments = ["pytest", "hypothesis", "Example_Package"]
    if pytest_plugins:
        arguments.append("--pytest-plugins")

    result, calls = _run_installer(installer, *arguments)

    assert result.returncode == 0, result.stdout + result.stderr
    expected = [
        "pytest>=99,!=99.1",
        "hypothesis>=88,<89",
        'example-package[extra]>=2; python_version >= "3.14"',
    ]
    if pytest_plugins:
        expected.extend(["pytest-xdist>=77,<78", "pytest-custom>=66"])
    assert calls == [[sys.executable, "install", "--upgrade", *expected]]


def test_ci_install_preserves_intersecting_requirements(tmp_path: Path) -> None:
    installer = _fixture_installer(
        tmp_path,
        """
[project]
dependencies = ["pytest>=9"]
[project.optional-dependencies]
dev = ["pytest<10"]
[tool.pytest.ini_options]
required_plugins = ["pytest>=9"]
""",
    )

    result, calls = _run_installer(installer, "--pytest-plugins", "pytest")

    assert result.returncode == 0, result.stdout + result.stderr
    assert calls == [[sys.executable, "install", "--upgrade", "pytest>=9", "pytest<10"]]


@pytest.mark.parametrize(
    "arguments", [["pytest"], ["--pytest-plugins", "pytest-xdist"]]
)
def test_ci_install_rejects_missing_declarations_before_installing(
    tmp_path: Path, arguments: list[str]
) -> None:
    installer = _fixture_installer(
        tmp_path,
        '[project.optional-dependencies]\ndev = ["pytest-xdist>=77"]\n',
    )

    result, calls = _run_installer(installer, *arguments)

    assert result.returncode == 2, result.stdout + result.stderr
    assert not calls


def test_ci_install_propagates_pip_failure(tmp_path: Path) -> None:
    installer = _fixture_installer(
        tmp_path, '[project.optional-dependencies]\ndev = ["pytest>=99"]\n'
    )

    result, calls = _run_installer(installer, "pytest", pip_status=17)

    assert result.returncode != 0, result.stdout + result.stderr
    assert "returned non-zero exit status 17" in result.stderr
    assert calls == [[sys.executable, "install", "--upgrade", "pytest>=99"]]


def test_publish_workflow_uses_declared_packaging_dependencies() -> None:
    workflow = (
        _INSTALLER.parents[1] / ".github" / "workflows" / "publish.yml"
    ).read_text(encoding="utf-8")

    assert "python scripts/install-ci-dependencies.py packaging" in workflow
    assert "python scripts/install-ci-dependencies.py build" in workflow
    assert "      - scripts/install-ci-dependencies.py\n" in workflow
    assert "      - scripts/_shared/**\n" in workflow
    assert 'python -m pip install --editable ".[dev]"' in workflow
    assert "pip install packaging" not in workflow
    assert "pip install build" not in workflow


@pytest.mark.parametrize("damage", ("selection", "source"))
def test_ci_install_rejects_invalid_snapshot_before_starting_pip(
    tmp_path: Path, damage: str
) -> None:
    manifest = '[project.optional-dependencies]\ndev = ["pytest>=99"]\n'
    installer = _fixture_installer(tmp_path, manifest)
    if damage == "selection":
        (installer.parents[1] / "pyproject.toml").write_text(manifest, encoding="utf-8")
        expected = "Declare [tool.h2h-dev-tools].components"
    else:
        source = installer.parent / "_shared/h2h_dev_tools/ci_dependencies.py"
        source.write_text(
            source.read_text(encoding="utf-8")
            + "\nraise AssertionError('unverified source executed')\n",
            encoding="utf-8",
        )
        expected = "Generated source was modified: ci_dependencies.py"

    result, calls = _run_installer(installer, "pytest")

    assert result.returncode != 0
    assert expected in result.stderr
    assert not calls
