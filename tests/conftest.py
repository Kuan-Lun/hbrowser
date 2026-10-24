import os
import re
import runpy
import sys
from pathlib import Path

import pytest

_SHARED_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "scripts/_shared/h2h_dev_tools"
)
sys.path.insert(0, str(_SHARED_DIRECTORY.parents[1]))
runpy.run_path(str(_SHARED_DIRECTORY / "source_loader.py"))["load_package"](
    _SHARED_DIRECTORY, "_shared.h2h_dev_tools"
)


@pytest.fixture(autouse=True)
def isolated_git_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disposable repositories own their config; keep Git-local authority intact."""
    for name in tuple(os.environ):
        if name in {
            "GIT_CONFIG",
            "GIT_CONFIG_PARAMETERS",
            "GIT_CONFIG_COUNT",
        } or re.fullmatch(r"GIT_CONFIG_(?:KEY|VALUE)_\d+", name):
            monkeypatch.delenv(name)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


@pytest.fixture
def offline_code_review_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise actual review evidence without contacting an external service."""
    from _shared.h2h_dev_tools import review_test_support
    from _shared.h2h_dev_tools.executable_fixture import write_python_executable

    response_source = review_test_support.synthetic_review_response_source()
    binary = tmp_path / "offline-review-cli"
    binary.mkdir()
    executable = binary / "codex"
    write_python_executable(
        executable,
        f"{response_source}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "if sys.argv[1:] == ['--version']:\n"
        "    print('codex-cli workflow-fixture')\n"
        "    raise SystemExit(0)\n"
        "prompt = sys.stdin.read()\n"
        "if os.environ.get('DOCUMENTATION_FAIL_REVIEW'):\n"
        "    raise SystemExit('fixture review requested changes')\n"
        "output = Path(sys.argv[sys.argv.index('--output-last-message') + 1])\n"
        "result = synthetic_review_response(prompt)\n"
        "output.write_text(json.dumps(result, ensure_ascii=False, "
        'separators=(",", ":")), encoding="utf-8")\n',
    )
    monkeypatch.setenv("PATH", str(binary) + os.pathsep + os.environ.get("PATH", ""))
