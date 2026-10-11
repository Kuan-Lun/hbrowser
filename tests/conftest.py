import os
import sys
from pathlib import Path

import pytest


@pytest.fixture
def offline_code_review_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise actual review evidence without contacting an external service."""
    binary = tmp_path / "offline-review-cli"
    binary.mkdir()
    executable = binary / "codex"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "if sys.argv[1:] == ['--version']:\n"
        "    print('codex-cli workflow-fixture')\n"
        "    raise SystemExit(0)\n"
        "sys.stdin.read()\n"
        "if os.environ.get('DOCUMENTATION_FAIL_REVIEW'):\n"
        "    raise SystemExit('fixture review requested changes')\n"
        "output = Path(sys.argv[sys.argv.index('--output-last-message') + 1])\n"
        "result = dict(verdict='pass', summary='Reviewed.', findings=[])\n"
        "output.write_text(json.dumps(result))\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(binary) + os.pathsep + os.environ.get("PATH", ""))
