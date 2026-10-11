from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/smoke-installed-wheel.py"


@pytest.mark.parametrize("editable", [False, True])
def test_smoke_rejects_source_even_with_matching_metadata_and_record(
    tmp_path: Path, editable: bool
) -> None:
    package = tmp_path / "hbrowser"
    package.mkdir()
    (package / "__init__.py").write_text("SOURCE_SENTINEL = True\n")
    metadata = tmp_path / "hbrowser-0.0.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: hbrowser\nVersion: 0.0.0\n"
    )
    (metadata / "RECORD").write_text("hbrowser/__init__.py,,\n")
    if editable:
        (metadata / "direct_url.json").write_text(
            json.dumps({"url": tmp_path.as_uri(), "dir_info": {"editable": True}})
        )
    command = (
        "import runpy,sys; sys.path.insert(0,sys.argv[1]); "
        "runpy.run_path(sys.argv[2],run_name='__main__')"
    )

    result = subprocess.run(
        [sys.executable, "-I", "-c", command, str(tmp_path), str(_SCRIPT)],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert result.returncode != 0
    expected = (
        "must come from a wheel, not an editable checkout"
        if editable
        else "must be installed inside the smoke environment"
    )
    assert expected in result.stderr
    assert "verified" not in result.stdout
