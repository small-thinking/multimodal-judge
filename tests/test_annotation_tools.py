"""Run the standalone annotation-tool regression suites in the existing CI job."""

from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("directory", ["annotation_tools", "annotation_tools/annotator"])
def test_standalone_annotation_tools(directory):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", directory, "-p", "test_*.py"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
