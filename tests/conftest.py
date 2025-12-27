from __future__ import annotations

import sys
from pathlib import Path


def pytest_configure():
    """Ensure project root is importable as a module during tests.

    This repo isn't packaged as an installed distribution, so we add the
    repository root to sys.path to allow `import app...` from test modules.
    """

    repo_root = Path(__file__).resolve().parent.parent
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
