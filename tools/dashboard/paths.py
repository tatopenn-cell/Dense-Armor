"""Where the files of the dashboard are, whatever folder the program was started from.

The desktop icon of the installer starts the program in ``tools/dashboard``, the batch file in the
root of the repository, a test in a temporary folder. A path written as ``test/fixtures/...`` is
right in the second case only, so every path to a file of the repository is built from here.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "test" / "fixtures" / "finance"
