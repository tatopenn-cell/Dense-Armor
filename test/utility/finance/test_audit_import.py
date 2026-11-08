"""The package import does not put sqlalchemy in sys.modules."""

import subprocess
import sys


def test_package_import_does_not_pull_sqlalchemy():
    code = (
        "import sys\n"
        "import dense_armor.utility.finance\n"
        "assert 'sqlalchemy' not in sys.modules, 'sqlalchemy imported'\n"
        "print('OK')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "OK"
