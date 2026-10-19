#!/usr/bin/env bash
# install.sh -- install Dense-Armor with the drone extra on this board.
#
# Supports Raspberry Pi OS (64-bit) and NVIDIA JetPack. Both are Ubuntu
# derivatives with a Python 3.10 or newer in the system; the script
# creates a virtual environment next to the repository, installs the
# drone extra, and prints the Python version and the module paths.
#
# The commands are the ones a user runs on the board. They are not
# executed by the CI, and they are not executed (no
# board is available). The page docs/drones/start_here.md tells the
# user to run this script after a fresh install of the operating
# system; the "written, not run on the board" statement is in that
# page.
#
# Usage:
#     bash edge/install.sh
#     bash edge/install.sh /path/to/repo
#
set -euo pipefail

REPO="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV="${REPO}/.venv"

if [ ! -f "${REPO}/pyproject.toml" ]; then
    echo "error: ${REPO} does not look like the Dense-Armor repository" >&2
    echo "       (no pyproject.toml found)" >&2
    exit 1
fi

if [ ! -x "$(command -v python3)" ]; then
    echo "error: python3 not found on PATH" >&2
    echo "       on Raspberry Pi OS: sudo apt-get install python3-venv" >&2
    echo "       on JetPack: the system python3 is enough" >&2
    exit 1
fi

echo "==> repository: ${REPO}"
echo "==> python: $(python3 --version)"
echo "==> creating virtual environment in ${VENV}"
python3 -m venv "${VENV}"

echo "==> upgrading pip"
"${VENV}/bin/pip" install --upgrade pip

echo "==> installing dense-armor[drone] in editable mode"
"${VENV}/bin/pip" install -e "${REPO}[drone]"

echo
echo "==> done"
echo "    python:      ${VENV}/bin/python"
echo "    version:     $("${VENV}/bin/python" --version)"
echo "    dense_armor: $("${VENV}/bin/python" -c 'import dense_armor; print(dense_armor.__file__)')"
echo "    pipelines:   $("${VENV}/bin/python" -c 'import pipelines; print(pipelines.__file__)')"
echo
echo "    to activate: source ${VENV}/bin/activate"
echo "    to test:     ${VENV}/bin/python -m pytest test/drones/ -q"
