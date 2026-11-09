#!/usr/bin/env sh
# Starts the web application of Dense-Armor (the dashboard of the finance system).
# Run it from any folder:  sh tools/dashboard/start-dashboard.sh   (options are passed on, for example --port 9000)
set -e
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/tools:$PWD"
RUNDIR="$PWD/tools/dashboard/run"
mkdir -p "$RUNDIR"
[ -f "$RUNDIR/demo.toml" ] || cp test/fixtures/finance/demo.toml "$RUNDIR/demo.toml"
python3 -c "import flask" 2>/dev/null || python3 -m pip install flask
echo "Dense-Armor dashboard on http://127.0.0.1:8080/   (Ctrl+C stops it)"
echo "The log, the feed and the stop file are in $RUNDIR"
exec python3 -m dashboard.app --config "$RUNDIR/demo.toml" "$@"
