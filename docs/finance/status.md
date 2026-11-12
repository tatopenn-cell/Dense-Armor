# Status: one JSON object that says whether the desk is on

A desk that runs a configuration, a log and a feed wants one question answered:
"is the system ready?" The status command answers with a single JSON object, one
line on stdout. It does not start, stop or change anything: it reads the
configuration, looks at three paths, and reports.

## 1. A valid configuration, no stop file, no log

The command takes one option, `--config`. The configuration is the TOML of F5;
the JSON has one key per block, plus `ok`.

~~~python
import subprocess, sys
from pathlib import Path
Path("/tmp/f8_page/STOP").unlink(missing_ok=True)
Path("/tmp/f8_page/log.jsonl").unlink(missing_ok=True)
cmd = [sys.executable, "-m", "dense_armor.core.finance.status",
       "--config", "/tmp/f8_page/config.toml"]
r = subprocess.run(cmd, capture_output=True, text=True, cwd="/content/Dense-Armor")
print("exit:", r.returncode)
print("stdout:", r.stdout.strip())
~~~

~~~
exit: 0
stdout: {"config": "/tmp/f8_page/config.toml", "emergency_stop": false, "feed": {"exists": false, "path": "/tmp/f8_page/feed.bin"}, "log": {"exists": false, "last_hash": null, "path": "/tmp/f8_page/log.jsonl", "records": 0, "verified": null}, "ok": true}
~~~

`exit: 0` because the configuration is valid. `emergency_stop: false` because
`/tmp/f8_page/STOP` does not exist. `log.exists: false` because nobody wrote a
log yet, and `verified: null` because there is nothing to verify: the command
does not create the log, and `null` distinguishes "not checked" from
"checked and invalid" (`false`).

## 2. The stop file is raised

The stop is a file, not a running process. The command looks for the path in
the `paths` table of the configuration. Here the file `/tmp/f8_page/STOP`
exists.

~~~python
import subprocess, sys
from pathlib import Path
Path("/tmp/f8_page/log.jsonl").unlink(missing_ok=True)
Path("/tmp/f8_page/STOP").write_bytes(b"")
cmd = [sys.executable, "-m", "dense_armor.core.finance.status",
       "--config", "/tmp/f8_page/config.toml"]
r = subprocess.run(cmd, capture_output=True, text=True, cwd="/content/Dense-Armor")
print("exit:", r.returncode)
print("stdout:", r.stdout.strip())
~~~

~~~
exit: 0
stdout: {"config": "/tmp/f8_page/config.toml", "emergency_stop": true, "feed": {"exists": false, "path": "/tmp/f8_page/feed.bin"}, "log": {"exists": false, "last_hash": null, "path": "/tmp/f8_page/log.jsonl", "records": 0, "verified": null}, "ok": true}
~~~

`exit: 0` again, and `ok: true`: the stop being raised is not an error of the
command. The command reports the state of the system, it does not decide that
the system cannot run. The exit code is the same whether the stop is raised or
not.

## 3. A log that verifies, then a log that does not

If the log exists, the command reads it with `DecisionLog`: the number of
records, the last hash, and `verify()`. Here the log has three records.

~~~python
from pathlib import Path
from dense_armor.core.finance.decision_log import DecisionLog
Path("/tmp/f8_page/log.jsonl").unlink(missing_ok=True)
Path("/tmp/f8_page/STOP").unlink(missing_ok=True)
log = DecisionLog("/tmp/f8_page/log.jsonl")
log.append(0.5, {"a": 1})
log.append(1.0, {"a": 2})
log.append(1.5, {"a": 3})

import subprocess, sys
cmd = [sys.executable, "-m", "dense_armor.core.finance.status",
       "--config", "/tmp/f8_page/config.toml"]
r = subprocess.run(cmd, capture_output=True, text=True, cwd="/content/Dense-Armor")
print("exit:", r.returncode)
print("stdout:", r.stdout.strip())
~~~

~~~
exit: 0
stdout: {"config": "/tmp/f8_page/config.toml", "emergency_stop": false, "feed": {"exists": false, "path": "/tmp/f8_page/feed.bin"}, "log": {"exists": true, "last_hash": "3b3c86c860f38053f690f4970d24846b5bcb39d5b851bebcbb09fff8abe86a5e", "path": "/tmp/f8_page/log.jsonl", "records": 3, "verified": true}, "ok": true}
~~~

Edit the second record and run the command again. The log's chain is broken,
but the command does not crash: it reports `verified: false` and exits 0.

~~~python
from pathlib import Path
lines = Path("/tmp/f8_page/log.jsonl").read_text(encoding="utf-8").splitlines(keepends=True)
Path("/tmp/f8_page/log.jsonl").write_text(
    lines[0] + lines[1].replace('"a":2', '"a":99') + lines[2],
    encoding="utf-8", newline="\n",
)

import subprocess, sys
cmd = [sys.executable, "-m", "dense_armor.core.finance.status",
       "--config", "/tmp/f8_page/config.toml"]
r = subprocess.run(cmd, capture_output=True, text=True, cwd="/content/Dense-Armor")
print("exit:", r.returncode)
print("stdout:", r.stdout.strip())
~~~

~~~
exit: 0
stdout: {"config": "/tmp/f8_page/config.toml", "emergency_stop": false, "feed": {"exists": false, "path": "/tmp/f8_page/feed.bin"}, "log": {"exists": true, "last_hash": "3b3c86c860f38053f690f4970d24846b5bcb39d5b851bebcbb09fff8abe86a5e", "path": "/tmp/f8_page/log.jsonl", "records": 3, "verified": false}, "ok": true}
~~~

## Details

**Choices stated explicitly.**
- The exit code is 0 for a valid configuration even when the stop is raised.
  The command reports, it does not decide. A caller that wants to stop on the
  stop file reads `emergency_stop: true` from the JSON and decides.
- A missing or invalid configuration gives exit 2 with
  `{"error": ..., "ok": false}`. A missing `--config` is argparse's own exit
  2 on stderr, and stdout is empty.
- A log that does not exist has `records: 0`, `last_hash: null` and
  `verified: null`. The last one distinguishes "not checked" (no file) from
  "checked and invalid" (`false`, broken chain). The command never creates
  the log.
- The three paths are the ones of the configuration, resolved against the
  folder of the TOML file by `load_config` of F5. The command does not
  resolve them a second time.

**Declared limits.**
- The stop is a file, and the command only checks that the file exists. It
  does not read the file, it does not check a timestamp, and it does not stop
  any process. A caller decides what the file means.
- The command does not send a signal, start a process or write a file. It is
  read-only.
- No network, no database, no broker. The three paths are the only state the
  command looks at.
- The `config` field of the JSON is the path as given on the command line.
  When `--config` is relative, the field is relative, even though the paths
  inside `paths` are absolute.

*These pages describe what the code does on the data shown. Not a forecast, not financial advice.*
