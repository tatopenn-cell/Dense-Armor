"""Tests for what the commands print: the answers, the evidence, the launcher and the prompt."""

import json
import subprocess
import sys
from pathlib import Path

from dense_armor.usable import cli, evaluate, keywords, onto_select, readers

HERE = Path(__file__).resolve().parent
SAMPLES = HERE / "samples"
LAUNCH = Path(cli.__file__).resolve().parent / "launch.py"


def run_cli(capsys, *argv) -> tuple[int, str]:
    code = cli.main(list(argv))
    return code, capsys.readouterr().out


# --------------------------------------------------------------
# analyze: a table, a signal, a function
# --------------------------------------------------------------
def test_analyze_table_names_jobs_and_classes(capsys):
    code, out = run_cli(capsys, "analyze", str(SAMPLES / "payments.csv"))
    assert code == 0
    assert "classify the rows" in out
    assert "notice when the data changes" in out
    for name in ("ADWIN", "PageHinkley", "HoeffdingTreeClassifier"):
        assert name in out


def test_forecast_lists_only_time_series_classes(capsys):
    code, out = run_cli(capsys, "select", "forecast", "--top", "1000", "--json")
    rows = json.loads(out)["primitives"]["forecast"]
    assert code == 0
    assert 0 < len(rows) <= 12
    assert {r["domain"] for r in rows} <= {"timeseries", "kalman", "dynamics"}


def test_analyze_function_file_uses_the_ontology(capsys):
    code, out = run_cli(capsys, "analyze", str(SAMPLES / "heal_signal.py"))
    assert code == 0
    assert "StreamingHealing" in out


def test_code_that_calls_a_class_names_it_as_evidence(capsys):
    code, out = run_cli(capsys, "analyze", str(Path(onto_select.__file__)))
    assert "by the code: it calls" in out
    assert "Constraint" in out


# --------------------------------------------------------------
# weak matches, data classes, texts
# --------------------------------------------------------------
def test_weak_matches_only_with_the_option(capsys):
    target = str(Path(cli.__file__).resolve().parent / "card.py")
    _, plain = run_cli(capsys, "analyze", target)
    _, weak = run_cli(capsys, "analyze", target, "--weak")
    assert "weak match by words" not in plain
    assert "weak match by words" in weak
    cli.render.SHOW_WEAK = False


def test_classes_that_only_declare_fields_ask_nothing(tmp_path, capsys):
    f = tmp_path / "schemas.py"
    f.write_text(
        "from pydantic import BaseModel, Field, model_validator\n\n\n"
        "class OscillatingNoiseInput(BaseModel):\n"
        "    amp: float = Field(default=0.1)\n\n\n"
        "class RunInput(BaseModel):\n"
        "    p: float = Field(default=0.0)\n\n"
        "    @model_validator(mode='after')\n"
        "    def _noise_needs_a_model(self):\n"
        "        return self\n\n\n"
        "def smooth_noise(x):\n"
        '    """Reduce the noise of a signal."""\n'
        "    return x\n",
        encoding="utf-8",
    )
    _, out = run_cli(capsys, "analyze", str(f))
    assert "only declares data or checks it" not in out
    assert "OscillatingNoiseInput" not in out
    assert "AdaptiveSignalStabilizer" in out
    _, full = run_cli(capsys, "analyze", str(f), "--full")
    cli.render.FULL = False
    assert full.count("only declares data or checks it") == 3


def test_template_statements_are_not_words(tmp_path):
    f = tmp_path / "feed.html"
    f.write_text(
        '{% extends "base.html" %}{% block c %}<p>info feed error</p>'
        "{% if error %}<b>error</b>{% endif %}{% endblock %}",
        encoding="utf-8",
    )
    text, _, _ = readers._extract_text(f, False)
    for word in ("endif", "extends", "block"):
        assert word not in text
    assert "error" in text


def test_text_that_calls_for_no_task_says_so(tmp_path, capsys):
    f = tmp_path / "notes.md"
    f.write_text(
        "The meeting covered the budget of the next quarter and the plan "
        "for the offices. " * 40,
        encoding="utf-8",
    )
    code, out = run_cli(capsys, "analyze", str(f))
    assert code == 0
    assert "do not call for a task of the library" in out


def test_language_of_a_text():
    assert (
        keywords.detect_language("the file is read and the words are counted") == "en"
    )
    assert (
        keywords.detect_language("il file viene letto e le parole sono contate") == "it"
    )


# --------------------------------------------------------------
# the evaluation on the samples
# --------------------------------------------------------------
def test_evaluate_prints_a_total(capsys):
    code = evaluate.main()
    out = capsys.readouterr().out
    assert code == 0
    assert "TOTAL" in out


# --------------------------------------------------------------
# the launcher and the prompt, as a user runs them
# --------------------------------------------------------------
def launch(*args, stdin=None):
    return subprocess.run(
        [sys.executable, str(LAUNCH), *args],
        input=stdin,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def test_launcher_lists_the_drift_detectors():
    r = launch("list", "--domain", "drift")
    assert r.returncode == 0
    assert "5 primitives" in r.stdout
    assert r.stderr.strip() == ""


def test_launcher_card_of_a_folder_loads_the_library_of_the_repo():
    r = launch("card", str(SAMPLES), "--json")
    assert r.returncode == 0, r.stderr[-300:]
    assert json.loads(r.stdout)["summary"]["n_files"] >= 10


def test_prompt_session_from_a_pipe_leaves_quietly():
    session = (
        "list drift\n"
        f"file {SAMPLES / 'payments.csv'}\n"
        "weak\nweak\n"
        f"card {SAMPLES / 'note.txt'}\n"
        "ontology gaps\n"
        "foo\n"
        "exit\n"
    )
    r = launch(stdin=session)
    assert r.returncode == 0
    assert r.stderr.strip() == ""
    assert "unknown command: foo" in r.stdout
    assert "weak matches: shown" in r.stdout
    assert "weak matches: hidden" in r.stdout
    assert "classes without a link" in r.stdout


def test_prompt_starts_with_weak_matches_hidden():
    card = Path(cli.__file__).resolve().parent / "card.py"
    r = launch(stdin=f"file {card}\nexit\n")
    assert r.returncode == 0
    assert "weak match by words" not in r.stdout
    assert "by the code: it calls" in r.stdout
