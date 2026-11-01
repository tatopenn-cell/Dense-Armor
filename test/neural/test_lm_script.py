"""Tests for the N7 language-model benchmark script."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "benchmarks" / "train_small_lm.py"


@pytest.fixture(scope="module")
def smoke_json(tmp_path_factory):
    """Run the script in --smoke mode once; return the parsed JSON."""
    out = tmp_path_factory.mktemp("n7") / "smoke.json"
    proc = subprocess.run(
        [sys.executable, "-u", str(SCRIPT), "--smoke", "--json", str(out)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, (
        f"smoke failed:\nstdout:\n{proc.stdout[-2000:]}\nstderr:\n{proc.stderr[-2000:]}"
    )
    assert out.exists(), "the smoke run did not write the JSON"
    return json.loads(out.read_text())


def test_smoke_writes_every_field(smoke_json):
    """Every field the page and the delivery use is present."""
    required = [
        "label",
        "mode",
        "backend",
        "device_kind",
        "device_count",
        "devices_used",
        "versions",
        "config",
        "batch_size",
        "seq_len",
        "n_steps",
        "accumulate",
        "data",
        "sessions_tokens_per_second",
        "tokens_per_second_median",
        "tokens_per_second_range",
        "val_loss",
        "loss_curve",
        "n_params",
        "generated_greedy",
        "generated_sample",
        "peak_bytes",
        "bigram_val_loss",
        "unigram_val_loss",
        "claim_1_val_over_bigram",
        "claim_1_holds",
        "claim_2_no_nan_inf",
        "claim_4_generated_len",
        "claim_5_holds",
        "claim_5_loss_variance_ratio",
        "loss_variance_first_100",
        "loss_variance_last_100",
        "loss_variance_ratio",
    ]
    missing = [k for k in required if k not in smoke_json]
    assert not missing, f"missing fields: {missing}"


def test_smoke_param_count_matches_config(smoke_json):
    """The reported parameter count is what the config builds.

    Build the same model in this process and count ``Param.value.size``.
    """
    cfg = smoke_json["config"]
    sys.path.insert(0, str(ROOT))
    import jax  # noqa: E402

    from benchmarks.train_small_lm import ModernDecoder  # noqa: E402

    model = ModernDecoder(
        vocab=cfg["vocab"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        max_len=cfg["max_len"],
        d_ff=cfg["d_ff"],
        key=jax.random.PRNGKey(0),
    )
    n_expected = sum(int(p.value.size) for p in model.params())
    assert smoke_json["n_params"] == n_expected, (
        f"JSON says {smoke_json['n_params']}, the config builds {n_expected}"
    )


def test_bigram_loss_on_a_hand_text():
    """A two-symbol alphabet where the bigram is exact by hand.

    Train "abababa" (7 ids, 6 bigrams, all a->b or b->a):
        counts[0,1] = 3 + 1 = 4, counts[0,0] = 0 + 1 = 1
        counts[1,0] = 3 + 1 = 4, counts[1,1] = 0 + 1 = 1
    So P(1 | 0) = 4 / 5. Validation "ab" contains the bigram (0,1), whose
    loss under P is -log(4/5) in nats.
    """
    import numpy as np  # noqa: E402

    sys.path.insert(0, str(ROOT))
    from benchmarks.train_small_lm import _bigram_loss  # noqa: E402

    train = np.array([0, 1, 0, 1, 0, 1, 0], dtype=np.int32)
    val = np.array([0, 1], dtype=np.int32)
    loss = _bigram_loss(train, val)
    expected = -np.log(4.0 / 5.0)
    assert abs(loss - expected) < 1e-9, f"{loss} vs {expected}"


def test_unigram_loss_on_a_hand_text():
    """Unigram with add-1 on "aabb" gives P(a)=P(b)=1/2, loss log 2."""
    import numpy as np  # noqa: E402

    sys.path.insert(0, str(ROOT))
    from benchmarks.train_small_lm import _unigram_loss  # noqa: E402

    train = np.array([0, 0, 1, 1], dtype=np.int32)
    val = np.array([0, 1], dtype=np.int32)
    loss = _unigram_loss(train, val)
    expected = -float(np.log(0.5))
    assert abs(loss - expected) < 1e-9, f"{loss} vs {expected}"


def test_no_nan_in_loss_curve_on_smoke(smoke_json):
    """The smoke run has finite losses, so claim 2 holds on the tiny model."""
    for step, v in smoke_json["loss_curve"].items():
        assert isinstance(step, str)
        assert v == v, f"NaN at step {step}"
        assert abs(v) != float("inf"), f"Inf at step {step}"


def test_generated_text_starts_with_prompt(smoke_json):
    """Greedy and sampled generations both start with the prompt string."""
    prompt = "First Citizen:"
    assert smoke_json["generated_greedy"].startswith(prompt)
    assert smoke_json["generated_sample"].startswith(prompt)
    assert smoke_json["claim_4_generated_len"] == len(smoke_json["generated_greedy"])


def test_smoke_devices_used_is_one(smoke_json):
    """Smoke runs with one device; devices_used is 1, device_count may be more."""
    assert smoke_json["devices_used"] == 1
    assert smoke_json["device_count"] >= 1


def test_smoke_claim_5_is_none_when_windows_overlap(smoke_json):
    """With 10 steps and a window of 100 the two windows overlap.

    claim 5 must be None, not a number: the first and last window are
    the same sequence, so the ratio is 1 by construction and says
    nothing about the run.
    """
    assert smoke_json["n_steps"] < 200
    assert smoke_json["claim_5_holds"] is None
    assert smoke_json["loss_variance_ratio"] is None
    assert smoke_json["claim_5_loss_variance_ratio"] is None
    assert "windows overlap" in smoke_json["claim_5_note"]


def test_loss_curve_keeps_1_100_500():
    """The recorded steps are 1, 100, 500, then every 500.

    A 500-step run records steps 1, 100 and 500 only. A 1000-step run
    records 1, 100, 500, 1000. Step 200 and 300 disappear when the run
    is longer than 500 steps.
    """
    import benchmarks.train_small_lm as lm  # noqa: F401

    src = Path(lm.__file__).read_text()
    assert "(step_i + 1) in (100,)" in src, (
        "the loss curve must record step 100 explicitly"
    )
    assert "(step_i + 1) % 500 == 0" in src, (
        "the loss curve must record every 500 steps after 100"
    )
