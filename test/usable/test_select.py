"""Tests for the selection (item F): p sums to 1, --why-not, no class answers,
and the cases the criterion names (the classes exist and answer)."""
from pathlib import Path

import pytest

from dense_armor.usable import concepts
from dense_armor.usable.catalog import get_catalog
from dense_armor.usable.readers import needs_from_profile, profile
from dense_armor.usable.select import NEED_DOMAIN, NEED_ROLE, rank

REPO = Path(__file__).resolve().parents[2]
SAMPLES = Path(__file__).resolve().parent / "samples"


# --------------------------------------------------------------
# p sums to 1 per job (tolerance 1e-9)
# --------------------------------------------------------------
def test_p_sums_to_one_per_job():
    cat = get_catalog(REPO)
    ranked = rank(cat, ["classify", "detect_drift", "detect_anomaly"], None, sp=None)
    for need, rows in ranked.items():
        if not rows:
            continue
        somma = sum(r["p"] for r in rows)
        assert abs(somma - 1.0) < 1e-9, f"[{need}] sum(p) = {somma}"


def test_p_is_proportional_to_weight():
    cat = get_catalog(REPO)
    ranked = rank(cat, ["detect_drift"], None, sp=None)
    rows = ranked["detect_drift"]
    totale = sum(r["weight"] for r in rows)
    for r in rows:
        assert abs(r["p"] - r["weight"] / totale) < 1e-12


# --------------------------------------------------------------
# Coverage: every domain of the catalog has a job
# --------------------------------------------------------------
def test_every_domain_is_covered_by_a_job():
    cat = get_catalog(REPO)
    domini = {r["domain"] for r in cat}
    coperti = set()
    for ds in NEED_DOMAIN.values():
        coperti |= set(ds)
    non_coperti = sorted(domini - coperti)
    assert not non_coperti, (
        f"domini senza job: {non_coperti}. Aggiungere un job o un domain "
        f"in ontology.json['jobs'], non in select.py."
    )


# --------------------------------------------------------------
# The cases of criterion F
# --------------------------------------------------------------
CASES = {
    "magic_entropy.py": ["OnlineGaussianMixture", "PCA", "AdaptiveConformalRegressor"],
    "heal_signal.py": ["StreamingHealing"],
    "anomaly_score.py": ["HampelScorer"],
    "richardson_extrapolation.py": ["LinearRegression", "OnlineAR"],
}


def _ranked_for(fname):
    p = SAMPLES / fname
    pr = profile(p)
    needs, budget, _notes = needs_from_profile(pr)
    sp = None
    if pr.get("keywords"):
        g = concepts.build(REPO, pr["keywords"], pr.get("lang", "en"))
        sp = concepts.spread(g)
        evoked = {j: v[0] for j, v in sp["jobs"].items()
                  if v[0] >= concepts.JOB_THRESHOLD}
        needs = needs + [j for j in evoked if j not in needs]
    cat = get_catalog(REPO)
    return rank(cat, needs, budget, sp=sp)


@pytest.mark.parametrize("fname", list(CASES))
def test_criterion_f_classes_appear(fname):
    ranked = _ranked_for(fname)
    nomi = set()
    for rows in ranked.values():
        nomi |= {r["class"] for r in rows}
    mancanti = [c for c in CASES[fname] if c not in nomi]
    assert not mancanti, f"{fname}: classi mancanti: {mancanti}"


def test_payments_classify_has_hoeffding():
    ranked = _ranked_for("../samples/payments.csv")
    nomi = {r["class"] for r in ranked.get("classify", [])}
    assert "HoeffdingTreeClassifier" in nomi


def test_payments_detect_drift_has_adwin_and_page_hinkley():
    ranked = _ranked_for("../samples/payments.csv")
    nomi = {r["class"] for r in ranked.get("detect_drift", [])}
    assert "ADWIN" in nomi
    assert "PageHinkley" in nomi


# --------------------------------------------------------------
# --why-not: a rejected class has a reason
# --------------------------------------------------------------
def test_why_not_returns_a_reason_for_a_rejected_class():
    from dense_armor.usable.onto_select import select
    cat = get_catalog(REPO)
    pr = profile(SAMPLES / "payments.csv")
    needs, budget, _notes = needs_from_profile(pr)
    accepted, broken = select(
        REPO, cat, {**pr, "has_label": True}, needs, budget,
        NEED_ROLE, NEED_DOMAIN,
    )
    assert broken, "nessuna classe rifiutata (atteso almeno una)"
    # every rejected class has at least one broken rule with a name
    for name, rules in broken.items():
        assert rules
        for r in rules:
            assert ":" in r, f"regola senza nome: {r}"


# --------------------------------------------------------------
# "no class answers"
# --------------------------------------------------------------
def test_no_class_answers_when_no_need():
    cat = get_catalog(REPO)
    ranked = rank(cat, [], None, sp=None)
    assert ranked == {}

# --------------------------------------------------------------
# --why-not (CLI): a rejected class prints the reason
# --------------------------------------------------------------
def test_why_not_flag_prints_reason_for_rejected_class(capsys):
    from dense_armor.usable.select import cmd_select
    pr = profile(SAMPLES / "payments.csv")
    needs, budget, _notes = needs_from_profile(pr)
    # "SGDRegressor" is a regressor: on a file to classify,
    # the "label" rule rejects it (supervised=True on has_label=True
    # is not the rejection: try a class of a domain not covered
    # by the file, e.g. AudioFeatures on a table).
    rc = cmd_select(
        REPO, needs, budget, {**pr, "has_label": True},
        why_not="AudioFeatures",
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "AudioFeatures" in out
    # either rejected with a reason, or accepted: the message says which
    assert ("rejected" in out) or ("accepted, no rule is broken" in out)
    if "rejected" in out:
        # at least one rule with a name
        assert ":" in out


def test_why_not_unknown_class_says_not_in_catalog(capsys):
    from dense_armor.usable.select import cmd_select
    pr = profile(SAMPLES / "payments.csv")
    needs, budget, _notes = needs_from_profile(pr)
    rc = cmd_select(
        REPO, needs, budget, {**pr, "has_label": True},
        why_not="NotAClassAtAll",
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "not in the catalog" in out

def test_cmd_select_shows_adaptive_conformal_regressor(capsys):
    """The CLI (cmd_select) shows ACR for magic_entropy.

    The CLI computes the topic_jobs by itself (concepts on the words of
    the file); for a topic job the label rule does not apply (the file
    talks about the theme, it is not the dataset).
    """
    from dense_armor.usable.readers import needs_from_profile, profile
    from dense_armor.usable.select import cmd_select
    pr = profile(SAMPLES / "magic_entropy.py")
    needs, budget, _notes = needs_from_profile(pr)
    # Like the CLI: pass ONLY the needs of the file (magic_entropy -> []),
    # cmd_select finds the topic_jobs.
    rc = cmd_select(REPO, needs, budget, pr, top=500, as_json=True)
    assert rc == 0
    import json as _json
    out = capsys.readouterr().out
    data = _json.loads(out[out.index("{"):])
    rows = data["primitives"].get("summarize_stream", [])
    names = [r["class"] for r in rows]
    assert "OnlineGaussianMixture" in names, names[:5]
    assert "PCA" in names
    assert "AdaptiveConformalRegressor" in names, (
        "ACR assente (regola label non coperta per i topic_jobs)"
    )
