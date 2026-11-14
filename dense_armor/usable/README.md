# usable

Give it a file, get the primitives of Dense-Armor that fit it.

## What it does

`usable` reads an object (a file, a folder, a repo, an http address), profiles
it with the primitives of Dense-Armor, and says which classes of the library
can be used for what the file is. The probability `p` comes from the ontology
(graph), not from a list of favourites: it is the share of the weight among
the classes of a job.

## Commands

    python -m dense_armor.usable                       the menu: commands, then `usable>`
    python -m dense_armor.usable list [--domain D] [--role R] [--json] [--refresh]
    python -m dense_armor.usable select JOB [JOB ...] [--budget-ms N] [--kind table|sound] [--top N] [--json] [--why-not NAME]
    python -m dense_armor.usable analyze FILE|URL [--top N] [--json] [--why-not NAME] [--weak]
    python -m dense_armor.usable card OBJECT [--base BASE] [--json]
    python -m dense_armor.usable ontology gaps [--last FILE] [--json]

### Fast start

`usable.cmd`, or `python dense_armor/usable/launch.py`, starts in about 0.3 s: it does not run
`dense_armor/__init__.py`, so JAX is not loaded. `python -m dense_armor.usable` loads the whole
package first and takes about 2 s per command. The commands `card` on a sound, a text or a picture
load the library in both cases, because they use its classes.

## Exit codes

    0   ok, or a deliberate exit (Esc, Ctrl+C, `exit`)
    1   a file was not found or could not be read (or the object is unsupported)
    2   wrong arguments (argparse, or `ontology` without `gaps`)

## Where Dense-Armor comes from

The package is imported from `USABLE_REPO` if set, else from the installed
`dense_armor` package. No path on a single machine is written in the code.
Set `USABLE_REPO` to the Dense-Armor tree when it is not installed:

    export USABLE_REPO=/path/to/Dense-Armor

## The ontology

The knowledge (synonyms, antonyms, concept -> class links, jobs with the
domains and roles they accept, the Italian -> English glossary) lives in
`ontology.json` and `glossary_it.json`. To widen it, add a synonym, an
antonym, a link, a job. Nothing is hardcoded in `select.py`:

- `ontology.json["jobs"]` maps a job to the domains, roles, and modules it
  accepts; `select.py` reads them.
- `ontology.json["links"]` maps a concept to the classes it names (rank 1
  primary, 0.5 secondary).
- `ontology.json["synonyms"]` / `["antonyms"]` carry the signs of the
  concept layer.

## Report the gaps

    python -m dense_armor.usable ontology gaps

prints how many classes of the catalog have no link, and (after an
`analyze`) how many words of the last file have no concept. The formula
is "distinct classes of the catalog minus distinct classes with a link",
not "rows minus distinct": the catalog keeps both `KNNClassifier [core]`
and `KNNClassifier [ml]`, and rows are not classes.

## Tests

    python -m pytest test/usable -q

The suite runs offline, opens no window, reads only `test/usable/samples/`
and `tmp_path`. `ruff` and `mypy` are development tools, not runtime
dependencies: the module runs on the standard library alone.
