## What changes

## How it is checked

- [ ] CI is green (pytest, build, codecov/patch)
- [ ] `ruff check` and `ruff format --check` pass on the new files
- [ ] `mkdocs build --strict` passes if a page changed
- [ ] Every number in a page or a docstring comes from a script of the repository
- [ ] No `# pragma: no cover` and no absolute path were added
