# ARC-AGI

ARC-AGI is a benchmark of tasks that are easy for a person and hard for a
program: a few `(input, output)` pairs of small grids, a rule that turns the
input into the output, and a test input to solve. The colours are ten, the
grids are at most 30 by 30, and no two tasks are alike.

The benchmark is described in Chollet, "On the Measure of Intelligence",
arXiv:1911.01547. Dense-Armor has the pieces a program needs to solve a task:
a `Grid` type, the objects of a grid, the relations between them, a typed
graph that links cells to objects with constraints, a small DSL of
transformations, a search over programs, a search over rules on objects, and a
solver based on Vector Symbolic Algebras.

The sections below describe the three solvers on one task, where the rule is
"recolour every object of colour 1 to colour 3". The full pages for the code
are in [Ontology](../ontology/index.md).

## The task

A task in the format of the benchmark is a Python dict with two lists:

~~~python
task = {
    "train": [
        {"input": [[1, 1, 0], [1, 0, 0], [0, 0, 2]],
         "output": [[3, 3, 0], [3, 0, 0], [0, 0, 2]]},
    ],
    "test": [{"input": [[1, 1, 0], [1, 0, 2]]}],
}
~~~

The rule is simple: the shape of colour 1 in the top-left of the input is
recoloured to colour 3, and everything else is left alone. A solver should
find that rule from the single training pair and apply it to the test input.

## three solvers

Dense-Armor has three solvers, in `dense_armor/core/graph/`. They are
independent; a user can pick any of them, or combine them.

- **Program search** (`solve`, `explore`, `refine`): look for a program that
  is a sequence of grid transformations (`rot90`, `hmirror`, `fill`,
  `recolor`, ...) and that reproduces every training output. Fast on tasks
  whose rule is a global transformation, weak on tasks whose rule is on
  objects (a recolour of the largest shape, say) because the search space of
  programs is too large for a few primitives.
- **Rule synthesis** (`synthesize`, `render`): look for a list of rules on
  the objects of the grid (a selector, an effector) that reproduces every
  training output. Fast on tasks whose rule is on objects, weak on tasks
  whose rule changes the size of the grid or the whole grid.
- **Vector Symbolic Algebras** (`vsa_solve`, `vsa_apply`): encode the
  objects as high-dimensional vectors and find the operations that map the
  input objects to the output objects. Different search strategy from the
  other two; useful when the rule is on the colour, the centre or the shape
  of objects.

For the task of this page, all three find the rule:

~~~python
from dense_armor.core.graph import Grid, ObjectEncoder, apply_to_test, synthesize, vsa_apply, vsa_solve

task = {"train": [{"input": [[1, 1, 0], [1, 0, 0], [0, 0, 2]],
                   "output": [[3, 3, 0], [3, 0, 0], [0, 0, 2]]}],
        "test": [{"input": [[1, 1, 0], [1, 0, 2]]}]}
print(apply_to_test(synthesize(task), task)["test"][0]["output"])

enc = ObjectEncoder(n=1024)
p = vsa_solve(task, enc)
print(vsa_apply(p, Grid.from_list(task["test"][0]["input"]), enc).to_list())
~~~

~~~text
[[3, 3, 0], [3, 0, 2]]
[[3, 3, 0], [3, 0, 2]]
~~~

Both give the expected test output. Program search would not find it here:
the rule is a recolour of one object, not a transformation of the whole grid,
and the search over the eleven grid transformations has no `recolour of
colour 1` in its space.

## Where to start

- [Start here](start_here.md): one task, the three solvers, the output, step by
  step. The page a new user reads first.
- [Limits](limits.md): what the three solvers do not do, with the reasons.
- [Ontology](../ontology/index.md): the code. The pages
  [Grids and objects](../ontology/grids.md),
  [Typed grid graph](../ontology/grid_graph.md),
  [Transformations](../ontology/transformations.md),
  [Program search](../ontology/program_search.md),
  [Rule synthesis](../ontology/rule_synthesis.md) and
  [Vector Symbolic Algebras](../ontology/vsa.md) walk through each piece.

## Details

The code of the three solvers is in `dense_armor/core/graph/`: `grid.py`,
`objects.py`, `relations.py`, `grid_graph.py`, `dsl.py`, `program.py`,
`search.py`, `refine.py`, `solver.py`, `submission.py`, `graph_diff.py`,
`rule_synthesis.py`, `apply_to_test.py`, `vsa.py`, `vsa_objects.py`,
`vsa_hypotheses.py`, `vsa_operations.py`, `vsa_abduction.py`,
`vsa_solver.py`. A program is a sequence of `Grid -> Grid` functions; a rule
is a pair `(selector, effector)`; a VSA program is a list of rules whose
predictor is a prototype. The three share `Grid` and `ArcObject`.

### References

- Chollet, F. (2019), "On the Measure of Intelligence", arXiv:1911.01547:
  the benchmark, its format, the four core knowledge priors.
- Joffe, I., Eliasmith, C. (2025), "Vector Symbolic Algebras for the
  Abstraction and Reasoning Corpus", arXiv:2511.08747: the VSA solver of
  this library is a variant of the paper. The paper reports on Table 4,
  page 17, 57.5% demonstration accuracy and 12.7% query accuracy on
  ARC-AGI-1 training, which is 10.8% task accuracy on the training split.
  This library does not use the paper's NN predictors (Section 3.2.2); it
  uses a prototype, and the number it reaches is not the number of the
  paper.
- Hodel, M. (2023), "The ARC-DSL", Section 2.3: the eleven grid
  transformations of `dsl.py` follow the names and the meaning of the
  writeup.

### Limits

A solver is only as good as the space it searches. On ARC-AGI, the space of
the three solvers is too narrow: a task that needs a merge of two objects, a
task that changes the size of the grid, a task that needs a comparison
between objects, a task with multi-coloured objects, a task with more than
three predicates in the rule, are not solved. The page [Limits](limits.md)
lists each of them with the reason. The benchmark is hard on purpose
(Chollet, arXiv:1911.01547, Section 2); a solver that works on a few tasks
and says so is more useful than a solver that claims to work on all of them.
