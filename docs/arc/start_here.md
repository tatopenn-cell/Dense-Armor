# Start here

## The task

You have a few `(input, output)` pairs of small grids and a test input. The
rule that turns each input into its output is the same for every pair; apply
it to the test input.

The task of this page is small on purpose:

~~~python
task = {
    "train": [
        {"input": [[1, 1, 0], [1, 0, 0], [0, 0, 2]],
         "output": [[3, 3, 0], [3, 0, 0], [0, 0, 2]]},
    ],
    "test": [{"input": [[1, 1, 0], [1, 0, 2]]}],
}
~~~

The rule is: the shape of colour 1 in the top-left is recoloured to colour
3, and everything else is left alone. Two ways to find it: a program search
over the grid transformations, or a rule synthesis over the objects of the
grid. On this task, the second finds it and the first does not: the rule is
a recolour of one object, not a transformation of the whole grid.

## 1. Try rule synthesis

`synthesize(task)` returns a list of rules. Each rule is a pair
`(selector, effector)`.

~~~python
from dense_armor.core.graph import synthesize

task = {"train": [{"input": [[1, 1, 0], [1, 0, 0], [0, 0, 2]],
                   "output": [[3, 3, 0], [3, 0, 0], [0, 0, 2]]}],
        "test": [{"input": [[1, 1, 0], [1, 0, 2]]}]}
rules = synthesize(task)
print(len(rules))
print(rules[0].selector)
print(rules[0].effector.kind.value)
~~~

~~~text
1
(Predicate(kind=<PredicateType.HAS_COLOR: 'has_color'>, arg=1),)
recolor
~~~

One rule. The selector is the predicate `has_color(1)`: the objects whose
colour is 1. The effector is `recolor`: the objects are recoloured. The
target colour is 3, in `rules[0].effector.to_color`.

## 2. Apply the rules to the test input

`apply_to_test(rules, task)` keeps the training pairs as they are and fills
each test entry with the output of `render` on the test input.

~~~python
from dense_armor.core.graph import apply_to_test, synthesize

task = {"train": [{"input": [[1, 1, 0], [1, 0, 0], [0, 0, 2]],
                   "output": [[3, 3, 0], [3, 0, 0], [0, 0, 2]]}],
        "test": [{"input": [[1, 1, 0], [1, 0, 2]]}]}
sub = apply_to_test(synthesize(task), task)
print(sub["test"][0]["output"])
~~~

~~~text
[[3, 3, 0], [3, 0, 2]]
~~~

The test input has a shape of colour 1, a single cell of colour 2 and a
background. The rule recolours the first object from 1 to 3, and leaves the
second one alone.

## 3. Try the same task with the VSA solver

The VSA solver encodes the objects as vectors and finds the operations that
map the input objects to the output objects. The interface is the same:
`solve` returns a program, `apply` applies it.

~~~python
from dense_armor.core.graph import Grid, ObjectEncoder, vsa_apply, vsa_solve

task = {"train": [{"input": [[1, 1, 0], [1, 0, 0], [0, 0, 2]],
                   "output": [[3, 3, 0], [3, 0, 0], [0, 0, 2]]}],
        "test": [{"input": [[1, 1, 0], [1, 0, 2]]}]}
enc = ObjectEncoder(n=1024)
program = vsa_solve(task, enc)
print(program.hypothesis.value, len(program.rules))
pred = vsa_apply(program, Grid.from_list(task["test"][0]["input"]), enc)
print(pred.to_list())
~~~

~~~text
8-connected 2
[[3, 3, 0], [3, 0, 2]]
~~~

The solver picks `8-connected` as the object hypothesis, finds two rules (a
`RECOLOUR` and an `IDENTITY`), and applies them to the test input. The result
is the same of the rule synthesis on this task. On other tasks the two
solvers give different results.

## 4. What is next

- [Limits](limits.md): what the two solvers do not do. A user who tries them
  on a real task of the benchmark sees some tasks solved and some not.
- [The index of the section](index.md): the three solvers and the links to
  the pages that explain the code.
- [The code pages](../ontology/index.md): for the details of each class and
  each function.

## Details

The three solvers share `Grid` and `ArcObject`. `synthesize` is in
`dense_armor/core/graph/rule_synthesis.py`, `vsa_solve` is in
`dense_armor/core/graph/vsa_solver.py`, `solve` (program search) is in
`dense_armor/core/graph/solver.py`. `apply_to_test` and `to_submission` write
the answer in the format of the benchmark; `render` and `vsa_apply` apply a
rule list or a VSA program to a grid.

### References

The format of a task and of a submission is the one of the ARC-AGI benchmark
(Chollet, "On the Measure of Intelligence", arXiv:1911.01547, Section 2). The
rule synthesis follows the arc-dsl writeup (Hodel, "The ARC-DSL", Section
2.3) for the rule on objects. The VSA solver follows Joffe and Eliasmith
(2025), arXiv:2511.08747, with the prototype variant described in the Limits
of the [VSA page](../ontology/vsa.md).

### Limits

The two solvers do not reach the accuracy of the paper they implement. The
rule synthesis works on tasks whose rule is on objects (a recolour, a
translation, a fill of holes); a task whose rule changes the size of the
grid, or the whole grid, or needs a comparison between objects, is not
solved. The VSA solver does not use the paper's NN predictors, so its
accuracy on ARC-AGI is not the 10.8% of Table 4, page 17 of the paper. See
the [Limits](limits.md) page for the details.
