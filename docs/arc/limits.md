# Limits

The three solvers of the [ARC section](index.md) do not solve every task of
ARC-AGI. Some limits are in the code, some are in the paper, some are
intrinsic to the approach. This page lists them, with the reason.

## Limits of rule synthesis

`synthesize` (see [Rule synthesis](../ontology/rule_synthesis.md)) looks for
a list of rules on the objects of the grid: a selector, an effector, and a
target. The rules are chosen by Minimum Description Length from a fixed set
of predicates and effectors.

- **Grid size change.** A task whose output has a different size from the
  input, and whose size is not a constant or a simple function of the
  input, is not solved. `render` works on the size of the input; the size
  change is a separate hypothesis (`grid_size_hypothesis`) that the rule
  synthesis does not use.
- **The whole grid.** A task whose rule is a transformation of the whole
  grid (`rot90`, `hmirror`, `upscale`) is handled by both solvers: the
  program search composes the transformation directly, and the rule
  synthesis applies it as a global effector (`GLOBAL_EFFECTORS` in
  `rule_synthesis.py`). On this kind of task the program search is
  usually simpler to write and to read.
- **Comparison between objects.** A task whose rule is "recolour the
  object with the largest area" is solved (the predicate `is_largest` is in
  the vocabulary). A task whose rule is "recolour the object whose area is
  the median" is expressible with `RANK(k)` only if the number of objects
  is known in advance (the median is `RANK((n+1)//2)` for `n` objects);
  there is no dynamic median. The predicate vocabulary has ten types; a
  rule that needs an eleventh is not expressible.
- **More than three predicates.** A selector is a list of at most three
  predicates. A rule that needs four (`has_color(1) and not touches_border
  and area_gt(2) and is_square`) is not considered.
- **More than five rules.** The search tries combinations of up to five
  rules. A task whose rule is a composition of six rules is not solved.
- **Merge and split.** A rule that merges two objects into one, or that
  splits one object into two, is not expressible. The effectors are on one
  object at a time, or on the whole grid.
- **Multi-coloured objects.** The objects are monochromatic (by the
  definition of `extract_objects` with `by_color=True`). A rule that
  depends on an object with two colours is not expressible.
- **A selector with a predicate on the output.** A rule cannot ask "is the
  area of the input object equal to the area of the output object"; the
  predicates read only the input object.

## Limits of the VSA solver

The VSA solver (`vsa_solve`, see [Vector Symbolic Algebras](../ontology/vsa.md))
encodes objects as vectors and finds the operations that map the input
objects to the output objects. The rule learning uses a prototype per
action.

- **The NN predictors are missing.** The paper (Joffe and Eliasmith 2025,
  arXiv:2511.08747, Section 3.2.2) uses a single-layer NN for the condition
  and one for each parameter, initialised with the outcome-weighted
  superposition of positive objects and trained with SGD. This library uses
  the initial superposition directly, as a prototype, with a threshold of
  0.3 on the cosine similarity. The number it reaches is not the 10.8% of
  Table 4, page 17 of the paper.
- **Property selection is not implemented.** The paper (§3.2.2) selects
  which object properties (`colour`, `centre`, `shape`) are relevant to
  each rule with a leave-one-out cross-validation on the demonstrations.
  This library does not have that step: the rule uses the full VSA of the
  object.
- **`decode_centre` and `decode_shape` do not work with the default
  `length_scale`.** The kernel of the SSP is narrow (similarity 0.016
  between two points half a unit apart), so `argmax` on the similarity does
  not find the centre or the shape of an object. The paper uses an NN for
  the centre (§3.2.2), not `argmax`. The VSA solver does not call these two
  methods during solving; they are available for inspection only.
- **Comparison between objects.** The paper (§5.2) declares that the
  solver cannot conceptualise many-to-one or many-to-many mappings,
  multi-coloured objects, chained operations on the same object, and
  high-level object properties. A task that needs any of these is not
  solved.
- **GRAVITY and GROW are not proposed by abduction.** The candidate
  generation of the abduction (`_candidate_actions`) does not propose them;
  they are available for a user who builds a program by hand.
- **GENERATE builds only rectangles.** A rule that needs a generated
  non-rectangular object is not expressible.
- **The cost of the action set is a reading of the paper.** The paper says
  the minimum hitting set has "minimal cost according to some function f"
  and the function "penalises each additional operation and parameter"; the
  formula is not given. This library uses one point per distinct operation
  plus one point per distinct `(operation, parameter)` pair.

## Limits of program search

`solve` (see [Program search](../ontology/program_search.md)) looks for a
program that is a sequence of grid transformations.

- **The search space is small.** The eleven grid transformations and their
  parameter values are the space. A task whose rule is a transformation of
  an object (a recolour, a translation, a fill of holes) is not in the
  space.
- **The depth is bounded.** The search composes at most `max_depth` steps
  (three by default). A task whose rule is a composition of four steps is
  not solved, unless the beam finds it in the refinement.
- **The refinement is a hill climb.** It accepts only strict improvements;
  a plateau of equal scores is not crossed.
- **One seed per candidate.** The refinement uses `seed + i` for candidate
  `i`; the same seed gives the same result, but a different seed gives a
  different result. Two runs of `solve` with the same `seed` give the same
  answer; two runs with different seeds may not.

## What the library does not do

The three solvers do not use an LLM. The theme of the ARC Prize 2025 report
(arXiv:2601.10904, page 1) is the **refinement loop**: a per-task iterative
optimization of the program guided by a feedback signal, often with an LLM
that proposes candidates. This library has the feedback signal
(`Program.verify`, `VSAProgram.apply`), and it has the search
(`explore`, `refine`, `synthesize`, `vsa_solve`), but the loop that proposes
candidates from outside the fixed vocabulary is not here. The report
separates the exploration (generate candidates) from the verification
(analyse them, produce a feedback signal); the library implements the second
and a fixed-vocabulary version of the first.

The paper itself says (§5.2) that a DSL is not AGI, and that a true AGI
solution must be DSL-open: it must discover new transformations on the fly.
This library is DSL-closed: the transformations and the operations are
fixed.

## Details

The limits above are the ones that a user hits on the training set of
ARC-AGI-1. A task that needs a rule with more than three predicates, a
generated non-rectangular object, a merge of two objects, or a comparison
between objects beyond the ten predicates of the vocabulary, is outside the
space of the three solvers. A task that needs an NN predictor for the
condition and the parameters, as in the paper's §3.2.2, is also outside the
space of the VSA solver of this library: the prototype is the
zero-parameter limit of the initialisation, and the threshold is a choice of
this implementation.

### References

- Chollet, F. (2019), "On the Measure of Intelligence", arXiv:1911.01547:
  the benchmark and its difficulty.
- Joffe, I., Eliasmith, C. (2025), "Vector Symbolic Algebras for the
  Abstraction and Reasoning Corpus", arXiv:2511.08747: Sections 3.2.2 (NN
  predictors) and 5.2 (the declared limits of the paper). Table 4, page 17,
  for the numbers of the paper: 10.8% task accuracy on ARC-AGI-1 training.
- Chollet, F. and others (2025), "ARC Prize 2025: Technical Report",
  arXiv:2601.10904: page 1 for the theme of the year (the refinement loop),
  page 5 for the two phases of the loop (exploration and verification).
- Hodel, M. (2023), "The ARC-DSL", Section 2.3: the eleven grid
  transformations of the program search.

### Choices stated explicitly

- **The prototype is not the NN.** A user who wants to reproduce the paper
  adds the NN predictors of Section 3.2.2; the prototype is the
  zero-parameter limit of the initialisation, and the threshold of 0.3 is
  a choice of this implementation, not the paper's.
- **The vocabulary is fixed.** The three solvers do not generate new
  predicates, new effectors or new transformations. A user who needs one
  writes it in Python and passes it as a primitive (program search) or as a
  predicate (rule synthesis).
- **The size change is a separate hypothesis.** The rule synthesis does
  not change the size of the grid; the grid size hypothesis
  (`grid_size_hypothesis`) is computed but not used by the rule synthesis.
  A user who wants to change the size of the grid uses program search, or
  builds a custom program.
- **The LLM loop is not here.** The refinement loop of the paper is
  implemented without an LLM. The page [Program
  search](../ontology/program_search.md) explains the fixed-vocabulary
  version; the LLM version is a future direction.
