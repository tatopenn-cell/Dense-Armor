"""Tests for dense_armor.core.graph.solver."""

from dense_armor.core.graph import dsl
from dense_armor.core.graph.grid import Grid
from dense_armor.core.graph.solver import solve


def _hmirror_task():
    return {
        "train": [
            {"input": [[1, 2], [3, 4]], "output": [[2, 1], [4, 3]]},
        ],
        "test": [{"input": [[5, 6], [7, 8]]}],
    }


def test_solve_simple_task():
    task = _hmirror_task()
    program = solve(
        task,
        [dsl.hmirror, dsl.rot90, dsl.vmirror],
        max_depth=2,
        beam_width=3,
        refine_iterations=5,
    )
    pairs = [
        (Grid.from_list([[1, 2], [3, 4]]), Grid.from_list([[2, 1], [4, 3]])),
    ]
    assert program.verify(pairs) == 1.0


def test_solve_empty_train_returns_empty_program():
    program = solve({"train": [], "test": []}, [dsl.hmirror])
    assert program.steps == ()


def test_solve_refines_every_candidate_in_beam():
    """Il solver non deve buttar via i candidati del beam: raffina ognuno."""
    from dense_armor.core.graph import dsl

    task = {
        "train": [
            {"input": [[1, 2], [3, 4]], "output": [[2, 1], [4, 3]]},
        ],
        "test": [],
    }
    program = solve(
        task,
        [dsl.hmirror, dsl.rot90, dsl.vmirror, dsl.rot180],
        max_depth=2,
        beam_width=4,
        refine_iterations=10,
        seed=0,
    )
    pairs = [
        (Grid.from_list([[1, 2], [3, 4]]), Grid.from_list([[2, 1], [4, 3]])),
    ]
    assert program.verify(pairs) == 1.0
