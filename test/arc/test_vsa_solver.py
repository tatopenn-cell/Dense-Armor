"""Tests for the VSA hypotheses, the abduction and the solver."""

import numpy as np

from dense_armor.core.graph import vsa_solver as solver
from dense_armor.core.graph.grid import Grid
from dense_armor.core.graph.vsa_abduction import (
    ConcreteAction,
    GridSizeHypothesis,
    _candidate_actions,
    abduct,
    action_cost,
    grid_size_hypothesis,
    minimum_hitting_set,
)
from dense_armor.core.graph.vsa_hypotheses import (
    ObjectHypothesis,
    VSAObject,
    cleanup_centre,
    cleanup_colour,
    object_hypothesis_heuristic,
    perceive,
)
from dense_armor.core.graph.vsa_objects import ObjectEncoder
from dense_armor.core.graph.vsa_operations import OperationParams, OperationType

ENC = ObjectEncoder(n=1024)


def _g(rows):
    return Grid.from_list(rows)


# --------------------------------------------------------------
# The six object hypotheses (Table 3)
# --------------------------------------------------------------
def test_the_six_hypotheses_count_the_objects_of_a_small_grid():
    g = _g([[1, 1, 0, 2], [0, 0, 0, 2], [3, 0, 0, 0]])
    count = {h: len(perceive(g, h, ENC)) for h in ObjectHypothesis}
    assert count == {
        ObjectHypothesis.EIGHT_CONNECTED: 4,
        ObjectHypothesis.FOUR_CONNECTED: 4,
        ObjectHypothesis.VERTICAL: 8,
        ObjectHypothesis.HORIZONTAL: 7,
        ObjectHypothesis.COLOUR: 4,
        ObjectHypothesis.PIXEL: 5,
    }


def test_eight_connected_joins_the_diagonal_and_four_connected_does_not():
    g = _g([[1, 0], [0, 1]])
    eight = [
        o for o in perceive(g, ObjectHypothesis.EIGHT_CONNECTED, ENC) if o.colour == 1
    ]
    four = [
        o for o in perceive(g, ObjectHypothesis.FOUR_CONNECTED, ENC) if o.colour == 1
    ]
    assert len(eight) == 1 and len(four) == 2


def test_the_cleanup_finds_the_colour_and_the_centre():
    assert cleanup_colour(ENC.colour_sps[4], ENC) == 4
    assert cleanup_centre(ENC.ssp.encode(np.array([2.0, -3.0])), ENC) == (2, -3)


def test_the_heuristic_ranks_the_six_hypotheses():
    grid_in = _g([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    grid_out = _g([[0, 0, 0], [0, 2, 0], [0, 0, 0]])
    scores = object_hypothesis_heuristic(grid_in, grid_out, ENC)
    assert {h for h, _ in scores} == set(ObjectHypothesis)
    values = [s for _, s in scores]
    assert values == sorted(values, reverse=True) and 0.0 < max(values) <= 1.0


def test_a_hypothesis_without_objects_scores_zero():
    grid_in = _g([[1, 0], [0, 0]])
    grid_out = _g([[0, 0], [0, 0]])
    scores = dict(object_hypothesis_heuristic(grid_in, grid_out, ENC))
    assert scores[ObjectHypothesis.PIXEL] == 0.0


# --------------------------------------------------------------
# The grid size (Table 2)
# --------------------------------------------------------------
def test_grid_size_hypotheses():
    one = _g([[1]])
    two = _g([[1, 1]])
    four = _g([[1, 1, 1, 1]])
    assert grid_size_hypothesis([]) == (GridSizeHypothesis.IDENTITY, None)
    assert grid_size_hypothesis([(one, one)]) == (GridSizeHypothesis.IDENTITY, None)
    assert grid_size_hypothesis([(two, one), (four, one)]) == (
        GridSizeHypothesis.CONSTANT,
        (1, 1),
    )
    assert grid_size_hypothesis([(one, two), (two, four)]) == (
        GridSizeHypothesis.FUNCTION,
        None,
    )


# --------------------------------------------------------------
# The cost of an action set and the minimum hitting set
# --------------------------------------------------------------
RECOLOUR_5 = ConcreteAction(OperationType.RECOLOUR, OperationParams(colour=5))
RECOLOUR_8 = ConcreteAction(OperationType.RECOLOUR, OperationParams(colour=8))
MOVE_DOWN = ConcreteAction(OperationType.MOVE, OperationParams(amount=(1, 0)))


def test_the_cost_counts_the_operations_and_the_distinct_actions():
    assert action_cost({RECOLOUR_5, RECOLOUR_8}) == 3
    assert action_cost({RECOLOUR_5, MOVE_DOWN}) == 4
    assert action_cost(set()) == 0


def test_the_hitting_set_is_the_cheapest_that_hits_every_set():
    assert minimum_hitting_set([]) == set()
    assert minimum_hitting_set([{1, 2}, {2, 3}, {4}]) == {2, 4}
    sets = [{RECOLOUR_5, MOVE_DOWN}, {RECOLOUR_8, MOVE_DOWN}]
    assert minimum_hitting_set(sets, cost=action_cost) == {MOVE_DOWN}


# --------------------------------------------------------------
# The actions that explain an input object and an output object
# --------------------------------------------------------------
def _obj(cells, colour):
    return VSAObject(cells=frozenset(cells), colour=colour, sp=np.zeros(1))


def _ops(in_obj, out_obj):
    grid = _g([[0] * 6 for _ in range(6)])
    return [a.op_type for a in _candidate_actions(in_obj, out_obj, grid, ENC)]


SQUARE = {(1, 1), (1, 2), (2, 1), (2, 2)}
BIG = {(r, c) for r in (1, 2, 3) for c in (1, 2, 3)}
RING = BIG - {(2, 2)}


def test_the_same_object_is_an_identity():
    assert _ops(_obj(SQUARE, 1), _obj(SQUARE, 1)) == [OperationType.IDENTITY]


def test_the_same_cells_with_another_colour_are_a_recolour():
    assert _ops(_obj(SQUARE, 1), _obj(SQUARE, 2)) == [OperationType.RECOLOUR]


def test_the_same_shape_elsewhere_is_a_move_and_a_recentre():
    moved = {(r + 1, c + 1) for r, c in SQUARE}
    ops = _ops(_obj(SQUARE, 1), _obj(moved, 1))
    assert OperationType.MOVE in ops and OperationType.RECENTRE in ops
    assert ops[-1] == OperationType.GENERATE


def test_fill_hollow_and_extract_are_found():
    assert OperationType.FILL in _ops(_obj(RING, 1), _obj(BIG, 1))
    assert OperationType.HOLLOW in _ops(_obj(BIG, 1), _obj(RING, 1))
    cropped = {(r - 1, c - 1) for r, c in SQUARE}
    assert OperationType.EXTRACT in _ops(_obj(SQUARE, 1), _obj(cropped, 1))


def test_a_new_shape_of_the_same_colour_is_a_reshape():
    assert OperationType.RESHAPE in _ops(_obj(SQUARE, 1), _obj({(1, 1), (1, 2)}, 1))


# --------------------------------------------------------------
# The whole solver on a small task: the objects of colour 1 become colour 2
# --------------------------------------------------------------
def _recolour_task():
    def pair(cells):
        grid_in = [[0] * 5 for _ in range(4)]
        grid_out = [[0] * 5 for _ in range(4)]
        for r, c in cells:
            grid_in[r][c] = 1
            grid_out[r][c] = 2
        return {"input": grid_in, "output": grid_out}

    return {
        "train": [pair({(1, 1), (1, 2), (2, 1), (2, 2)}), pair({(1, 2), (2, 2)})],
        "test": [],
    }


def test_abduction_explains_a_recolour_task_with_identity_and_recolour():
    result = abduct(_recolour_task(), ENC)
    assert result is not None
    assert {a.op_type for a in result.action_set} == {
        OperationType.IDENTITY,
        OperationType.RECOLOUR,
    }
    assert abduct({"train": []}, ENC) is None


def test_the_solver_learns_the_recolour_and_applies_it_to_a_new_grid():
    program = solver.solve(_recolour_task(), ENC)
    assert program is not None
    assert program.size_hypothesis == GridSizeHypothesis.IDENTITY
    assert len(program.rules) == 2
    test = _g([[0, 1, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
    assert solver.apply(program, test, ENC).to_list() == [
        [0, 2, 2, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]


def test_a_constant_size_program_renders_into_that_size():
    grid = _g([[0, 0, 0, 0, 0], [0, 3, 3, 0, 0], [0, 3, 3, 0, 0], [0, 0, 0, 0, 0]])
    block = [
        o
        for o in perceive(grid, ObjectHypothesis.EIGHT_CONNECTED, ENC)
        if o.colour == 3
    ][0]
    rule = solver.Rule(
        prototype=block.sp,
        action=ConcreteAction(OperationType.EXTRACT, OperationParams()),
    )
    eight = ObjectHypothesis.EIGHT_CONNECTED
    constant = solver.VSAProgram(eight, GridSizeHypothesis.CONSTANT, (2, 2), [rule])
    assert solver.apply(constant, grid, ENC).to_list() == [[3, 3], [3, 3]]
    function = solver.VSAProgram(eight, GridSizeHypothesis.FUNCTION, None, [rule])
    assert solver.apply(function, grid, ENC).to_list() == [[3, 3], [3, 3]]
    nothing = solver.VSAProgram(eight, GridSizeHypothesis.CONSTANT, (2, 3), [])
    assert solver.apply(nothing, grid, ENC).to_list() == [[0, 0, 0], [0, 0, 0]]
