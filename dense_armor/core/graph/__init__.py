"""Graph Engine: Graph, GFT, spectral filters, Chebyshev."""

from dense_armor.core.graph.actions import Action, ActionRejected, apply_action
from dense_armor.core.graph.adacgp import AdaCGP
from dense_armor.core.graph.anomaly import (
    graph_anomaly_score,
    page_rank,
    random_walk_restart,
    root_cause_walk,
)
from dense_armor.core.graph.apply_to_test import apply_to_test
from dense_armor.core.graph.causal import CausalGraph
from dense_armor.core.graph.change import structural_change
from dense_armor.core.graph.core import Graph
from dense_armor.core.graph.dag import ExecutionDAG
from dense_armor.core.graph.dataflow import Dataflow
from dense_armor.core.graph.dsl import (
    dmirror,
    downscale,
    fill,
    hmirror,
    recolor,
    rot90,
    rot180,
    rot270,
    shift,
    upscale,
    vmirror,
)
from dense_armor.core.graph.filters import (
    ChebyshevFilter,
    SpectralFilter,
    apply_filter,
    chebyshev_coefficients,
    chebyshev_filter,
    heat_kernel,
    low_pass,
)
from dense_armor.core.graph.fourier import GraphFourier, gft, igft
from dense_armor.core.graph.gate import GateNode
from dense_armor.core.graph.graph_diff import diff_graphs, hungarian_match, level2_nodes
from dense_armor.core.graph.grid import Grid
from dense_armor.core.graph.grid_graph import build_grid_graph
from dense_armor.core.graph.learning import learn_graph, weights_from_drops
from dense_armor.core.graph.objects import ArcObject, extract_objects
from dense_armor.core.graph.ontology import Constraint, ConstraintViolation, TypedGraph
from dense_armor.core.graph.program import Program
from dense_armor.core.graph.provenance import Provenance
from dense_armor.core.graph.recovery import recover_graph_signal
from dense_armor.core.graph.refine import refine
from dense_armor.core.graph.relations import (
    adjacent,
    distance,
    inside,
    same_color,
    same_shape,
)
from dense_armor.core.graph.rule_synthesis import Rule, render, synthesize
from dense_armor.core.graph.search import explore
from dense_armor.core.graph.signal import GraphSignal
from dense_armor.core.graph.solver import solve
from dense_armor.core.graph.spectral import (
    eigendecomposition,
    fiedler_vector,
    fix_sign,
)
from dense_armor.core.graph.submission import to_submission
from dense_armor.core.graph.vsa import (
    N_DIMENSIONS,
    SSPSpace,
    bind,
    bundle,
    invert,
    random_unit_vector,
    similarity,
    unbind,
)
from dense_armor.core.graph.vsa_abduction import (
    AbductionResult,
    ConcreteAction,
    GridSizeHypothesis,
    abduct,
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
from dense_armor.core.graph.vsa_operations import (
    OperationParams,
    OperationResult,
    OperationType,
    execute_action,
    result_to_grid,
)
from dense_armor.core.graph.vsa_solver import (
    Rule as VSARule,
)
from dense_armor.core.graph.vsa_solver import (
    VSAProgram,
)
from dense_armor.core.graph.vsa_solver import (
    apply as vsa_apply,
)
from dense_armor.core.graph.vsa_solver import (
    solve as vsa_solve,
)
from dense_armor.core.graph.wavelet import SpectralWavelet, chebyshev_wavelet_terms

__all__ = [
    "Graph",
    "GraphSignal",
    "GraphFourier",
    "SpectralFilter",
    "ChebyshevFilter",
    "SpectralWavelet",
    "ExecutionDAG",
    "Dataflow",
    "GateNode",
    "Provenance",
    "TypedGraph",
    "Constraint",
    "ConstraintViolation",
    "Action",
    "ActionRejected",
    "apply_action",
    "graph_anomaly_score",
    "page_rank",
    "random_walk_restart",
    "root_cause_walk",
    "structural_change",
    "AdaCGP",
    "learn_graph",
    "weights_from_drops",
    "recover_graph_signal",
    "chebyshev_wavelet_terms",
    "CausalGraph",
    "Grid",
    "ArcObject",
    "extract_objects",
    "build_grid_graph",
    "downscale",
    "upscale",
    "shift",
    "recolor",
    "fill",
    "dmirror",
    "vmirror",
    "hmirror",
    "rot270",
    "rot180",
    "rot90",
    "Program",
    "to_submission",
    "solve",
    "refine",
    "explore",
    "apply_filter",
    "chebyshev_coefficients",
    "chebyshev_filter",
    "eigendecomposition",
    "fiedler_vector",
    "fix_sign",
    "gft",
    "heat_kernel",
    "igft",
    "low_pass",
    "N_DIMENSIONS",
    "SSPSpace",
    "similarity",
    "bind",
    "unbind",
    "invert",
    "bundle",
    "random_unit_vector",
    "ObjectEncoder",
    "ObjectHypothesis",
    "VSAObject",
    "perceive",
    "cleanup_colour",
    "cleanup_centre",
    "object_hypothesis_heuristic",
    "OperationType",
    "OperationParams",
    "OperationResult",
    "execute_action",
    "result_to_grid",
    "GridSizeHypothesis",
    "grid_size_hypothesis",
    "ConcreteAction",
    "AbductionResult",
    "abduct",
    "minimum_hitting_set",
    "apply_to_test",
    "diff_graphs",
    "hungarian_match",
    "level2_nodes",
    "Rule",
    "synthesize",
    "render",
    "VSAProgram",
    "VSARule",
    "vsa_solve",
    "vsa_apply",
    "adjacent",
    "inside",
    "same_shape",
    "same_color",
    "distance",
]
